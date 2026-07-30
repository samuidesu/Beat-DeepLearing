"""Full DeepLab-v3+ model: dilated ResNet -> ASPP encoder -> decoder.

Forward pass:
    image [B, 3, H, W]
      --backbone-->  c2 [B, 64, H/4, W/4]   low-level feature (crisp edges)
                     c5 [B, 512, H/8, W/8]  high-level feature (stride 8)
      --neck------>  enc [B, 256, H/4, W/4] ASPP context, projected + 2x upsampled
      --head------>  logits [B, 21, H, W]   fuse enc + c2, classify, 4x upsample

The model returns RAW logits (no softmax) at full input resolution. Training
feeds them straight into nn.CrossEntropyLoss(ignore_index=255); inference
takes argmax over dim 1 to get the per-pixel class-id mask.

DeepLab-v3+ = v3's ASPP encoder + a small DECODER. v3 upsampled the stride-8
logits straight to full resolution (an 8x jump), blurring boundaries. v3+ adds
a decoder that fuses the encoder output with a stride-4 low-level backbone
feature (c2) before a gentler 4x upsample, so thin structures and object edges
come out sharper. See neckV3.py (encoder) and headV3.py (decoder).

Two-stage finetune protocol:
    Stage 1: freeze only the LOW backbone (stem/layer1/layer2); train the
             dilation-modified HIGH stages (layer3/layer4) together with the
             new neck/head. Rationale: layer3/4's geometry was surgically
             changed (stride -> dilation), so their pretrained weights need to
             adapt anyway -- no point keeping them frozen while the neck/head
             warm up on features that are themselves about to shift.
    Stage 2: unfreeze the ENTIRE backbone and finetune end-to-end with
             LAYERED learning rates (low backbone slowest, high backbone
             middle, neck/head fastest) -- the deeper you are into pretrained
             territory, the gentler the updates.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import torch
import torch.nn as nn

# Package-relative imports when used as `model.deeplab`; plain imports when
# this file is run directly as a script. This project is DeepLab-v3+ only, so
# it needs just the dilated backbone, the ASPP encoder and the decoder.
try:
    from .backbone import ResNetBackbone
    from .neckV3 import DeepLabV3ASPP
    from .headV3 import DeepLabHeadV3
except ImportError:
    from backbone import ResNetBackbone
    from neckV3 import DeepLabV3ASPP
    from headV3 import DeepLabHeadV3


class DeepLab(nn.Module):
    """ResNet-backbone DeepLab-v3+ for PASCAL VOC 2012 semantic segmentation.

    Encoder (backbone + ASPP neck) + a small decoder head:
        * backbone: dilated ResNet, output stride 8. Its layer1 output (c2,
          stride 4) is the crisp low-level feature the decoder fuses in; its
          final output (c5, stride 8) feeds the ASPP.
        * neck (neckV3.DeepLabV3ASPP): ASPP context -> 1x1 project to 256 ->
          upsample x2 to stride 4.
        * head (headV3.DeepLabHeadV3): reduce c2, concat with the encoder
          output, 3x3 fuse, classify, 4x upsample to input resolution.

    Args:
        num_classes: 21 for VOC seg (20 object classes + background).
        pretrained: load ImageNet-pretrained backbone weights.
        backbone: backbone arch, "resnet18" or "resnet34".
        aspp_rates: ASPP atrous branch dilations (3, 6, 9); a 1x1 branch and a
            global-pooling branch are added automatically.
        aspp_hidden: per-branch feature width in the ASPP (256).
        low_level_proj: channels the decoder reduces the low-level feature (c2)
            to before fusion (48, the DeepLab-v3+ default).
        dropout: spatial dropout inside the neck and decoder (0 disables).
    """

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = True,
        backbone: str = "resnet34",
        aspp_rates: Sequence[int] = (3, 6, 9),
        aspp_hidden: int = 256,
        low_level_proj: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Backbone -> c2 (stride 4, low-level) + c5 (stride 8, high-level).
        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)

        # Encoder neck: ASPP over c5 -> projected features at stride 4.
        self.neck = DeepLabV3ASPP(
            in_channels=self.backbone.out_channels,
            hidden_channels=aspp_hidden,
            rates=aspp_rates,
            dropout=dropout,
        )

        # Decoder head: fuse the encoder output with the stride-4 low-level
        # feature c2, classify, and upsample to input resolution.
        self.head = DeepLabHeadV3(
            aspp_channels=self.neck.out_channels,          # 256
            low_level_channels=self.backbone.stage_channels[0],  # c2 = 64
            low_level_proj=low_level_proj,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the segmenter.

        Input:
            x: image batch [B, 3, H, W] (H, W ideally multiples of 8 so the
                stride-8 grid is exact; the eval pipeline pads to guarantee it).

        Output:
            raw per-pixel class logits [B, num_classes, H, W].
        """
        # Remember the input's spatial size: the head interpolates the logits
        # back to EXACTLY this size, so prediction and label mask always align.
        input_size = x.shape[-2:]  # (H, W)

        # Need BOTH the low-level feature (c2) and the high-level one (c5).
        feats = self.backbone(x, return_intermediate=True)
        c2 = feats["c2"]                       # [B, 64,  H/4, W/4]
        c5 = feats["c5"]                       # [B, 512, H/8, W/8]

        enc = self.neck(c5)                    # [B, 256, H/4, W/4]
        logits = self.head(c2, enc, input_size)  # [B, 21, H, W]
        return logits

    # ---- Two-stage finetuning helpers ---------------------------------------
    def freeze_backbone_low(self):
        """Stage 1: freeze ONLY stem/layer1/layer2.

        The dilation-modified high stages (layer3/layer4) stay trainable
        alongside the new neck/head -- their pretrained weights must adapt to
        the new atrous geometry anyway.
        """
        self.backbone.freeze_low_layers()

    def unfreeze_backbone_all(self):
        """Stage 2: unfreeze the ENTIRE backbone for the layered-LR finetune."""
        self.backbone.unfreeze()

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Split parameters into the three layered-LR groups.

        Output (dict of lists, ready for torch.optim param_groups):
            "backbone_low":  stem + layer1 + layer2   (slowest LR -- pristine
                             pretrained low-level features)
            "backbone_high": layer3 + layer4          (middle LR -- pretrained
                             but geometry-modified)
            "neck_head":     neck + head              (fastest LR -- trained
                             from scratch)

        Frozen parameters (requires_grad=False) are EXCLUDED, so the same
        call works for both stages: in stage 1 "backbone_low" simply comes
        back empty.
        """
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone_low": [
                p for p in self.backbone.low_level_parameters() if p.requires_grad
            ],
            "backbone_high": [
                p for p in self.backbone.high_level_parameters() if p.requires_grad
            ],
            "neck_head": [
                p
                for m in (self.neck, self.head)
                if m is not None
                for p in m.parameters()
                if p.requires_grad
            ],
        }
        return groups

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call this AFTER model.train() each epoch. model.train() flips every BN
        back to training mode, which would let frozen layers keep updating
        their running mean/var -- undesirable. This restores eval mode for any
        BN whose affine weights are frozen, preserving the pretrained stats.
        (In stage 1 that is stem/layer1/layer2's BNs; in stage 2, none.)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # A frozen BN has requires_grad == False on its weight.
                if m.weight is not None and not m.weight.requires_grad:
                    m.eval()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the parameters that currently require gradients."""
        return (p for p in self.parameters() if p.requires_grad)


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/deeplab.py
if __name__ == "__main__":
    # 416 is a multiple of 8, so H/8 -> H/4 -> H stays exact through the
    # encoder's x2 and the decoder's 4x upsamples.
    dummy = torch.randn(2, 3, 416, 416)

    # pretrained=False avoids a network download for this shape check.
    model = DeepLab(num_classes=21, pretrained=False)
    out = model(dummy)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[v3+] logits: {tuple(out.shape)} "
        f"(expected (2, 21, 416, 416))  total params={n_total/1e6:.2f}M"
    )

    # Stage-1 freeze check: low backbone frozen, high backbone + neck/head on.
    # (parameter_groups() validates the v3+ neck AND decoder params land in the
    # neck_head group.)
    model.freeze_backbone_low()
    groups = model.parameter_groups()
    for name, params in groups.items():
        n = sum(p.numel() for p in params) / 1e6
        print(f"stage-1 group {name}: {n:.2f}M trainable")

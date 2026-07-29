"""Full DeepLab-v1-style model: dilated ResNet -> LargeFOV neck -> head.

Forward pass:
    image [B, 3, H, W]
      --backbone-->  c5   [B, 512, H/8, W/8]   dilated ResNet, output stride 8
      --neck------>  feat [B, 128, H/8, W/8]   LargeFOV atrous context (rate 12)
      --head------>  logits [B, 21, H, W]      1x1 classify + 8x bilinear resize

The model returns RAW logits (no softmax) at full input resolution. Training
feeds them straight into nn.CrossEntropyLoss(ignore_index=255); inference
takes argmax over dim 1 to get the per-pixel class-id mask.

Contrast with this repo's FCN project -- same task, opposite strategy:

    FCN:     backbone downsamples to stride 32, an FPN neck fuses 4 taps back
             up to stride 4, then 4x upsample.  "Lose resolution, rebuild it."
    DeepLab: backbone never drops below stride 8 (strides in layer3/4 replaced
             by dilation), so ONE output, NO fusion neck, just 8x upsample.
             "Never lose the resolution in the first place."

Two-stage finetune protocol (differs from the FCN project's -- see below):

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
# this file is run directly as a script.
try:
    from .backbone import ResNetBackbone
    from .neck import DeepLabV1Neck
    from .neckV2 import DeepLabV2ASPP
    from .neckV3 import DeepLabV3ASPP
    from .head import DeepLabHead
    from .headV3 import DeepLabHeadV3
except ImportError:
    from backbone import ResNetBackbone
    from neck import DeepLabV1Neck
    from neckV2 import DeepLabV2ASPP
    from neckV3 import DeepLabV3ASPP
    from head import DeepLabHead
    from headV3 import DeepLabHeadV3


class DeepLab(nn.Module):
    """ResNet-backbone DeepLab for PASCAL VOC 2012 semantic segmentation.

    ONE model class serves all three papers -- only the neck (and its head
    wiring) differs, selectable via `neck_type`, while backbone and the
    finetune protocol are shared:

        "largefov" (DeepLab-v1): a single rate-12 atrous 3x3 conv (the neck.py
            DeepLabV1Neck). One field of view.
        "aspp" (DeepLab-v2): Atrous Spatial Pyramid Pooling -- several parallel
            atrous 3x3 convs at different rates, each emitting class SCORES,
            SUMMED (neckV2.py DeepLabV2ASPP). Multiple fields of view; the
            branches are the classifier, so no head.
        "aspp_v3" (DeepLab-v3): atrous branches emit FEATURES, plus a 1x1
            branch and an image-level global-pooling branch; all CONCATENATED
            (neckV3.py DeepLabV3ASPP) and fused by a head (headV3.py). Adds
            global context on top of v2's multi-scale views.

    Args:
        num_classes: 21 for VOC seg (20 object classes + background).
        pretrained: load ImageNet-pretrained backbone weights.
        backbone: backbone arch, "resnet18" or "resnet34".
        neck_type: "largefov" (v1), "aspp" (v2) or "aspp_v3" (v3).
        neck_hidden_channels: v1 only -- width of the LargeFOV atrous conv (256).
        neck_out_channels: v1/v2 -- v1 head input width; v2 branch hidden width (128).
        atrous_rate: v1 only -- dilation of the LargeFOV 3x3 conv (12).
        aspp_rates: v2 only -- the parallel branch dilations. (3, 6, 9, 12)
            here (tighter than the paper's ASPP-L (6, 12, 18, 24)).
        aspp_v3_rates: v3 only -- the atrous branch dilations (3, 6, 9); a 1x1
            branch and a global-pooling branch are added automatically.
        aspp_v3_hidden: v3 only -- per-branch feature width AND the head's
            projection width (256).
        neck_dropout: spatial dropout inside the neck/head (0 disables). Shared.
    """

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = True,
        backbone: str = "resnet34",
        neck_type: str = "largefov",
        neck_hidden_channels: int = 256,
        neck_out_channels: int = 128,
        atrous_rate: int = 12,
        aspp_rates: Sequence[int] = (3, 6, 9, 12),
        aspp_v3_rates: Sequence[int] = (3, 6, 9),
        aspp_v3_hidden: int = 256,
        neck_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.neck_type = neck_type

        # Backbone -> ONE stride-8 feature map [B, 512, H/8, W/8].
        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)

        # Neck: pick the context module. The variants differ in WHERE
        # classification happens, so the head wiring differs too:
        #   largefov -> neck emits neck_out_channels FEATURES; a separate head
        #               does the 1x1 classify + upsample.
        #   aspp     -> each branch emits num_classes SCORES directly and the
        #               neck sums + upsamples them, so there is NO head (the
        #               branches are the classifier). self.head stays None and
        #               forward() calls the neck with the target size instead.
        #   aspp_v3  -> branches emit FEATURES, plus a global-pooling branch;
        #               concatenated and handed to a fusion head (like
        #               largefov structurally, but a much wider concat input).
        if neck_type == "largefov":
            # DeepLab-v1: one rate-12 atrous conv, 512 -> 256 -> 128 features.
            self.neck = DeepLabV1Neck(
                in_channels=self.backbone.out_channels,
                hidden_channels=neck_hidden_channels,
                out_channels=neck_out_channels,
                atrous_rate=atrous_rate,
                dropout=neck_dropout,
            )
            # Head: 1x1 per-pixel classifier + bilinear resize to input size.
            self.head = DeepLabHead(
                in_ch=self.neck.out_channels,
                num_classes=num_classes,
            )
        elif neck_type == "aspp":
            # DeepLab-v2: parallel multi-rate atrous branches, each emitting
            # num_classes scores; summed and upsampled inside the neck.
            # neck_out_channels becomes the branches' internal (hidden) width.
            self.neck = DeepLabV2ASPP(
                in_channels=self.backbone.out_channels,
                num_classes=num_classes,
                hidden_channels=neck_out_channels,
                rates=aspp_rates,
                dropout=neck_dropout,
            )
            self.head = None  # ASPP predicts logits directly -- no head.
        elif neck_type == "aspp_v3":
            # DeepLab-v3: multi-rate atrous branches + a 1x1 branch + an
            # image-level (global-pool) branch, all emitting aspp_v3_hidden
            # FEATURES and CONCATENATED. A fusion head then projects the wide
            # concat, classifies, and upsamples.
            self.neck = DeepLabV3ASPP(
                in_channels=self.backbone.out_channels,
                hidden_channels=aspp_v3_hidden,
                rates=aspp_v3_rates,
            )
            # Head input is the concatenated width (hidden * (len(rates)+2));
            # it squeezes that back down to aspp_v3_hidden before classifying.
            self.head = DeepLabHeadV3(
                in_ch=self.neck.out_channels,
                proj_channels=aspp_v3_hidden,
                num_classes=num_classes,
                dropout=neck_dropout,
            )
        else:
            raise ValueError(
                f"unknown neck_type {neck_type!r}; choose "
                "'largefov' (v1), 'aspp' (v2) or 'aspp_v3' (v3)")

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
        input_size = x.shape[-2:]                 # (H, W)

        c5 = self.backbone(x)                     # [B, 512, H/8, W/8]
        if self.head is None:
            # ASPP (v2): the neck classifies AND upsamples in one shot.
            logits = self.neck(c5, input_size)    # [B, 21,  H,   W]
        else:
            # LargeFOV (v1): neck -> features, head -> classify + upsample.
            feat = self.neck(c5)                  # [B, 128, H/8, W/8]
            logits = self.head(feat, input_size)  # [B, 21,  H,   W]
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
                p for p in self.backbone.low_level_parameters()
                if p.requires_grad
            ],
            "backbone_high": [
                p for p in self.backbone.high_level_parameters()
                if p.requires_grad
            ],
            "neck_head": [
                p for m in (self.neck, self.head) if m is not None
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
    dummy = torch.randn(2, 3, 416, 416)

    # Build ALL THREE neck variants and confirm they produce identical output
    # shapes (only the param count and the internal neck/head differ).
    for neck_type in ("largefov", "aspp", "aspp_v3"):
        # pretrained=False avoids a network download for this shape check.
        model = DeepLab(num_classes=21, pretrained=False, neck_type=neck_type)
        out = model(dummy)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[{neck_type}] logits: {tuple(out.shape)} "
              f"(expected (2, 21, 416, 416))  total params={n_total/1e6:.2f}M")

    # Stage-1 freeze check on the v3 model: low backbone frozen, high backbone
    # + neck/head on. (parameter_groups() is neck-agnostic, so this validates
    # the v3 neck AND head params land in the neck_head group.)
    model.freeze_backbone_low()
    groups = model.parameter_groups()
    for name, params in groups.items():
        n = sum(p.numel() for p in params) / 1e6
        print(f"stage-1 group {name}: {n:.2f}M trainable")

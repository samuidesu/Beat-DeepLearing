"""Full HRNet model: HRNetV2-W32 backbone -> selectable head (simple / OCR).

Forward pass:
    image [B, 3, H, W]
      --backbone-->  4 maps: 32@s4, 64@s8, 128@s16, 256@s32 (parallel branches)
      --head------>  logits [B, 21, H, W]
                     (OCR head in training mode: (main, aux) tuple -- the loss
                      consumes both, weighted by config.AUX_LOSS_WEIGHT)

The head is chosen at construction (`head="simple"` / `"ocr"`), mirroring how
the DeepLab project selected necks -- and like there, everything else
(backbone, loss contract, two-stage layered-LR finetune, eval protocol) stays
identical, so simple-vs-OCR is a clean single-variable comparison.

Two-stage finetune protocol (same shape as the DeepLab experiments):
    Stage 1: freeze the LOW backbone (stem + layer1); train the parallel-
             branch stages + head. Only meaningful when ImageNet-pretrained
             weights were loaded -- freezing RANDOM weights is pointless, so
             without the checkpoint run --epochs-stage1 0.
    Stage 2: unfreeze everything, three LR tiers (low slowest, stages middle,
             head fastest).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import torch
import torch.nn as nn

try:
    from .backbone import HRNetBackbone
    from .head import SimpleHead, OCRHead
except ImportError:
    from backbone import HRNetBackbone
    from head import SimpleHead, OCRHead


class HRNet(nn.Module):
    """HRNetV2-W32 segmenter for PASCAL VOC 2012.

    Args:
        num_classes: 21 for VOC seg (20 object classes + background).
        head: "simple" (HRNetV2 concat head) or "ocr" (OCR head + aux loss).
        pretrained: path to the ImageNet backbone checkpoint, or None for
            random init (backbone prints a warning when the file is missing).
        channels: branch widths, (32, 64, 128, 256) = W32.
        num_blocks / num_modules: HighResolutionModule shape (official 4 and
            (1, 4, 3)).
        ocr_mid_channels / ocr_key_channels: OCR head widths (512 / 256);
            ignored by the simple head.
        head_dropout: Dropout2d inside the head (0 disables).
    """

    def __init__(
        self,
        num_classes: int = 21,
        head: str = "simple",
        pretrained: str | None = None,
        channels: Sequence[int] = (32, 64, 128, 256),
        num_blocks: int = 4,
        num_modules: Sequence[int] = (1, 4, 3),
        ocr_mid_channels: int = 512,
        ocr_key_channels: int = 256,
        head_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head

        self.backbone = HRNetBackbone(
            channels=channels,
            num_blocks=num_blocks,
            num_modules=num_modules,
            pretrained=pretrained,
        )

        # Both heads consume the concat of all 4 branches: sum of the widths.
        concat_ch = sum(self.backbone.out_channels)   # 480 for W32
        if head == "simple":
            self.head = SimpleHead(
                in_channels=concat_ch,
                num_classes=num_classes,
                dropout=head_dropout,
            )
        elif head == "ocr":
            self.head = OCRHead(
                in_channels=concat_ch,
                mid_channels=ocr_mid_channels,
                key_channels=ocr_key_channels,
                num_classes=num_classes,
                dropout=head_dropout,
            )
        else:
            raise ValueError(
                f"unknown head {head!r}; choose 'simple' or 'ocr'")

    def forward(self, x: torch.Tensor):
        """Run the segmenter.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32 -- the eval
                pipeline pads to guarantee it).

        Output:
            raw logits [B, num_classes, H, W]; the OCR head in TRAINING mode
            returns (main, aux) instead (both full resolution).
        """
        input_size = x.shape[-2:]
        feats = self.backbone(x)          # 4 maps, strides 4/8/16/32
        return self.head(feats, input_size)

    # ---- Two-stage finetuning helpers (same interface as the DeepLab models,
    # ---- so train.py's build_layered_optimizer works unchanged) -------------
    def freeze_backbone_low(self):
        """Stage 1: freeze ONLY the stem + layer1."""
        self.backbone.freeze_low_layers()

    def unfreeze_backbone_all(self):
        """Stage 2: unfreeze the ENTIRE backbone for the layered-LR finetune."""
        self.backbone.unfreeze()

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Split parameters into the three layered-LR groups.

        Output (dict of lists, ready for torch.optim param_groups):
            "backbone_low":  stem + layer1            (slowest LR)
            "backbone_high": transitions + stages 2-4 (middle LR)
            "neck_head":     the segmentation head    (fastest LR)

        Frozen parameters (requires_grad=False) are EXCLUDED, so the same call
        works for both stages: in stage 1 "backbone_low" comes back empty.
        (The key names match the DeepLab models so train.py is reusable; HRNet
        has no separate neck -- the head plays both roles.)
        """
        return {
            "backbone_low": [
                p for p in self.backbone.low_level_parameters()
                if p.requires_grad
            ],
            "backbone_high": [
                p for p in self.backbone.high_level_parameters()
                if p.requires_grad
            ],
            "neck_head": [
                p for p in self.head.parameters() if p.requires_grad
            ],
        }

    def set_bn_eval_on_frozen(self):
        """Keep BatchNorm layers whose params are frozen in eval mode.

        Call AFTER model.train() each epoch: model.train() flips every BN back
        to training mode, which would let frozen layers keep updating their
        running mean/var. (In stage 1 that is the stem/layer1 BNs.)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and not m.weight.requires_grad:
                    m.eval()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the parameters that currently require gradients."""
        return (p for p in self.parameters() if p.requires_grad)


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/hrnet.py
if __name__ == "__main__":
    dummy = torch.randn(2, 3, 480, 480)

    for head in ("simple", "ocr"):
        model = HRNet(num_classes=21, head=head, pretrained=None)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[{head}] eval logits: {tuple(out.shape)} "
              f"(expected (2, 21, 480, 480))  total params={n_total/1e6:.2f}M")

    # OCR training mode returns (main, aux).
    model.train()
    main, aux = model(dummy)
    print(f"[ocr] train: main {tuple(main.shape)}  aux {tuple(aux.shape)}")

    # Stage-1 freeze check: low backbone frozen, stages + head trainable.
    model.freeze_backbone_low()
    for name, params in model.parameter_groups().items():
        n = sum(p.numel() for p in params) / 1e6
        print(f"stage-1 group {name}: {n:.2f}M trainable")

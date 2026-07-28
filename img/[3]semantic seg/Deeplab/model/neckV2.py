"""DeepLab-v2 ASPP that DIRECTLY predicts class logits (original-paper style).

Two ways to build an ASPP exist; this file implements the ORIGINAL DeepLab-v2:

    each atrous branch = context conv -> ... -> 1x1 that emits `num_classes`
    scores, and the branch score maps are SUMMED (multi-field-of-view fusion).

So the branches ARE the classifier -- there is no post-fusion feature head. The
alternative (v3-style: branches emit FEATURES, concat/sum, then a separate 1x1
classifier) is what this project used before; see git history if you want the
A/B. Because the ASPP now produces the final score map, it also does the
bilinear upsample to input resolution, and the DeepLab model skips its head for
this neck.

    x [B, in_channels, H/8, W/8]  +  target (H, W)
      -- each ASPPBranch -->  [B, num_classes, H/8, W/8]
      -- sum over branches -->  [B, num_classes, H/8, W/8]
      -- bilinear to (H, W) -->  [B, num_classes, H, W]   raw logits
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPBranch(nn.Module):
    """One atrous branch that outputs class logits directly.

    dilated 3x3 (in -> hidden) -> BN -> ReLU -> Dropout2d -> 1x1 (hidden ->
    num_classes). The 3x3 conv carries the branch's field of view (set by
    `rate`); the 1x1 is this branch's per-pixel linear classifier.

    Args:
        in_channels: channels of the backbone's stride-8 feature (512).
        hidden_channels: width of the atrous conv output (the branch's
            internal feature width, 128 here).
        num_classes: 21 for VOC (20 objects + background).
        rate: dilation = padding of the 3x3 conv (its field of view).
        dropout: Dropout2d prob before the classifier (0 disables).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        rate: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepLabV2ASPP(nn.Module):
    """DeepLab-v2 ASPP: parallel atrous branches -> sum logits -> upsample.

    Input:
        x: [B, in_channels, H/8, W/8]
        output_size: (H, W) of the ORIGINAL network input, so the returned
            logits align exactly with the label mask.

    Output:
        [B, num_classes, H, W]  full-resolution raw logits.
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 21,
        hidden_channels: int = 128,
        rates: Sequence[int] = (3, 6, 9, 12),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(rates) == 0:
            raise ValueError("rates must contain at least one dilation")
        if any(rate <= 0 for rate in rates):
            raise ValueError("all dilation rates must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.branches = nn.ModuleList(
            [
                ASPPBranch(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_classes=num_classes,
                    rate=rate,
                    dropout=dropout,
                )
                for rate in rates
            ]
        )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.output_stride = 8
        self.rates = tuple(rates)

    def forward(
        self,
        x: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected BCHW input, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        # Original DeepLab-v2 fusion: sum the per-branch class-score maps.
        logits = sum(branch(x) for branch in self.branches)

        # Bilinear upsample the 21-channel score map straight to input size
        # (predict-then-upsample: cheaper than upsampling features, and there
        # are no features left to upsample -- the branches already classified).
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


if __name__ == "__main__":
    neck = DeepLabV2ASPP(
        in_channels=512,
        num_classes=21,
        hidden_channels=128,
        rates=(3, 6, 9, 12),
        dropout=0.1,
    )

    dummy = torch.randn(2, 512, 60, 60)          # stride-8 feature for 480x480
    output = neck(dummy, output_size=(480, 480))

    print("output:", tuple(output.shape))
    print("expected:", (2, 21, 480, 480))

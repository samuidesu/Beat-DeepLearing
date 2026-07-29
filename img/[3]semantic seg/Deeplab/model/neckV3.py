"""DeepLab-v3 ASPP neck: multi-rate context + image-level features (FEATURES).

This is the v3 evolution of the ASPP idea. The difference from the v2 neck
(neckV2.py) is WHAT the branches emit and HOW they are fused:

    v2 (neckV2): each branch ends in a 1x1 that emits num_classes SCORES;
                 the score maps are SUMMED -> the branches ARE the classifier,
                 no head.
    v3 (here):   each branch emits hidden-width FEATURES; all branches are
                 CONCATENATED and handed to a separate fusion head (headV3.py)
                 that projects + classifies. Concatenation keeps every
                 branch's response distinct (sum would blend them), and the
                 head learns how to weight them.

Two things v3 adds on top of v2's parallel atrous branches:

    1. A 1x1 branch (rate-1, no dilation): a plain per-pixel projection that
       captures the "no context spreading" view -- useful for small objects
       that the large-rate branches would over-smooth.
    2. An IMAGE-LEVEL branch: global-average-pool the whole feature map to
       1x1 (one vector summarizing the entire image), project it, then
       broadcast it back to the feature-map size. This injects global context
       that even a rate-18 atrous conv cannot see, and fixes a known ASPP
       failure: at very large rates on a small feature map, a 3x3 atrous conv
       degenerates toward a 1x1 (most of its sampling grid falls in padding).

    x [B, in_channels, H/8, W/8]
      -- each atrous 3x3 branch --> [B, hidden, H/8, W/8]   (len(rates) of them)
      -- 1x1 branch             --> [B, hidden, H/8, W/8]
      -- global-pool branch     --> [B, hidden, 1, 1] -> upsampled to H/8,W/8
      -- concat over channels   --> [B, hidden * (len(rates)+2), H/8, W/8]

The neck stops at concatenated FEATURES (still stride 8, NOT classified, NOT
upsampled); headV3.DeepLabHeadV3 does the fuse -> classify -> upsample.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPBranchV3(nn.Module):
    """One ASPP branch that outputs FEATURES: conv -> BN -> ReLU.

    Args:
        in_channels: channels of the backbone's stride-8 feature (512).
        hidden_channels: this branch's output feature width (e.g. 256).
        kernel_size: 1 for the plain pointwise branch, 3 for an atrous branch.
        rate: dilation of the 3x3 branch (ignored when kernel_size == 1). The
            3x3 conv's padding is set equal to the dilation so H, W are
            preserved.

    Input:  [B, in_channels, H/8, W/8]
    Output: [B, hidden_channels, H/8, W/8]  (features, NOT class scores)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        rate: int = 1,
    ) -> None:
        super().__init__()

        # A 1x1 branch has no field of view to spread, so no dilation; a 3x3
        # branch uses dilation=rate and padding=rate (for a 3x3 kernel,
        # padding == dilation keeps the spatial size unchanged).
        if kernel_size == 1:
            dilation = 1
            padding = 0
        elif kernel_size == 3:
            dilation = rate
            padding = rate
        else:
            raise ValueError(
                f"kernel_size must be 1 or 3, got {kernel_size}")

        self.block = nn.Sequential(
            # bias=False because the following BatchNorm has its own shift.
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input [B, in_channels, H/8, W/8] -> [B, hidden_channels, H/8, W/8]."""
        return self.block(x)


class DeepLabV3ASPP(nn.Module):
    """DeepLab-v3 ASPP: parallel atrous branches + image-level pool, concatenated.

    Args:
        in_channels: channels of the backbone's stride-8 feature (512).
        hidden_channels: per-branch feature width (256).
        rates: dilations of the atrous 3x3 branches. A 1x1 branch and a
            global-pooling branch are ALWAYS added on top, so the total branch
            count is len(rates) + 2.

    Attributes:
        out_channels (int): channels of the concatenated output,
            hidden_channels * (len(rates) + 2). The fusion head reads this to
            size its 1x1 projection.

    Input:  [B, in_channels, H/8, W/8]
    Output: [B, out_channels, H/8, W/8]  (concatenated features, stride 8)
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        rates: Sequence[int] = (3, 6, 9),
    ) -> None:
        super().__init__()

        if len(rates) == 0:
            raise ValueError("rates must contain at least one dilation")
        if any(rate <= 0 for rate in rates):
            raise ValueError("all dilation rates must be positive")

        # One atrous 3x3 branch per rate, PLUS a plain 1x1 branch.
        self.branches = nn.ModuleList(
            [
                ASPPBranchV3(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=3,
                    rate=rate,
                )
                for rate in rates
            ]
            + [
                ASPPBranchV3(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=1,
                )
            ]
        )

        # Image-level branch: pool the whole map to a single vector, project
        # it, then (in forward) broadcast it back to the feature-map size.
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.rates = tuple(rates)
        self.output_stride = 8
        # len(rates) atrous branches + 1 pointwise branch + 1 global branch.
        self.out_channels = hidden_channels * (len(rates) + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse all branches by concatenation.

        Input:
            x: [B, in_channels, H/8, W/8] backbone stride-8 feature.

        Output:
            [B, out_channels, H/8, W/8] concatenated branch features (still
            stride 8 -- the head classifies and upsamples).
        """
        if x.ndim != 4:
            raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}")

        # Atrous + 1x1 branches keep the input's spatial size already.
        features = [branch(x) for branch in self.branches]

        # Global branch: [B, hidden, 1, 1] -> broadcast back to [B, hidden, H/8, W/8].
        global_feature = self.global_branch(x)
        global_feature = F.interpolate(
            global_feature,
            size=x.shape[-2:],          # current feature-map (H/8, W/8)
            mode="bilinear",
            align_corners=False,
        )
        features.append(global_feature)

        # Concatenate over channels: every branch's response stays distinct.
        return torch.cat(features, dim=1)


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/neckV3.py
if __name__ == "__main__":
    rates = (3, 6, 9)
    neck = DeepLabV3ASPP(in_channels=512, hidden_channels=256, rates=rates)

    dummy = torch.randn(2, 512, 60, 60)          # stride-8 feature for 480x480
    output = neck(dummy)

    expected_c = 256 * (len(rates) + 2)          # 256 * 5 = 1280
    print("output:", tuple(output.shape),
          f"(expected (2, {expected_c}, 60, 60))")
    print("out_channels:", neck.out_channels)

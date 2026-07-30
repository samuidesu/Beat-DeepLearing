"""DeepLab-v3+ encoder: ASPP context -> 1x1 project -> upsample to stride 4.

This is the ENCODER half of DeepLab-v3+. It is the v3 ASPP with two extra steps
so its output is ready for the v3+ decoder (headV3.py):

    1. ASPP: parallel atrous 3x3 branches (one per rate) + a 1x1 branch + an
       image-level global-pooling branch, each emitting hidden-width FEATURES,
       all CONCATENATED. (Same as plain v3.)
    2. Project: a 1x1 conv squeezes the wide concat (hidden * num_branches) down
       to `PROJECT_CHANNELS` (256) -- the standard DeepLab ASPP output width.
    3. Upsample x2: stride 8 -> stride 4, so the encoder output lines up with
       the backbone's stride-4 low-level feature (c2) that the decoder fuses in.

    c5 [B, in_channels, H/8, W/8]
      -- ASPP branches --> concat [B, hidden*(len(rates)+2), H/8, W/8]
      -- 1x1 project    --> [B, 256, H/8, W/8]
      -- bilinear x2     --> [B, 256, H/4, W/4]   <- handed to the decoder

Why v3+ over v3: plain v3 upsamples the stride-8 logits straight to full
resolution (8x), so object boundaries stay blurry. v3+ instead recovers detail
in a small DECODER (headV3.py) that fuses this encoder output with the crisp
stride-4 low-level feature -- sharper edges for thin structures. This neck stops
at stride-4 FEATURES; the decoder does the low-level fusion, classification and
final 4x upsample.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard DeepLab ASPP output-projection width. The decoder (headV3.py) expects
# the encoder to hand it exactly this many channels, so the two must agree.
PROJECT_CHANNELS = 256


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
            raise ValueError(f"kernel_size must be 1 or 3, got {kernel_size}")

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
    """DeepLab-v3+ encoder: ASPP -> concat -> 1x1 project -> 2x upsample.

    Args:
        in_channels: channels of the backbone's stride-8 feature (512).
        hidden_channels: per-branch feature width (256).
        rates: dilations of the atrous 3x3 branches. A 1x1 branch and a
            global-pooling branch are ALWAYS added on top, so the total branch
            count is len(rates) + 2.
        dropout: Dropout2d probability after the 1x1 projection (0 disables).

    Attributes:
        out_channels (int): channels of the returned feature (PROJECT_CHANNELS,
            256). The decoder reads this to size its fusion conv.
        output_stride (int): stride of the returned feature (4, because this
            module already upsamples the ASPP result x2 from stride 8).

    Input:  [B, in_channels, H/8, W/8]
    Output: [B, 256, H/4, W/4]  (projected encoder features at stride 4)
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        rates: Sequence[int] = (3, 6, 9),
        dropout: float = 0.1,
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

        # Width of the concatenated branches: len(rates) atrous + 1 pointwise +
        # 1 global = len(rates) + 2 branches, each hidden_channels wide.
        # (MUST be computed before self.project, which consumes it.)
        concat_channels = hidden_channels * (len(rates) + 2)

        # Project the wide concat down to the standard ASPP output width (256).
        # This is the learned fusion of the branches (v3+ keeps it in the
        # encoder; the decoder does a second, low-level fusion later).
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, PROJECT_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(PROJECT_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.rates = tuple(rates)
        # After the x2 upsample below the output is at stride 4, width 256.
        self.output_stride = 4
        self.out_channels = PROJECT_CHANNELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ASPP, project, and upsample x2 to stride 4.

        Input:
            x: [B, in_channels, H/8, W/8] backbone stride-8 feature (c5).

        Output:
            [B, 256, H/4, W/4] projected encoder features at stride 4, ready
            for the v3+ decoder to fuse with the stride-4 low-level feature.
        """
        if x.ndim != 4:
            raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        # Atrous + 1x1 branches keep the input's spatial size already.
        features = [branch(x) for branch in self.branches]

        # Global branch: [B, hidden, 1, 1] -> broadcast back to [B, hidden, H/8, W/8].
        global_feature = self.global_branch(x)
        global_feature = F.interpolate(
            global_feature,
            size=x.shape[-2:],  # current feature-map (H/8, W/8)
            mode="bilinear",
            align_corners=False,
        )
        features.append(global_feature)

        # Concatenate (every branch stays distinct), project to 256.
        feat = torch.cat(features, dim=1)
        feat = self.project(feat)

        # Upsample x2: stride 8 -> stride 4, to match the decoder's low-level
        # feature. scale_factor=2 is exact here because the eval pipeline pads
        # inputs to a multiple of 8, so H/8 is integer and H/8 * 2 == H/4.
        return F.interpolate(
            feat,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/neckV3.py
if __name__ == "__main__":
    rates = (3, 6, 9)
    neck = DeepLabV3ASPP(in_channels=512, hidden_channels=256, rates=rates)

    dummy = torch.randn(2, 512, 60, 60)  # stride-8 feature for a 480x480 input
    output = neck(dummy)

    # project -> 256 channels; x2 upsample -> 120x120 (stride 4).
    print("output:", tuple(output.shape), "(expected (2, 256, 120, 120))")
    print("out_channels:", neck.out_channels, " output_stride:", neck.output_stride)

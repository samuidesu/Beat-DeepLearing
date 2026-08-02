"""HRNet building blocks: residual blocks, fusion units, HighResolutionModule.

HRNet's core idea lives in this file. A "stage" holds SEVERAL parallel branches
at different resolutions (stride 4 / 8 / 16 / 32), and after every few blocks
the branches EXCHANGE information: every branch receives every other branch,
resampled to its own resolution, and sums them. Two directions of exchange:

    Low2High  (j > i, lower-res -> higher-res): 1x1 conv to match channels,
              then bilinear-UPSAMPLE. Cheap, because upsampling carries no new
              spatial detail -- the 1x1 only aligns channel semantics.
    High2Low  (j < i, higher-res -> lower-res): a CHAIN of 3x3 stride-2 convs,
              one per factor of 2. Strided convs (not pooling) so the branch
              LEARNS what detail to keep while shrinking.

NOTE on the original draft of this file: the two fusion units were sketched as
plain classes -- three fixes were needed to make them real PyTorch modules:
  * inherit from nn.Module/nn.Sequential (otherwise parameters are invisible
    to .parameters(), .to(device) and state_dict);
  * F.interpolate takes `scale_factor=`, not `factor=`;
  * the downsample convs need stride=2 AND padding=1 (a bare 3x3 conv shrinks
    the map by 2 pixels instead of halving it), and one conv per FACTOR OF 2
    (range(log2(factor)) steps, not factor//2 - 1).

Weight-naming contract: every module here subclasses nn.Sequential with plain
integer children, so the state_dict keys (fuse_layers.1.0.0.0.weight, ...)
match the OFFICIAL HRNet ImageNet checkpoint exactly -- that is what lets
backbone.load_pretrained() consume hrnetv2_w32_imagenet_pretrained.pth
without any key translation.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3x3 convs + identity shortcut).

    HRNet uses these inside every branch of stages 2-4. expansion = 1: output
    channels == `planes`.

    Input:  [B, inplanes, H, W]
    Output: [B, planes,   H, W]  (stride is always 1 inside HRNet branches)
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Residual add BEFORE the final ReLU (post-activation ResNet).
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """Standard ResNet Bottleneck (1x1 squeeze -> 3x3 -> 1x1 expand x4).

    Only used in HRNet's stage 1 (`layer1`): 4 of these take the stride-4 stem
    output from 64 to 256 channels before the parallel branches split off.

    Input:  [B, inplanes, H, W]
    Output: [B, planes * 4, H, W]
    """

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 projection shortcut when in/out channels differ (first block).
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class Low2High(nn.Sequential):
    """Fusion unit: bring a LOWER-resolution branch UP to a higher one.

    1x1 conv (channel alignment) -> BN, then bilinear upsample in forward.
    Subclasses nn.Sequential so the children are indexed 0/1 -- the exact key
    layout (`...0.weight`, `...1.running_mean`) of the official checkpoint's
    fuse layers.

    Args:
        in_channels: channels of the lower-resolution source branch (larger).
        out_channels: channels of the higher-resolution target branch.

    forward Input:
        x: [B, in_channels, h, w] (small map).
        size: (H, W) of the TARGET branch's map. Interpolating to an explicit
            size (not a fixed scale factor) keeps fusion exact for any padded
            input.
    forward Output: [B, out_channels, H, W].
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:  # type: ignore[override]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class High2Low(nn.Sequential):
    """Fusion unit: bring a HIGHER-resolution branch DOWN to a lower one.

    One 3x3 stride-2 conv per factor of 2 (so stride 4 -> 32 = 3 chained
    convs). Intermediate steps keep the SOURCE channel count and end in ReLU;
    the LAST step maps to the target channels and has NO ReLU -- the fusion
    sum is activated once, after all branches are added (official design).

    Each step is an inner nn.Sequential, giving official-checkpoint keys
    `...<step>.0.weight` / `...<step>.1.*`.

    Args:
        in_channels: channels of the higher-resolution source branch (smaller).
        out_channels: channels of the lower-resolution target branch.
        num_steps: how many /2 downsamples (i - j in fuse-layer terms).

    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H / 2**num_steps, W / 2**num_steps]
    """

    def __init__(self, in_channels: int, out_channels: int, num_steps: int) -> None:
        steps = []
        for k in range(num_steps):
            last = k == num_steps - 1
            steps.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels if last else in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels if last else in_channels),
                    *([] if last else [nn.ReLU(inplace=True)]),
                )
            )
        super().__init__(*steps)


class HighResolutionModule(nn.Module):
    """One HRNet module: parallel branches + full cross-resolution fusion.

    Structure (for `num_branches` = N):
        branches[i]:    num_blocks BasicBlocks at branch i's resolution/width.
        fuse_layers[i]: how every OTHER branch j reaches branch i --
                        Low2High if j > i, High2Low if j < i, None if j == i.

    forward: run each branch, then rebuild every output as
        y[i] = ReLU( sum_j fuse(x[j] -> resolution i) )
    so after EVERY module, each resolution has seen all the others. This
    repeated exchange is what keeps the stride-4 branch semantically strong --
    it constantly receives context from the deep, low-resolution branches.

    Input / Output: list of N tensors, [B, channels[i], H/2^i', W/2^i'].
    """

    def __init__(
        self,
        num_branches: int,
        channels: Sequence[int],
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        if num_branches != len(channels):
            raise ValueError(
                f"num_branches={num_branches} != len(channels)={len(channels)}"
            )
        self.num_branches = num_branches
        self.channels = tuple(channels)

        # ---- Parallel branches: num_blocks BasicBlocks each ----
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[BasicBlock(channels[i], channels[i]) for _ in range(num_blocks)]
                )
                for i in range(num_branches)
            ]
        )

        # ---- Fusion: fuse_layers[i][j] transforms branch j -> branch i ----
        fuse_layers = []
        for i in range(num_branches):
            row = []
            for j in range(num_branches):
                if j == i:
                    row.append(None)  # own branch: identity
                elif j > i:
                    # lower-res -> this (higher) res: 1x1 + upsample.
                    row.append(Low2High(channels[j], channels[i]))
                else:
                    # higher-res -> this (lower) res: (i-j) strided 3x3 convs.
                    row.append(High2Low(channels[j], channels[i], num_steps=i - j))
            fuse_layers.append(nn.ModuleList(row))
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(x) != self.num_branches:
            raise ValueError(f"expected {self.num_branches} inputs, got {len(x)}")

        # 1) Independent per-branch computation.
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 2) Full cross-resolution fusion (sum, then one ReLU).
        out = []
        for i in range(self.num_branches):
            target_size = x[i].shape[-2:]
            y = x[i]  # j == i contribution
            for j in range(self.num_branches):
                if j == i:
                    continue
                fuse = self.fuse_layers[i][j]
                if j > i:
                    y = y + fuse(x[j], target_size)  # Low2High needs the size
                else:
                    y = y + fuse(x[j])  # High2Low just downsamples
            out.append(self.relu(y))
        return out


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/branch.py
if __name__ == "__main__":
    # A 3-branch module like one from stage3 (W32 widths 32/64/128).
    channels = (32, 64, 128)
    module = HighResolutionModule(3, channels, num_blocks=4)

    feats = [
        torch.randn(2, 32, 120, 120),  # stride 4
        torch.randn(2, 64, 60, 60),  # stride 8
        torch.randn(2, 128, 30, 30),  # stride 16
    ]
    outs = module(feats)
    for i, o in enumerate(outs):
        print(f"branch {i}: {tuple(o.shape)} (expected {tuple(feats[i].shape)})")

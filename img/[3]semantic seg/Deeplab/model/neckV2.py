from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer"
            )
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        padding = dilation * (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DeepLabV2ASPP(nn.Module):
    """DeepLab-v2-style ASPP for an output-stride-8 feature map.

    Input:
        [B, in_channels, H/8, W/8]

    Output:
        [B, out_channels, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 128,
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
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                )
                for rate in rates
            ]
        )

        self.dropout = nn.Dropout2d(dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_stride = 8
        self.rates = tuple(rates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected BCHW input, got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        branch_outputs = [
            branch(x)
            for branch in self.branches
        ]

        # DeepLab-v2-style ASPP fusion: element-wise sum.
        out = torch.stack(branch_outputs, dim=0).sum(dim=0)
        out = self.dropout(out)

        return out


if __name__ == "__main__":
    neck = DeepLabV2ASPP(
        in_channels=512,
        out_channels=128,
        rates=(3, 6, 9, 12),
        dropout=0.1,
    )

    dummy = torch.randn(2, 512, 60, 60)
    output = neck(dummy)

    print("output:", tuple(output.shape))
    print("expected:", (2, 128, 60, 60))
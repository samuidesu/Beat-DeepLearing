"""DeepLab-v1-style context neck.

The backbone already returns a high-level stride-8 feature.  Therefore this
module is not an FPN and performs no top-down feature fusion.  It implements a
LargeFOV-style atrous context layer followed by a 1x1 projection.

    c5 [B, 512, H/8, W/8]
      -> 3x3 atrous conv, rate 12
      -> 1x1 projection
      -> [B, 128, H/8, W/8]

The segmentation head can then map the 128 channels to class logits and
bilinearly resize the logits to the original image size.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """Convolution followed by BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
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


class DeepLabV1Neck(nn.Module):
    """Large-field-of-view context head for a stride-8 ResNet feature.

    Args:
        in_channels: Channels returned by the backbone; 512 for ResNet-18/34.
        hidden_channels: Width of the atrous context representation.
        out_channels: Channels passed to the final segmentation classifier.
        atrous_rate: Dilation of the LargeFOV 3x3 convolution.  Rate 12 is a
            common DeepLab-v1-style choice for an output-stride-8 feature.
        dropout: Spatial dropout probability; set to 0 to disable.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        out_channels: int = 128,
        atrous_rate: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            ConvBNReLU(
                in_channels,
                hidden_channels,
                kernel_size=3,
                dilation=atrous_rate,
            )
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(
            ConvBNReLU(
                hidden_channels,
                out_channels,
                kernel_size=1,
            )
        )

        self.context = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_stride = 8
        self.atrous_rate = atrous_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        return self.context(x)


if __name__ == "__main__":
    neck = DeepLabV1Neck(
        in_channels=512,
        hidden_channels=256,
        out_channels=128,
        atrous_rate=12,
    )
    dummy = torch.randn(2, 512, 52, 52)
    output = neck(dummy)
    print("output:", tuple(output.shape), "expected: (2, 128, 52, 52)")

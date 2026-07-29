"""DeepLab-v3 fusion head: concat ASPP features -> classify -> upsample.

The v3 ASPP neck (neckV3.py) emits CONCATENATED features from all its branches
(len(rates) atrous + 1 pointwise + 1 global = several thousand channels), not
class scores. This head turns that fat concatenated tensor into per-pixel class
logits at input resolution:

    feat [B, in_ch, H/8, W/8]           (in_ch = hidden * num_branches, e.g. 1280)
     -> 1x1 "fuse"    [B, proj, H/8, W/8]   BN + ReLU + Dropout: mix the branches
     -> 1x1 classify  [B, 21,   H/8, W/8]   per-pixel linear classifier
     -> bilinear up   [B, 21,   H,   W]     back to input resolution

Two 1x1 convs, two jobs:
  * the FUSE 1x1 learns how to WEIGHT and COMBINE the concatenated branches
    (this is why v3 concatenates instead of summing -- the head, not a fixed
    sum, decides each branch's contribution) and squeezes the wide concat back
    down to `proj` channels;
  * the CLASSIFY 1x1 maps each fused vector to 21 class logits.

Dropout2d sits between them as v3's regularizer on the fused representation.

Predict-then-upsample (classify at stride 8, THEN interpolate the 21-channel
logits) stays cheaper than upsampling wide features, and interpolating to the
EXPLICIT input size (passed in by the model) keeps logits and label masks
aligned even when the eval pipeline padded the image to a multiple of 8.

Outputs are RAW logits (no softmax): the loss is
nn.CrossEntropyLoss(ignore_index=255), which applies log-softmax internally and
skips VOC's 255 void/pad pixels. Output stays channel-first [B, 21, H, W]
because CrossEntropyLoss wants [B, C, ...].
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHeadV3(nn.Module):
    """Fuse concatenated ASPP features, classify per pixel, upsample to input.

    Args:
        in_ch: channels of the concatenated neck output
            (DeepLabV3ASPP.out_channels, e.g. hidden * (len(rates)+2) = 1280).
        proj_channels: width the fuse 1x1 squeezes the concat down to before
            classification (the DeepLab-v3 ASPP projection width, 256).
        num_classes: 21 for VOC (20 object classes + explicit background).
        dropout: Dropout2d probability on the fused features (0 disables).
    """

    def __init__(
        self,
        in_ch: int = 256 * 5,
        proj_channels: int = 256,
        num_classes: int = 21,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Fuse: 1x1 squeezes the wide concat (in_ch) to proj_channels and, via
        # its learned weights, decides how much each branch contributes.
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        # Classifier: per-pixel linear map proj_channels -> class logits.
        self.classifier = nn.Conv2d(proj_channels, num_classes, kernel_size=1)

    def forward(
        self,
        feat: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Classify the fused features and resize to the input size.

        Input:
            feat: [B, in_ch, H/8, W/8] concatenated ASPP features from the neck.
            output_size: (H, W) of the ORIGINAL network input, so the logits
                align exactly with the label mask.

        Output:
            raw logits [B, num_classes, H, W], channel-first, ready for
            nn.CrossEntropyLoss(ignore_index=255). At inference, argmax over
            dim 1 gives the predicted class id per pixel.
        """
        logits = self.classifier(self.fuse(feat))   # [B, 21, H/8, W/8]

        # Bilinear, not nearest: nearest would copy every stride-8 logit into a
        # blocky 8x8 tile (jagged borders); bilinear interpolates smoothly.
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/headV3.py
if __name__ == "__main__":
    # Fake concatenated neck output: hidden=256, 5 branches -> 1280 channels,
    # stride-8 grid for a 480x480 input.
    in_ch = 256 * 5
    feat = torch.randn(2, in_ch, 60, 60)

    head = DeepLabHeadV3(in_ch=in_ch, proj_channels=256, num_classes=21)
    out = head(feat, output_size=(480, 480))

    print("out:", tuple(out.shape), "(expected (2, 21, 480, 480))")

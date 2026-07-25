"""DeepLab segmentation head (PASCAL VOC: 21 classes = 20 objects + background).

The neck's LargeFOV context module already did all the feature mixing, so the
head is deliberately tiny -- per-pixel LINEAR classification plus resizing:

    feat [B, 128, H/8, W/8]
     -> 1x1 conv "classifier"   [B, 21, H/8, W/8]  (per-pixel class logits)
     -> bilinear upsample       [B, 21, H,   W]    (back to input resolution)

Predict-then-upsample (NOT upsample-then-predict) on purpose: upsampling the
21-channel logit map is ~6x cheaper than upsampling the 128-channel feature
map, and it is exactly what DeepLab-v1 did (the paper then refined the
upsampled result with a DenseCRF; we skip the CRF -- it is obsolete, and the
atrous backbone is the part of DeepLab that survived).

Only 8x upsampling is needed here vs. FCN-8s' 32x-from-stride-32 story: the
dilated backbone never dropped below stride 8, so far less resolution has to
be "invented" by interpolation -- that is the whole DeepLab bet.

The upsample is nn.functional.interpolate to an EXPLICIT target size (passed
in by the model), not a fixed scale_factor=8: eval images are padded to a
multiple of the stride, and interpolating to the exact padded input size
avoids any off-by-a-pixel drift between logits and label masks.

Outputs are RAW logits, no softmax (same "the loss decodes the raw output"
contract as every project in this repo). The loss is
nn.CrossEntropyLoss(ignore_index=255): per-pixel softmax over the 21 classes,
skipping the white object-boundary pixels that VOC label PNGs mark as 255.
Output stays channel-first [B, 21, H, W] because CrossEntropyLoss wants
[B, C, ...].
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHead(nn.Module):
    """1x1 classifier + bilinear resize: context features -> full-res logits.

    Args:
        in_ch: channels of the neck output (matches DeepLabV1Neck.out_channels,
            default 128).
        num_classes: 21 for VOC segmentation -- 20 object classes PLUS an
            explicit background class (index 0 in the VOC label PNGs).
    """

    def __init__(self, in_ch: int = 128, num_classes: int = 21) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1x1 is enough for the classifier: all spatial reasoning already
        # happened in the dilated backbone and the LargeFOV neck; this is
        # per-pixel linear classification of the 128-d context vector.
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(
        self,
        feat: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Predict per-pixel class logits and resize them to the input size.

        Input:
            feat: [B, in_ch, H/8, W/8] from the neck.
            output_size: (H, W) of the ORIGINAL network input -- the model
                passes its input's spatial size so logits and label masks
                align exactly.

        Output:
            raw logits [B, num_classes, H, W], channel-first, ready for
            nn.CrossEntropyLoss(ignore_index=255). At inference, argmax over
            dim 1 gives the predicted class id per pixel.
        """
        logits = self.classifier(feat)          # [B, 21, H/8, W/8]

        # Bilinear, not nearest: nearest would copy every stride-8 logit into
        # a blocky 8x8 tile (jagged mask borders); bilinear interpolates
        # smoothly between grid points.
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/head.py
if __name__ == "__main__":
    # Fake neck output (stride-8, 128 channels) for a 416x416 input.
    feat = torch.randn(2, 128, 52, 52)

    head = DeepLabHead(in_ch=128, num_classes=21)
    out = head(feat, output_size=(416, 416))

    print("out:", tuple(out.shape), "(expected (2, 21, 416, 416))")

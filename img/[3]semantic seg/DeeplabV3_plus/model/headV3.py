"""DeepLab-v3+ decoder: fuse encoder features with a low-level feature, classify.

This is the DECODER half of DeepLab-v3+ -- the small module that separates v3+
from plain v3. Plain v3 upsamples the stride-8 logits straight to full
resolution (an 8x jump), so boundaries stay blurry. v3+ recovers detail here by
fusing two sources at stride 4:

    * the ENCODER output (neckV3.py) -- rich semantics, already upsampled to
      stride 4, 256 channels;
    * a LOW-LEVEL backbone feature (c2, after layer1) -- stride 4, only 64
      channels, but crisp spatial detail (edges) the encoder lost.

    low-level c2 [B, 64, H/4, W/4]
      -- 1x1 reduce -->  [B, 48, H/4, W/4]        (paper: shrink low-level to
                                                   ~48ch so it doesn't outweigh
                                                   the 256-ch encoder features)
    encoder feat  [B, 256, H/4, W/4]
      -- concat over channels -->  [B, 48+256, H/4, W/4]
      -- 3x3, 3x3 fuse        -->  [B, 256, H/4, W/4]   (refine the fusion)
      -- 1x1 classify         -->  [B, 21,  H/4, W/4]
      -- bilinear to input    -->  [B, 21,  H,   W]     (a 4x jump, not 8x)

The two 3x3 convs (vs. v3's single 1x1) let the decoder blend semantics and edge
detail spatially. The final upsample is only 4x (stride 4 -> 1) because the
decoder already runs at stride 4, so less detail is "invented" by interpolation
than v3's 8x -- the whole point of v3+.

Interpolation targets the EXPLICIT input size passed by the model, so logits and
label masks align exactly even when the eval pipeline padded the image to a
multiple of 8. Outputs are RAW logits (no softmax): the loss is
nn.CrossEntropyLoss(ignore_index=255). Output is channel-first [B, 21, H, W].
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Decoder feature width (the DeepLab-v3+ default). The fuse convs run at this
# width and the classifier reads it.
DECODER_CHANNELS = 256


class DeepLabHeadV3(nn.Module):
    """v3+ decoder: reduce low-level feature, concat with encoder, fuse, classify.

    Args:
        aspp_channels: channels of the encoder (neckV3) output (256).
        low_level_channels: channels of the low-level backbone feature c2
            (64 for ResNet-18/34's layer1 output).
        low_level_proj: channels the low-level feature is reduced to before the
            concat (the paper uses 48 so it does not overwhelm the 256-channel
            encoder features).
        num_classes: 21 for VOC (20 object classes + explicit background).
        dropout: Dropout2d probability inside the fuse block (0 disables).
    """

    def __init__(
        self,
        aspp_channels: int = 256,
        low_level_channels: int = 64,
        low_level_proj: int = 48,
        num_classes: int = 21,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Reduce the low-level feature (c2) to a small channel count so it does
        # not dominate the fusion (the encoder features carry the semantics).
        self.low_level_reduce = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_proj),
            nn.ReLU(inplace=True),
        )

        # Fuse: two 3x3 convs blend the concatenated [low-level | encoder]
        # features spatially, keeping the width at DECODER_CHANNELS.
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_channels + low_level_proj, DECODER_CHANNELS,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(DECODER_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DECODER_CHANNELS, DECODER_CHANNELS,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(DECODER_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        # Per-pixel linear classifier.
        self.classifier = nn.Conv2d(DECODER_CHANNELS, num_classes, kernel_size=1)

    def forward(
        self,
        low_level_feat: torch.Tensor,
        aspp_feat: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Fuse low-level + encoder features, classify, upsample to input size.

        Input:
            low_level_feat: [B, low_level_channels, H/4, W/4] backbone c2.
            aspp_feat: [B, aspp_channels, H/4, W/4] encoder output (neckV3),
                already upsampled to stride 4 so it matches c2's resolution.
            output_size: (H, W) of the ORIGINAL network input, so the logits
                align exactly with the label mask.

        Output:
            raw logits [B, num_classes, H, W], channel-first, ready for
            nn.CrossEntropyLoss(ignore_index=255). At inference, argmax over
            dim 1 gives the predicted class id per pixel.
        """
        low = self.low_level_reduce(low_level_feat)  # [B, 48, H/4, W/4]
        feat = torch.cat((low, aspp_feat), dim=1)    # [B, 48+256, H/4, W/4]
        logits = self.classifier(self.fuse(feat))    # [B, 21, H/4, W/4]

        # Final upsample to the exact input size (a 4x jump from stride 4).
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/headV3.py
if __name__ == "__main__":
    # Fake stride-4 features for a 480x480 input: c2 (64ch) + encoder (256ch).
    c2 = torch.randn(2, 64, 120, 120)
    aspp = torch.randn(2, 256, 120, 120)

    head = DeepLabHeadV3(aspp_channels=256, low_level_channels=64, num_classes=21)
    out = head(c2, aspp, output_size=(480, 480))

    print("out:", tuple(out.shape), "(expected (2, 21, 480, 480))")

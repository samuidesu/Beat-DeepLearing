"""HRNet segmentation heads: "simple" (HRNetV2) and "ocr" (HRNet-OCR).

Both heads start the same way: upsample all 4 branch outputs to stride 4 and
CONCATENATE them (32+64+128+256 = 480 channels) -- that concat IS the
"HRNetV2" representation. They differ in what happens next:

SIMPLE (HRNetV2 paper head):
    concat 480 -> 1x1 mix (480) -> 1x1 classify (21) -> upsample to input.
    Per-pixel classification of the concatenated multi-resolution feature;
    no pixel interacts with any other pixel beyond conv receptive fields.

OCR (object-contextual representations):
    The idea: a pixel is best classified in the context of the OBJECT it
    belongs to, not just its local window. Three steps --
      1. SOFT REGIONS: an auxiliary FCN head predicts coarse class scores
         (also trained, with weight config.AUX_LOSS_WEIGHT -- this is the
         auxiliary loss).
      2. GATHER: for each class k, average all pixel features weighted by
         their (softmaxed) score for k -> ONE feature vector per class, the
         "object-region representation" [B, C, K].
      3. DISTRIBUTE: every pixel ATTENDS over the K class vectors (query =
         pixel feature, key/value = class vectors); the attended context is
         concatenated back onto the pixel feature and mixed. Pixels of the
         same object thereby share one coherent representation.
    Training returns (main_logits, aux_logits); eval returns main only.

Outputs are RAW logits, upsampled to the EXPLICIT input size passed by the
model (keeps logits aligned with padded eval inputs). Loss:
nn.CrossEntropyLoss(ignore_index=255) -- see losses/hrnet_loss.py for how the
OCR tuple is consumed.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _concat_branches(feats: List[torch.Tensor]) -> torch.Tensor:
    """Upsample every branch to branch-0 resolution (stride 4) and concat.

    Input:  4 maps [B, ci, H/2^i', W/2^i'] (strides 4/8/16/32).
    Output: [B, sum(ci) = 480, H/4, W/4].
    """
    size = feats[0].shape[-2:]
    up = [feats[0]] + [
        F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        for f in feats[1:]
    ]
    return torch.cat(up, dim=1)


class SimpleHead(nn.Module):
    """HRNetV2 head: concat -> 1x1 mix -> 1x1 classify -> upsample.

    Args:
        in_channels: width of the branch concat (480 for W32).
        num_classes: 21 for VOC.
        dropout: Dropout2d before the classifier (0 disables).
    """

    def __init__(self, in_channels: int = 480, num_classes: int = 21,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.num_classes = num_classes
        # 1x1 mix at the SAME width (official HRNetV2 head): lets the head
        # re-weight the four resolutions' channels before classifying.
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, feats: List[torch.Tensor],
                output_size: Tuple[int, int]) -> torch.Tensor:
        """4 branch maps -> full-resolution logits.

        Input:
            feats: the backbone's 4 outputs (strides 4/8/16/32).
            output_size: (H, W) of the network input.
        Output:
            raw logits [B, num_classes, H, W].
        """
        x = _concat_branches(feats)              # [B, 480, H/4, W/4]
        logits = self.last_layer(x)              # [B, 21,  H/4, W/4]
        return F.interpolate(logits, size=output_size,
                             mode="bilinear", align_corners=False)


class SpatialGather(nn.Module):
    """OCR step 2: pool pixel features into one vector per CLASS.

    For each class k: softmax the aux scores over ALL pixels (so they sum to 1
    per class -- a spatial attention map), then take the weighted average of
    the pixel features. No learned parameters.

    Input:
        feats: [B, C, h, w] pixel features.
        probs: [B, K, h, w] aux class scores (raw logits are fine; softmax
            over the PIXEL axis happens here).
    Output:
        [B, C, K, 1] -- K class ("object region") vectors, kept 4-D so 1x1
        convs can consume them like a feature map.
    """

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feats.shape
        k = probs.shape[1]
        probs = probs.view(b, k, h * w)                 # [B, K, P]
        feats = feats.view(b, c, h * w).permute(0, 2, 1)  # [B, P, C]
        probs = F.softmax(probs, dim=2)                 # attention over pixels
        # [B, K, P] @ [B, P, C] -> [B, K, C]: weighted average per class.
        context = torch.matmul(probs, feats)
        return context.permute(0, 2, 1).unsqueeze(3)    # [B, C, K, 1]


def _project(in_ch: int, out_ch: int) -> nn.Sequential:
    """1x1 conv + BN + ReLU: the query/key/value projections of the OCR
    attention. (The official implementation stacks two of these per
    projection; one is kept here for compactness -- same mechanism.)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ObjectAttention(nn.Module):
    """OCR step 3: every pixel attends over the K class vectors.

    Scaled dot-product attention with pixels as queries and the class/object
    vectors as keys AND values:

        sim[p, k] = <query(pixel p), key(class k)> / sqrt(key_ch)
        context_p = sum_k softmax_k(sim) * value(class k)

    Output is projected back up to `channels` so it can be concatenated with
    the original pixel features.

    Args:
        channels: pixel-feature width (512).
        key_channels: query/key/value width inside the attention (256).
    """

    def __init__(self, channels: int, key_channels: int) -> None:
        super().__init__()
        self.key_channels = key_channels
        self.f_pixel = _project(channels, key_channels)    # query (pixels)
        self.f_object = _project(channels, key_channels)   # key   (classes)
        self.f_down = _project(channels, key_channels)     # value (classes)
        self.f_up = _project(key_channels, channels)       # context -> C

    def forward(self, feats: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Input: feats [B, C, h, w], context [B, C, K, 1].
        Output: attended object context per pixel, [B, C, h, w]."""
        b, c, h, w = feats.shape
        query = self.f_pixel(feats).view(b, self.key_channels, -1)      # [B, key, P]
        query = query.permute(0, 2, 1)                                  # [B, P, key]
        key = self.f_object(context).view(b, self.key_channels, -1)     # [B, key, K]
        value = self.f_down(context).view(b, self.key_channels, -1)     # [B, key, K]
        value = value.permute(0, 2, 1)                                  # [B, K, key]

        # [B, P, key] @ [B, key, K] -> [B, P, K], scaled then class-softmaxed.
        sim = torch.matmul(query, key) * (self.key_channels ** -0.5)
        sim = F.softmax(sim, dim=-1)

        # [B, P, K] @ [B, K, key] -> [B, P, key] -> [B, key, h, w] -> f_up.
        ctx = torch.matmul(sim, value).permute(0, 2, 1)
        ctx = ctx.reshape(b, self.key_channels, h, w)
        return self.f_up(ctx)


class OCRHead(nn.Module):
    """HRNet-OCR head: soft regions -> gather class vectors -> attend -> classify.

    Args:
        in_channels: width of the branch concat (480 for W32).
        mid_channels: pixel-feature width after the 3x3 entry conv (512).
        key_channels: attention query/key width (256).
        num_classes: 21 for VOC.
        dropout: Dropout2d in the fusion bottleneck (0 disables).

    forward returns (main_logits, aux_logits) in TRAINING mode -- the loss
    weights the aux term -- and main_logits alone in eval mode, so the eval /
    metric code can keep treating the model as logits-in-logits-out.
    """

    def __init__(self, in_channels: int = 480, mid_channels: int = 512,
                 key_channels: int = 256, num_classes: int = 21,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Step 1: soft-region (auxiliary) classifier on the raw concat.
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        # Pixel features for the OCR machinery: 3x3 conv 480 -> 512.
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # Steps 2 + 3.
        self.gather = SpatialGather()
        self.attention = ObjectAttention(mid_channels, key_channels)
        # Fuse [attended context | pixel features] back to mid_channels.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.cls_head = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, feats: List[torch.Tensor], output_size: Tuple[int, int]):
        """4 branch maps -> full-resolution logits (+ aux logits in training).

        Input:
            feats: the backbone's 4 outputs (strides 4/8/16/32).
            output_size: (H, W) of the network input.
        Output:
            training: (main [B,21,H,W], aux [B,21,H,W]) -- feed BOTH to the
                loss (losses/hrnet_loss.py handles the tuple).
            eval:      main [B,21,H,W] only.
        """
        x = _concat_branches(feats)                    # [B, 480, H/4, W/4]

        aux = self.aux_head(x)                         # [B, 21, H/4, W/4]
        pixels = self.conv3x3_ocr(x)                   # [B, 512, H/4, W/4]
        context = self.gather(pixels, aux)             # [B, 512, 21, 1]
        obj_ctx = self.attention(pixels, context)      # [B, 512, H/4, W/4]
        fused = self.bottleneck(torch.cat([obj_ctx, pixels], dim=1))
        main = self.cls_head(fused)                    # [B, 21, H/4, W/4]

        main = F.interpolate(main, size=output_size,
                             mode="bilinear", align_corners=False)
        if not self.training:
            return main
        aux = F.interpolate(aux, size=output_size,
                            mode="bilinear", align_corners=False)
        return main, aux


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/head.py
if __name__ == "__main__":
    feats = [
        torch.randn(2, 32, 120, 120),
        torch.randn(2, 64, 60, 60),
        torch.randn(2, 128, 30, 30),
        torch.randn(2, 256, 15, 15),
    ]

    simple = SimpleHead(in_channels=480, num_classes=21)
    out = simple(feats, output_size=(480, 480))
    print("simple:", tuple(out.shape), "(expected (2, 21, 480, 480))")

    ocr = OCRHead(in_channels=480, num_classes=21)
    ocr.train()
    main, aux = ocr(feats, output_size=(480, 480))
    print("ocr train:", tuple(main.shape), tuple(aux.shape),
          "(expected 2x (2, 21, 480, 480))")
    ocr.eval()
    with torch.no_grad():
        main = ocr(feats, output_size=(480, 480))
    print("ocr eval:", tuple(main.shape), "(expected (2, 21, 480, 480))")

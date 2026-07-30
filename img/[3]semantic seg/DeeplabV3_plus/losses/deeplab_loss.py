"""DeepLab loss: per-pixel cross-entropy. Identical in kind to the FCN loss.

Compare the losses this repo has built so far:

    YOLO3   : box(MSE-ish) + objectness(BCE/focal) + class(BCE), anchor matching
    FCOS    : class(focal)  + box(GIoU)            + centerness(BCE), range assignment
    FCN     : cross-entropy(logits, mask)
    DeepLab : cross-entropy(logits, mask)                      <- this file

DeepLab changes WHERE the resolution comes from (a dilated backbone instead
of an FPN rebuild), but the task contract is unchanged: the model emits
per-pixel logits at full input resolution, the label map already IS one
target per prediction, so one CE term is the whole loss. (DeepLab-v1's paper
adds a DenseCRF -- but that is un-trainable POST-PROCESSING applied after
inference, not a loss term; we skip it.)

nn.CrossEntropyLoss details worth knowing:
  * input:  RAW logits [B, C, H, W] (log-softmax is applied internally --
    numerically stable, same "loss decodes the raw output" contract as the
    other projects);
  * target: class ids [B, H, W], dtype long;
  * ignore_index=255 makes VOC's void contours AND our pad pixels contribute
    exactly zero loss and zero gradient;
  * reduction="mean" (default) averages over the NON-ignored pixels only.

Class imbalance note: background is ~70% of VOC pixels, yet plain CE works
(no focal loss, unlike FCOS). Detection faced "1 object vs ~10k background
locations" per image; here every image contributes tens of thousands of
labeled FOREGROUND pixels, so gradients never drown. If rare classes lag,
per-class weights in CrossEntropyLoss(weight=...) are the first knob to try.
"""

import torch
import torch.nn as nn


class DeepLabLoss(nn.Module):
    """Per-pixel cross-entropy with ignore handling.

    Returns (loss, items) exactly like the detection losses, so train.py's
    accumulate/log loop carries over unchanged. `items` only has {"total"}:
    there are no components to break out.

    Args:
        ignore_index (int): label value excluded from the loss (VOC void
            contours + mask padding; 255).
    """

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor):
        """Compute the loss for one batch.

        Input:
            logits: raw model output [B, num_classes, H, W].
            masks:  GT class ids [B, H, W] (long), values 0..C-1 or 255.

        Output:
            loss:  scalar tensor (for backward()).
            items: {"total": float} for logging (detached).

        Edge case: a batch whose pixels are ALL ignored would yield NaN (0/0
        average). Our pipeline can't produce one (every crop keeps real
        pixels), so we don't special-case it.
        """
        loss = self.ce(logits, masks)
        return loss, {"total": float(loss.detach())}


# ---- Quick self-test: run this file directly ---------------------------------
# python losses/deeplab_loss.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 21, 64, 64
    criterion = DeepLabLoss(ignore_index=255)

    logits = torch.randn(B, C, H, W, requires_grad=True)
    masks = torch.randint(0, C, (B, H, W))

    loss, items = criterion(logits, masks)
    print("loss:", items, "(random logits over 21 classes => ~ln(21) ≈ 3.04)")

    # Ignore check: replacing half the labels with 255 must change the loss
    # only via which pixels are averaged -- and backward must still work.
    masks_ign = masks.clone()
    masks_ign[:, :, : W // 2] = 255
    loss_ign, items_ign = criterion(logits, masks_ign)
    loss_ign.backward()
    grad_ignored = logits.grad[:, :, :, : W // 2].abs().sum()
    print("loss with half ignored:", items_ign)
    print("grad on ignored half:", float(grad_ignored), "(expected 0.0)")

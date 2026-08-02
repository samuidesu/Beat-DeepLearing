"""HRNet loss: per-pixel cross-entropy, with OCR's auxiliary term.

Same task contract as the FCN/DeepLab losses -- the label map already IS one
target per prediction, so cross-entropy is the whole loss. The one new thing
is OCR's AUXILIARY head: during training the model returns (main, aux) and the
total is

    total = CE(main, mask) + aux_weight * CE(aux, mask)

Why an aux loss at all? OCR's "gather" step builds one feature vector per
class from the aux head's soft regions. If those regions were garbage, the
gathered class vectors would be garbage too, and the attention that follows
could not recover. Supervising the aux head directly (at a discounted weight,
0.4 in the paper) keeps the regions honest WITHOUT making them the objective.

The simple head (and the OCR head in eval mode) returns a single tensor; the
loss then degrades to plain CE, so ONE loss class serves both heads and both
modes. NOTE: the logged train "total" therefore includes the aux term while
the val "total" (eval mode) is main-only CE -- compare val curves to val
curves, not train to val.

nn.CrossEntropyLoss details (same as the earlier projects):
  * input: RAW logits [B, C, H, W] (log-softmax applied internally);
  * target: class ids [B, H, W], dtype long;
  * ignore_index=255 -> VOC void contours and pad pixels contribute exactly
    zero loss and zero gradient;
  * reduction="mean" averages over the non-ignored pixels only.
"""

import torch
import torch.nn as nn


class HRNetLoss(nn.Module):
    """Cross-entropy that also understands the OCR head's (main, aux) tuple.

    Returns (loss, items) like every loss in this repo, so train.py's
    accumulate/log loop carries over unchanged. `items` holds {"total"} for a
    single-tensor input, plus {"main", "aux"} when the aux term is present.

    Args:
        ignore_index: label value excluded from the loss (255).
        aux_weight: weight of the auxiliary CE term (0.4, the OCR paper's
            value). Only used when the model hands over a tuple.
    """

    def __init__(self, ignore_index: int = 255, aux_weight: float = 0.4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, logits, masks: torch.Tensor):
        """Compute the loss for one batch.

        Input:
            logits: [B, C, H, W] raw logits, OR the OCR training tuple
                (main [B, C, H, W], aux [B, C, H, W]).
            masks: GT class ids [B, H, W] (long), values 0..C-1 or 255.

        Output:
            loss:  scalar tensor (for backward()).
            items: {"total": float, ...} for logging (detached).
        """
        if isinstance(logits, (tuple, list)):
            main, aux = logits
            main_loss = self.ce(main, masks)
            aux_loss = self.ce(aux, masks)
            loss = main_loss + self.aux_weight * aux_loss
            return loss, {
                "total": float(loss.detach()),
                "main": float(main_loss.detach()),
                "aux": float(aux_loss.detach()),
            }

        loss = self.ce(logits, masks)
        return loss, {"total": float(loss.detach())}


# ---- Quick self-test: run this file directly ---------------------------------
# python losses/hrnet_loss.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 21, 64, 64
    criterion = HRNetLoss(ignore_index=255, aux_weight=0.4)

    main = torch.randn(B, C, H, W, requires_grad=True)
    aux = torch.randn(B, C, H, W, requires_grad=True)
    masks = torch.randint(0, C, (B, H, W))

    # Single tensor -> plain CE (~ln(21) for random logits).
    loss, items = criterion(main, masks)
    print("single:", items)

    # Tuple -> main + 0.4 * aux; check the arithmetic and that backward works.
    loss, items = criterion((main, aux), masks)
    expect = items["main"] + 0.4 * items["aux"]
    print("tuple:", items, f"(total should be ~{expect:.4f})")
    loss.backward()
    print("aux grad flows:", aux.grad is not None and float(aux.grad.abs().sum()) > 0)

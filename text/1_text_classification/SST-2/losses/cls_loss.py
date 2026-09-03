"""Classification loss: cross-entropy over the 2 sentiment classes.

The simplest loss in this repo, and worth stating why. Segmentation ran the
same cross-entropy over H*W predictions per image with an ignore_index for
void pixels; here there is exactly ONE prediction per sample and no positions
to ignore, so the loss collapses to plain CE over [B, 2] logits.

The one addition is LABEL SMOOTHING. With two classes and a confident RNN,
the CE optimum pushes the correct logit towards +inf -- the model keeps
sharpening probabilities long after the decision is right, which is pure
overfitting on a 67k-sentence corpus. Smoothing replaces the hard target
(1, 0) with (1-eps, eps): being right is still rewarded, being *certain* is
not. eps=0.05 is mild; set config.LABEL_SMOOTHING = 0 to disable.

nn.CrossEntropyLoss details (same contract as everywhere else in the repo):
  * input: RAW logits [B, C] -- log_softmax is applied internally, so the
    model must NOT end with a softmax;
  * target: class ids [B], dtype long;
  * reduction="mean" averages over the batch.
"""

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """Cross-entropy with optional label smoothing and class weights.

    Returns (loss, items) like every loss in this repo, so train.py's
    accumulate/log loop carries over unchanged.

    Args:
        label_smoothing: eps in [0, 1); 0 disables.
        class_weights: optional [C] tensor to counter class imbalance. SST-2
            is 44/56 negative/positive -- mild enough that the default None
            (no reweighting) is the right choice; the argument exists because
            the next datasets in this folder may not be so balanced.
    """

    def __init__(self, label_smoothing: float = 0.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """Compute the loss for one batch.

        Input:
            logits: [B, C] raw scores.
            labels: [B] long, class ids.
        Output:
            loss:  scalar tensor (for backward()).
            items: {"loss": float} for logging (detached).
        """
        loss = self.ce(logits, labels)
        return loss, {"loss": float(loss.detach())}


# ---- Quick self-test: run this file directly --------------------------------
# python losses/cls_loss.py
if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    B, C = 8, 2
    labels = torch.randint(0, C, (B,))

    # Uniform logits -> loss = ln(C) = 0.6931 for 2 classes.
    flat = torch.zeros(B, C, requires_grad=True)
    loss, items = ClassificationLoss()(flat, labels)
    print("uniform logits:", items, f"(expected ~{math.log(C):.4f})")

    # Confident and correct -> loss near 0 without smoothing...
    confident = torch.full((B, C), -5.0)
    confident[torch.arange(B), labels] = 5.0
    print("confident, no smoothing: ", ClassificationLoss()(confident, labels)[1])
    # ...but smoothing keeps a floor: the target itself is no longer one-hot.
    print("confident, smoothing 0.05:", ClassificationLoss(0.05)(confident, labels)[1])

    loss, _ = ClassificationLoss(0.05)(flat, labels)
    loss.backward()
    print("grad flows:", flat.grad is not None and float(flat.grad.abs().sum()) > 0)

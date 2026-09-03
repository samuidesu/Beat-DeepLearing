"""Classification head: token features -> one document vector -> 4 logits.

    outputs [B, L, F] (+ final [B, F])  --pool-->  [B, F]  --linear-->  [B, 4]

The segmentation heads had to UPSAMPLE back to per-pixel predictions; a
document classifier does the opposite -- it must COLLAPSE a variable number of
token features into one fixed vector. Three ways to do it, selectable via
config.POOLING:

    "last"  the encoder's final hidden state (forward end + backward end).
            The textbook RNN classifier: everything the recurrence chose to
            remember, and nothing else. Its weakness is that a single vector
            has to survive the whole walk -- and on AG News that walk is ~44
            tokens, not SST-2's 7.
    "max"   element-wise max over time. Each feature dimension reports its
            strongest activation anywhere in the document, which suits topic
            classification: one decisive word ("midfielder", "Nasdaq") should
            be able to carry the prediction regardless of position.
    "mean"  masked average over time -- smoother, but a long document can
            dilute the one phrase that mattered.

MASKING is the part that must be right. Padded positions are not data: for
"max" they are set to -inf before the reduction (a plain max would happily
return a padding activation whenever it is the largest), and for "mean" the
sum is divided by the TRUE length, not by L. Get this wrong and accuracy
quietly drops with batch composition -- the same document scores differently
depending on which others shared its batch.

WHY 4 LOGITS AND NOT 3: with C mutually exclusive classes, softmax over C
logits is over-parameterized by exactly one direction (adding a constant to
every logit changes nothing), so C-1 outputs would suffice. Nobody does that,
because the symmetric form keeps every class's weight vector interpretable,
lets nn.CrossEntropyLoss take the argmax directly, and costs one extra row of
a linear layer. Same reasoning as the binary case in the SST-2 project, where
2 logits were used instead of 1 sigmoid.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Masked pooling + dropout + linear classifier.

    Args:
        in_features: encoder output width (hidden_size * directions).
        num_classes: 4 for AG News.
        pooling: "last" / "max" / "mean".
        dropout: applied to the pooled document vector.
    """

    def __init__(self, in_features: int, num_classes: int = 4,
                 pooling: str = "last", dropout: float = 0.5):
        super().__init__()
        if pooling not in ("last", "max", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        # A single linear layer on purpose: with a 512-dim BiLSTM feature, an
        # extra hidden layer adds parameters and overfitting, not accuracy.
        # The capacity belongs in the encoder.
        self.fc = nn.Linear(in_features, num_classes)

    @staticmethod
    def _mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Build a [B, L] bool mask that is True at REAL token positions.

        arange(L) < length, broadcast over the batch -- the standard trick.
        """
        ar = torch.arange(max_len, device=lengths.device)        # [L]
        return ar[None, :] < lengths[:, None]                    # [B, L]

    def pool(self, outputs: torch.Tensor, final: torch.Tensor,
             lengths: torch.Tensor) -> torch.Tensor:
        """Collapse [B, L, F] token features into [B, F]. See module docstring."""
        if self.pooling == "last":
            # The encoder already extracted this correctly (packing guarantees
            # it is the state after the last REAL token, in both directions).
            return final

        mask = self._mask_from_lengths(lengths, outputs.size(1))[..., None]  # [B, L, 1]
        if self.pooling == "max":
            # -inf on padding so it can never win the max. (Padding rows come
            # back as zeros from the encoder, and 0 > a negative activation --
            # this is exactly the silent bug the masking prevents.)
            masked = outputs.masked_fill(~mask, float("-inf"))
            return masked.max(dim=1).values
        # mean: sum the real positions, divide by the true length.
        summed = (outputs * mask).sum(dim=1)                     # [B, F]
        return summed / lengths.clamp(min=1)[:, None].to(summed.dtype)

    def forward(self, outputs: torch.Tensor, final: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Input: encoder outputs/final + lengths. Output: logits [B, C]."""
        pooled = self.pool(outputs, final, lengths)
        return self.fc(self.dropout(pooled))


# ---- Quick self-test: run this file directly --------------------------------
# python model/head.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, F = 2, 4, 3
    # Row 1 is 2 tokens long; its padded tail holds LARGE values that masked
    # pooling must ignore (an unmasked max/mean would be fooled by them).
    outputs = torch.tensor([
        [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [0., 0., 0.]],
        [[1., 1., 1.], [3., 3., 3.], [99., 99., 99.], [99., 99., 99.]],
    ])
    final = torch.zeros(B, F)
    lengths = torch.tensor([3, 2])

    for pooling in ("max", "mean"):
        head = ClassifierHead(F, num_classes=4, pooling=pooling, dropout=0.0)
        pooled = head.pool(outputs, final, lengths)
        print(f"[{pooling}] pooled row0={pooled[0].tolist()} row1={pooled[1].tolist()}")
    print("expected max:  row0=[1,2,3]  row1=[3,3,3]  (99s masked out)")
    print("expected mean: row0=[.33,.67,1]  row1=[2,2,2]")

    head = ClassifierHead(F, num_classes=4, pooling="last", dropout=0.0)
    logits = head(outputs, final, lengths)
    print("logits:", tuple(logits.shape), "(expected (2, 4))")

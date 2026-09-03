"""Embedding layer: token ids -> word vectors. This project's "backbone".

    ids [B, L]  --lookup-->  vectors [B, L, D]

An embedding is just a lookup table of shape [vocab_size, D] -- row i holds
the vector for token id i -- so the "forward pass" is indexing, and the
backward pass adds gradient only to the rows that appeared in the batch. That
is exactly why the two-stage protocol transfers from the CNN projects:

    stage 1  freeze()   -> the table is a fixed feature extractor (GloVe),
                           the RNN learns to read it
    stage 2  unfreeze() -> the table finetunes at a much smaller LR, so
                           sentiment-specific nuance ("cheesy" is negative for
                           films, neutral for food) can be learned without
                           destroying the general geometry

Two details that are easy to get wrong and cost accuracy silently:

  * padding_idx=0 pins the <pad> row to the zero vector AND blocks gradient
    for it, so padding stays inert even after unfreezing.
  * dropout is applied to the LOOKED-UP VECTORS, not to whole words. (Word-
    level dropout -- randomly replacing tokens with <unk> -- is a different,
    also useful, regularizer; this file does the standard element-wise one.)
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Word-vector table with optional pretrained init and freeze control.

    Args:
        vocab_size: number of rows (len(vocab)).
        embed_dim: vector width (100 for glove.6B.100d).
        pad_idx: row pinned to zero and excluded from gradients.
        pretrained: [vocab_size, embed_dim] float tensor from
            dataset.glove.build_embedding_matrix, or None for random init.
        dropout: dropout probability on the output vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_idx: int = 0,
        pretrained: torch.Tensor = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        # Recorded so train.py can warn when stage 1 would freeze RANDOM
        # weights -- the same guard the HRNet project has for its backbone.
        self.pretrained_loaded = False

        if pretrained is not None:
            if pretrained.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained matrix {tuple(pretrained.shape)} does not match "
                    f"({vocab_size}, {embed_dim}) -- vocab/dim mismatch"
                )
            # copy_ under no_grad: overwrite the values, keep the Parameter.
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained)
                self.embedding.weight[pad_idx].zero_()
            self.pretrained_loaded = True

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Input: ids [B, L] long. Output: vectors [B, L, D] float."""
        return self.dropout(self.embedding(ids))

    # ---- two-stage finetuning helpers ----
    def freeze(self):
        """Stage 1: stop training the word vectors."""
        self.embedding.weight.requires_grad_(False)

    def unfreeze(self):
        """Stage 2: let the word vectors finetune (at a small LR)."""
        self.embedding.weight.requires_grad_(True)


# ---- Quick self-test: run this file directly --------------------------------
# python model/embedding.py
if __name__ == "__main__":
    torch.manual_seed(0)
    V, D = 50, 8
    pre = torch.randn(V, D)
    emb = TokenEmbedding(V, D, pad_idx=0, pretrained=pre, dropout=0.0)

    ids = torch.tensor([[5, 7, 0, 0], [3, 4, 9, 0]])  # 0 = <pad>
    out = emb(ids)
    print("out:", tuple(out.shape), "(expected (2, 4, 8))")
    print("pad vector is zero:", bool(out[0, 2].abs().sum() == 0))
    print("row 5 matches pretrained:", bool(torch.allclose(out[0, 0], pre[5])))

    # Freeze / unfreeze must flip requires_grad, and a frozen table must
    # receive no gradient at all.
    emb.freeze()
    print("frozen requires_grad:", emb.embedding.weight.requires_grad, "(expected False)")
    emb.unfreeze()
    emb(ids).sum().backward()
    grad = emb.embedding.weight.grad
    print(
        "pad row grad after backward:",
        float(grad[0].abs().sum()),
        "(expected 0.0 -- padding_idx blocks it)",
    )
    print("row 5 grad:", float(grad[5].abs().sum()), "(expected > 0)")

"""Full model: GloVe embedding -> RNN encoder -> pooled linear classifier.

Forward pass:
    ids [B, L] + lengths [B]
      --embedding-->  vectors [B, L, 100]          (the "backbone": GloVe)
      --encoder---->  outputs [B, L, 512]          (the "neck": BiLSTM/GRU/RNN)
                      final   [B, 512]
      --head------->  logits  [B, 4]               (pool + dropout + linear)

The three-part split is deliberately the same one the segmentation projects
used (backbone / neck / head), because the two-stage finetuning protocol is
defined on it:

    Stage 1: freeze the EMBEDDING (pretrained GloVe); train encoder + head,
             which start from random weights.
    Stage 2: unfreeze everything, three LR tiers -- embedding slowest,
             encoder middle, head fastest.

parameter_groups() returns exactly those three tiers, so train.py's
build_layered_optimizer is the same function as in the DeepLab/HRNet projects.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

import torch
import torch.nn as nn

try:
    from .embedding import TokenEmbedding
    from .encoder import RNNEncoder
    from .head import ClassifierHead
except ImportError:  # running this file directly
    from embedding import TokenEmbedding
    from encoder import RNNEncoder
    from head import ClassifierHead


class RNNClassifier(nn.Module):
    """Document classifier for AG News.

    Args:
        vocab_size: len(vocab).
        num_classes: 4.
        embed_dim: word-vector width (100 for glove.6B.100d).
        hidden_size: RNN hidden width per direction.
        cell: "rnn" / "lstm" / "gru".
        num_layers / bidirectional: encoder shape.
        pooling: "last" / "max" / "mean" (see head.py).
        dropout: shared by embedding, between encoder layers, and before fc.
        pad_idx: padding id (0).
        pretrained_vectors: [vocab_size, embed_dim] GloVe matrix, or None to
            train the word vectors from scratch.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embed_dim: int = 100,
        hidden_size: int = 256,
        cell: str = "lstm",
        num_layers: int = 2,
        bidirectional: bool = True,
        pooling: str = "last",
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cell = cell
        self.pooling = pooling

        self.embedding = TokenEmbedding(
            vocab_size, embed_dim, pad_idx=pad_idx,
            pretrained=pretrained_vectors, dropout=dropout)
        self.encoder = RNNEncoder(
            input_size=embed_dim, hidden_size=hidden_size, cell=cell,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.head = ClassifierHead(
            in_features=self.encoder.out_dim, num_classes=num_classes,
            pooling=pooling, dropout=dropout)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Input: ids [B, L] long, lengths [B] long. Output: logits [B, C]."""
        vectors = self.embedding(ids)
        outputs, final = self.encoder(vectors, lengths)
        return self.head(outputs, final, lengths)

    # ---- Two-stage finetuning helpers (same interface as the CNN models) ----
    def freeze_embedding(self):
        """Stage 1: freeze the pretrained word vectors."""
        self.embedding.freeze()

    def unfreeze_all(self):
        """Stage 2: unfreeze everything for the layered-LR finetune."""
        self.embedding.unfreeze()

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Split parameters into the three layered-LR groups.

        Output (dict of lists, ready for torch.optim param_groups):
            "embedding" -> pretrained GloVe table   (slowest LR)
            "encoder"   -> the recurrent layers     (middle LR)
            "head"      -> pooling + classifier     (fastest LR)

        Frozen parameters are EXCLUDED, so the same call serves both stages:
        in stage 1 "embedding" comes back empty and train.py skips that tier.
        """
        return {
            "embedding": [p for p in self.embedding.parameters() if p.requires_grad],
            "encoder": [p for p in self.encoder.parameters() if p.requires_grad],
            "head": [p for p in self.head.parameters() if p.requires_grad],
        }

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the parameters that currently require gradients."""
        return (p for p in self.parameters() if p.requires_grad)


# ---- Quick self-test: run this file directly --------------------------------
# python model/rnn_classifier.py
if __name__ == "__main__":
    torch.manual_seed(0)
    V, B, L = 1000, 4, 12
    ids = torch.randint(2, V, (B, L))
    lengths = torch.tensor([12, 9, 5, 1])
    for i, n in enumerate(lengths):          # zero out the padded tails
        ids[i, n:] = 0

    for cell in ("rnn", "lstm", "gru"):
        model = RNNClassifier(V, cell=cell, hidden_size=64, num_layers=2)
        model.eval()
        with torch.no_grad():
            logits = model(ids, lengths)
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[{cell}] logits {tuple(logits.shape)} (expected (4, 4))  "
              f"params={n_par:.2f}M")

    # Stage-1 freeze: the embedding tier must disappear from the groups.
    model.freeze_embedding()
    for name, params in model.parameter_groups().items():
        n = sum(p.numel() for p in params) / 1e6
        print(f"stage-1 group {name:<9}: {n:.2f}M trainable")
    model.unfreeze_all()
    for name, params in model.parameter_groups().items():
        n = sum(p.numel() for p in params) / 1e6
        print(f"stage-2 group {name:<9}: {n:.2f}M trainable")

"""Recurrent encoder: word vectors -> contextual features. The "neck".

    vectors [B, L, D]  --RNN-->  outputs [B, L, H*dirs]  +  final [B, H*dirs]

A recurrent layer walks the document one token at a time, carrying a hidden
state that summarizes everything read so far:

    vanilla RNN   h_t = tanh(W x_t + U h_{t-1} + b)
    GRU           two gates decide how much of h_{t-1} to keep and how much of
                  the new candidate to write
    LSTM          three gates (input/forget/output) around a separate CELL
                  state c_t that flows through the sequence with only
                  element-wise multiply-add on its path

That last point is the whole story of the family. Backpropagating through T
steps multiplies by the recurrent Jacobian T times, so a vanilla RNN's
gradient shrinks (or explodes) geometrically and it effectively cannot learn
dependencies more than ~10 tokens apart. The gated cells give the gradient an
additive highway.

AG News is where that difference should finally be measurable. SST-2's median
sentence was 7 tokens -- short enough that the Elman RNN's memory horizon was
never really the binding constraint, and the three cells landed within 2
points of each other. Here the median document is 44 tokens, six times longer,
and with "last" pooling every prediction depends on a state that survived the
entire walk. config.CELL picks which cell runs, everything else stays
identical -- see the README comparison for whether the gap actually widened.

Two mechanics implemented here that are specific to variable-length text:

  PACKING. A padded batch is a rectangle, but the documents inside are not.
  Feeding <pad> into the recurrence would keep updating the hidden state after
  the document ended, so the "final" state would be the state after k padding
  steps -- a different (and wrong) vector for every length.
  pack_padded_sequence reorganizes the batch into the per-timestep groups of
  still-active rows, so cuDNN steps each row exactly `length` times.

  BIDIRECTIONALITY. A forward RNN's representation of token 3 knows nothing
  about token 40. A second layer reading right-to-left fixes that; the two
  directions are concatenated, which is why the head sees 2*H features.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# The three cells this project compares, mapped to their torch modules. They
# share an identical call signature, which is exactly what makes the
# single-variable comparison possible.
_CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RNNEncoder(nn.Module):
    """Multi-layer (optionally bidirectional) recurrent encoder.

    Args:
        input_size: embedding width D.
        hidden_size: hidden width PER DIRECTION.
        cell: "rnn" / "lstm" / "gru".
        num_layers: stacked recurrent layers (each reads the one below).
        bidirectional: run a second pass right-to-left and concatenate.
        dropout: applied BETWEEN stacked layers by torch (ignored when
            num_layers == 1, which torch warns about -- hence the guard).
    """

    def __init__(self, input_size: int, hidden_size: int, cell: str = "lstm",
                 num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        if cell not in _CELLS:
            raise ValueError(f"unknown cell {cell!r}; choose one of {list(_CELLS)}")
        self.cell = cell
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                       # [B, L, *] everywhere
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # nonlinearity is an nn.RNN-only argument; tanh is the Elman default
        # and the one the LSTM/GRU candidate states use too, so the three
        # cells stay comparable.
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.rnn = _CELLS[cell](**kwargs)

    @property
    def out_dim(self) -> int:
        """Feature width handed to the head (hidden_size * directions)."""
        return self.hidden_size * self.num_directions

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """Encode one padded batch.

        Input:
            x: embedded tokens [B, L, D].
            lengths: true lengths [B] (long). Must be on CPU for packing --
                this method moves them, so callers can pass a CUDA tensor.

        Output:
            outputs: per-token features [B, L, H*dirs], padded rows zeroed.
            final:   document-level state [B, H*dirs] -- the LAST layer's
                final hidden state, forward and backward concatenated.
        """
        total_length = x.size(1)

        # enforce_sorted=False lets torch sort/unsort internally, so the batch
        # can stay in dataset order (sorting by length ourselves would also
        # correlate batches with document length, which we do not want).
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        # total_length pins the output width back to L even when the longest
        # row was shorter (defensive: our collate makes them equal).
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=total_length)

        # LSTM returns (h_n, c_n); we want h_n. The cell state c_n is the
        # LSTM's internal memory -- useful to inspect, not what a classifier
        # reads.
        h_n = hidden[0] if self.cell == "lstm" else hidden

        # h_n is [num_layers * num_directions, B, H] with layers-major order.
        # Reshape, take the TOP layer, then concatenate its directions.
        B = x.size(0)
        h_n = h_n.view(self.num_layers, self.num_directions, B, self.hidden_size)
        top = h_n[-1]                                   # [dirs, B, H]
        final = torch.cat([top[i] for i in range(self.num_directions)], dim=1)
        return outputs, final                           # [B, L, H*dirs], [B, H*dirs]


# ---- Quick self-test: run this file directly --------------------------------
# python model/encoder.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, H = 3, 6, 8, 5
    x = torch.randn(B, L, D)
    lengths = torch.tensor([6, 4, 1])
    x[1, 4:] = 0                                        # zero the padded tail
    x[2, 1:] = 0

    for cell in ("rnn", "lstm", "gru"):
        enc = RNNEncoder(D, H, cell=cell, num_layers=2, bidirectional=True, dropout=0.0)
        enc.eval()
        with torch.no_grad():
            out, final = enc(x, lengths)
        print(f"[{cell}] outputs {tuple(out.shape)} (expected (3, 6, 10))  "
              f"final {tuple(final.shape)} (expected (3, 10))")

    # Packing check: padded positions must come back as exact zeros...
    print("padded positions zero:", bool(out[1, 4:].abs().sum() == 0))
    # ...and the forward half of `final` must equal the output at the LAST
    # REAL token (that is what "final hidden state" means).
    fwd_final = final[1, :H]
    fwd_at_last_real = out[1, lengths[1] - 1, :H]
    print("final == output at last real token:",
          bool(torch.allclose(fwd_final, fwd_at_last_real, atol=1e-6)))

    # Padding-invariance: re-encoding a batch with MORE padding must not
    # change the result. This is the bug packing exists to prevent.
    x_pad = torch.cat([x, torch.zeros(B, 4, D)], dim=1)
    with torch.no_grad():
        _, final2 = enc(x_pad, lengths)
    print("invariant to extra padding:", bool(torch.allclose(final, final2, atol=1e-6)))

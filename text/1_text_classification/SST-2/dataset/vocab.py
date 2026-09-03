"""Tokenizer + vocabulary: the text counterpart of the image transforms.

In the CNN projects, dataset/transforms.py turned a PIL image into a float
tensor (resize, crop, normalize). Here the same job -- "raw sample -> tensor
the model can read" -- is done in two steps:

    sentence  --tokenize-->  ["a", "charming", "film"]     (strings)
    tokens    --Vocab----->  [12, 4839, 210]               (integer ids)

The ids then index the embedding table, which is where the actual numbers
(word vectors) live. Two properties matter and both are enforced here:

  1. The vocabulary is built from the TRAINING split only. A vocab built over
     dev/test would let evaluation words influence training-time decisions --
     the text equivalent of computing normalization stats over the val set.
  2. The mapping must be IDENTICAL at train and inference time, or every id
     points at the wrong vector. Hence save()/load(): train.py writes
     vocab.json next to the checkpoint, eval.py and predict.py read it back.

Special tokens are pinned at fixed ids (config.PAD_IDX=0, config.UNK_IDX=1):
  <pad> fills short sentences up to the batch length and is masked out
        everywhere (padding_idx in the embedding, packing in the encoder,
        masks in the pooling);
  <unk> catches words that were never seen in training.
"""

import json
import os
import re
import sys
from collections import Counter

# Make the project root importable so `import config` works whether this file
# is run directly (python dataset/vocab.py) or imported as dataset.vocab.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402

# One token = a run of letters/digits/apostrophes, OR a single other non-space
# character (so punctuation becomes its own token instead of sticking to the
# previous word).
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list:
    """Split a sentence into lowercase tokens.

    GLUE's SST-2 tsv is ALREADY tokenized PTB-style ("it 's a charming ...",
    "does n't"), so on the corpus itself a plain .split() would do. The regex
    exists for predict.py, where a user types real text with attached
    punctuation ("Don't waste your time!") -- running the same function on
    both paths guarantees train/inference tokenization can never drift apart.

    GloVe 6B is lowercase-only, so lowercasing here is not just normalization:
    it is what makes a word findable in the pretrained table at all.

    Input:  raw sentence string.
    Output: list of token strings.
    """
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Bidirectional token <-> id mapping with frequency-based pruning.

    Attributes:
        itos: list of tokens, indexed by id (id -> string).
        stoi: dict token -> id (string -> id).
        freqs: Counter of raw training frequencies (kept for the log/report).
    """

    def __init__(self, itos, freqs=None):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.freqs = Counter(freqs or {})

    # ---- construction --------------------------------------------------
    @classmethod
    def build(cls, token_lists, min_freq=1, max_size=None,
              specials=(config.PAD_TOKEN, config.UNK_TOKEN)):
        """Build a vocabulary from an iterable of token lists.

        Input:
            token_lists: iterable of lists of tokens (the TRAIN split).
            min_freq: drop tokens appearing fewer than this many times. They
                become <unk> at lookup time, which is exactly how unseen dev
                words behave -- so a higher min_freq also TEACHES the model
                what <unk> looks like instead of leaving that embedding
                untrained.
            max_size: cap on the number of non-special tokens (None = no cap).
            specials: tokens forced to the front, in order, so <pad> gets id 0
                and <unk> id 1 (config.PAD_IDX / UNK_IDX).

        Output:
            a Vocab instance.
        """
        freqs = Counter()
        for tokens in token_lists:
            freqs.update(tokens)

        # Sort by frequency (descending), ties broken alphabetically so the
        # vocabulary -- and therefore every id -- is fully deterministic.
        ordered = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [tok for tok, n in ordered if n >= min_freq]
        if max_size is not None:
            kept = kept[:max_size]
        return cls(list(specials) + kept, freqs)

    # ---- lookup --------------------------------------------------------
    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def encode(self, tokens, max_len=None) -> list:
        """Map tokens to ids, unknown -> UNK, truncating to `max_len`.

        Truncation keeps the FIRST max_len tokens. For sentiment that is a
        real (if minor) choice: the end of a review often carries the verdict,
        so on long documents keeping the tail can work better. SST-2 sentences
        are short enough that config.MAX_LEN=64 never triggers.
        """
        ids = [self.stoi.get(tok, config.UNK_IDX) for tok in tokens]
        return ids[:max_len] if max_len else ids

    def decode(self, ids) -> list:
        """Map ids back to token strings (for debugging / printing)."""
        return [self.itos[i] if 0 <= i < len(self.itos) else config.UNK_TOKEN
                for i in ids]

    # ---- persistence ---------------------------------------------------
    def save(self, path: str):
        """Write the vocabulary as json (itos is enough to rebuild stoi)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos,
                       "freqs": {t: self.freqs[t] for t in self.itos
                                 if t in self.freqs}},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        """Read back a vocabulary written by save()."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["itos"], payload.get("freqs"))


# ---- Quick self-test: run this file directly --------------------------------
# python dataset/vocab.py
if __name__ == "__main__":
    print("tokenize:", tokenize("Don't waste your time -- it 's TERRIBLE!"))

    corpus = [tokenize(s) for s in [
        "a charming film",
        "a charming and funny film",
        "a dull film",
    ]]
    v = Vocab.build(corpus, min_freq=1)
    # 3 specials-free types appear >=2 times: a(3) charming(2) film(3).
    print("size:", len(v), "(expected 2 specials + 6 types = 8)")
    print("itos[:5]:", v.itos[:5], "(pad, unk, then most frequent first)")
    print("pad id:", v.stoi[config.PAD_TOKEN], "unk id:", v.stoi[config.UNK_TOKEN],
          "(expected 0 and 1)")

    ids = v.encode(tokenize("a dull unseen-word film"), max_len=10)
    print("encode:", ids)
    print("decode:", v.decode(ids), "(unseen word -> <unk>)")

    # min_freq prunes: only a/film survive at >=3, charming(2) becomes <unk>.
    v2 = Vocab.build(corpus, min_freq=3)
    print("min_freq=3 size:", len(v2), "->", v2.itos)

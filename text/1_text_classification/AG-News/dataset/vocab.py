"""Tokenizer + vocabulary: the text counterpart of the image transforms.

    document  --tokenize-->  ["wall", "st", ".", "bears", "claw", ...]
    tokens    --Vocab----->  [412, 9038, 4, 2211, 15577, ...]

The ids then index the embedding table, which is where the actual numbers
(word vectors) live. Two properties matter and both are enforced here:

  1. The vocabulary is built from the TRAINING portion only. Building it over
     val/test would let evaluation words influence training-time decisions --
     the text equivalent of computing normalization stats over the val set.
  2. The mapping must be IDENTICAL at train and inference time, or every id
     points at the wrong vector. Hence save()/load(): train.py writes
     vocab.json next to the checkpoint, eval.py and predict.py read it back.

Special tokens are pinned at fixed ids (config.PAD_IDX=0, config.UNK_IDX=1):
  <pad> fills short documents up to the batch length and is masked out
        everywhere (padding_idx in the embedding, packing in the encoder,
        masks in the pooling);
  <unk> catches words below min_freq or never seen in training.

Unchanged from the SST-2 project on purpose: the tokenizer is what decides
whether a word can be found in GloVe at all, so keeping it identical is what
makes the two projects' GloVe-coverage numbers comparable.
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
    """Split a document into lowercase tokens.

    AG News is RAW text, not pre-tokenized the way GLUE's SST-2 tsv was:
    "Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics".
    So unlike in the SST-2 project, the regex is not a safety net for
    predict.py -- it is doing the real work on the corpus itself. Splitting
    punctuation off is what turns "Street's" into ["street", "'", "s"] and
    "ultra-cynics" into ["ultra", "-", "cynics"], both of whose pieces GloVe
    knows, instead of two out-of-vocabulary strings.

    GloVe 6B is lowercase-only, so lowercasing here is not just normalization:
    it is what makes a word findable in the pretrained table at all.

    Input:  raw text string.
    Output: list of token strings.
    """
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Bidirectional token <-> id mapping with frequency-based pruning.

    Attributes:
        itos: list of tokens, indexed by id (id -> string).
        stoi: dict token -> id (string -> id).
    """

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    # ---- construction --------------------------------------------------
    @classmethod
    def build(cls, token_lists, min_freq=1, max_size=None,
              specials=(config.PAD_TOKEN, config.UNK_TOKEN)):
        """Build a vocabulary from an iterable of token lists.

        Input:
            token_lists: iterable of lists of tokens (the TRAIN portion).
            min_freq: drop tokens appearing fewer than this many times. They
                become <unk> at lookup time, which is exactly how unseen test
                words behave -- so min_freq > 1 also TEACHES the model what
                <unk> looks like instead of leaving that embedding untrained.
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
        return cls(list(specials) + kept)

    # ---- lookup --------------------------------------------------------
    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def encode(self, tokens, max_len=None) -> list:
        """Map tokens to ids, unknown -> UNK, truncating to `max_len`.

        Truncation keeps the FIRST max_len tokens, which suits AG News
        particularly well: every row is "title + lead paragraph", i.e. written
        by a journalist to put the topic in the first few words. Whatever
        falls off the end of a 128-token document is the least informative
        part of it.
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
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        """Read back a vocabulary written by save()."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])


# ---- Quick self-test: run this file directly --------------------------------
# python dataset/vocab.py
if __name__ == "__main__":
    raw = "Wall St. Bears Claw Back Into the Black (Reuters)"
    print("tokenize:", tokenize(raw))

    corpus = [tokenize(s) for s in [
        "Oil prices rise on supply fears",
        "Oil prices fall as supply returns",
        "Yankees beat Red Sox",
    ]]
    v = Vocab.build(corpus, min_freq=1)
    print("size:", len(v), "(expected 2 specials + 13 types = 15)")
    print("itos[:6]:", v.itos[:6], "(pad, unk, then most frequent first)")
    print("pad id:", v.stoi[config.PAD_TOKEN], "unk id:", v.stoi[config.UNK_TOKEN],
          "(expected 0 and 1)")

    ids = v.encode(tokenize("oil prices and nasdaq"), max_len=10)
    print("encode:", ids)
    print("decode:", v.decode(ids), "(unseen words -> <unk>)")

    # min_freq=2 prunes everything seen once: only oil/prices/supply survive.
    v2 = Vocab.build(corpus, min_freq=2)
    print("min_freq=2 size:", len(v2), "->", v2.itos)

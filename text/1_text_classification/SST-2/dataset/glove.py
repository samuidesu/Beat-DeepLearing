"""GloVe word vectors: download + build the pretrained embedding matrix.

GloVe is this project's ImageNet checkpoint. The CNN projects loaded a
backbone trained to recognize objects; here we load a 400k-word table trained
(by counting co-occurrences over 6 billion tokens of Wikipedia + Gigaword) to
place related words near each other. Either way the pattern is the same: start
from general knowledge learned on a huge corpus, then finetune on the small
task-specific dataset.

Why it matters so much on SST-2: the training set has ~67k short fragments and
~15k distinct words, many appearing once or twice. Learning a useful 100-dim
vector for "delightful" from a handful of examples is hopeless; GloVe already
knows it sits next to "charming" and "wonderful", so the RNN can generalize
from one to the other on day one. Expect several accuracy points from this
alone -- the README records the from-scratch (--no-glove) comparison.

The file format is plain text, one word per line:
    the 0.418 0.24968 -0.41242 ...      (1 + 100 whitespace-separated fields)

Only the words in OUR vocabulary are kept: 400k x 100 floats would be 160 MB
of embedding table for a model that can only ever look up ~15k of them.

How to download: python dataset/glove.py --download   (862 MB zip, one time)
"""

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.download import download_with_mirrors, extract_zip  # noqa: E402

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
_GLOVE_ZIP = "glove.6B.zip"
# HuggingFace's mirror of the Stanford release first: it is fast, supports byte
# ranges (so the segmented downloader gets its full speedup) and unlike
# nlp.stanford.edu it does not go offline for days.
_GLOVE_URLS = [
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
]


def glove_present(path: str = None) -> bool:
    """True when the configured glove.6B.<dim>d.txt file exists."""
    return os.path.isfile(path or config.GLOVE_PATH)


def download_glove(path: str = None):
    """Fetch + extract the GloVe vectors named by config.GLOVE_NAME.

    The zip holds all four dimensionalities (50d/100d/200d/300d, ~2 GB
    unpacked); only the one file we actually use is extracted.
    """
    path = path or config.GLOVE_PATH
    if glove_present(path):
        print(f"GloVe already present at {path}")
        return
    dest_dir = os.path.dirname(path)
    os.makedirs(dest_dir, exist_ok=True)
    archive = os.path.join(dest_dir, _GLOVE_ZIP)
    print("Downloading GloVe 6B (~862 MB zip, one time)...")
    download_with_mirrors(_GLOVE_URLS, archive, md5=None)
    extract_zip(archive, dest_dir, members=[os.path.basename(path)])
    if not glove_present(path):
        raise RuntimeError(f"{path} missing after extraction")
    # Keep the 347 MB txt, drop the 862 MB zip.
    os.remove(archive)


# -----------------------------------------------------------------------------
# Embedding matrix
# -----------------------------------------------------------------------------
def build_embedding_matrix(vocab, dim: int = None, path: str = None,
                           verbose: bool = True):
    """Build the [len(vocab), dim] pretrained embedding matrix for `vocab`.

    Scans the GloVe text file ONCE and keeps only the rows whose word is in
    the vocabulary (a dict lookup per line; 400k lines take a few seconds).

    Initialization of the misses matters as much as the hits:
      * <pad> stays EXACTLY zero -- it must contribute nothing, and the
        embedding's padding_idx keeps it at zero during training too.
      * every other out-of-GloVe word (typos, rare names, "n't"-style
        fragments) gets N(0, 0.1) noise, so it starts small and unopinionated
        next to the real vectors rather than dominating them.

    Input:
        vocab: the Vocab (its .itos gives the row order).
        dim: vector width; defaults to config.EMBED_DIM (must match the file).
        path: glove.6B.<dim>d.txt; defaults to config.GLOVE_PATH.
        verbose: print the coverage summary.

    Output:
        (matrix [V, dim] float tensor, n_found int).
    """
    dim = dim or config.EMBED_DIM
    path = path or config.GLOVE_PATH
    if not glove_present(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/glove.py --download` "
            f"(or train with --no-glove)")

    # Random init first, then overwrite the rows GloVe knows about.
    matrix = torch.randn(len(vocab), dim) * 0.1
    matrix[config.PAD_IDX].zero_()

    wanted = vocab.stoi
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            # rsplit on the FIRST space only: a handful of GloVe "words" are
            # punctuation that would confuse a naive split-then-take-[0].
            word, _, rest = line.rstrip().partition(" ")
            idx = wanted.get(word)
            if idx is None:
                continue
            vec = [float(v) for v in rest.split(" ")]
            if len(vec) != dim:
                raise ValueError(
                    f"{path} has {len(vec)}-dim vectors but config.EMBED_DIM={dim}")
            matrix[idx] = torch.tensor(vec)
            found += 1

    if verbose:
        # Coverage below ~90% usually means a tokenization mismatch (e.g.
        # forgetting to lowercase), not a genuinely exotic corpus.
        print(f"[glove] {found}/{len(vocab)} vocabulary words found "
              f"({found / len(vocab):.1%} coverage), dim={dim}")
    return matrix, found


# ---- Quick self-test / downloader: run this file directly --------------------
# python dataset/glove.py --download
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect GloVe vectors")
    p.add_argument("--download", action="store_true", help="download glove.6B first")
    args = p.parse_args()

    if args.download:
        download_glove()

    try:
        from .sst2 import build_vocab_from_train
    except ImportError:
        from sst2 import build_vocab_from_train

    vocab = build_vocab_from_train()
    matrix, found = build_embedding_matrix(vocab)
    print(f"matrix: {tuple(matrix.shape)}  pad row norm: "
          f"{matrix[config.PAD_IDX].norm():.4f} (expected 0.0000)")

    # Sanity check the semantics: with real GloVe vectors, "good" must be
    # closer to "great" than to "terrible" under cosine similarity.
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a[None], b[None]).item()

    if all(w in vocab for w in ("good", "great", "terrible")):
        g, gr, t = (matrix[vocab.stoi[w]] for w in ("good", "great", "terrible"))
        print(f"cos(good, great)    = {cos(g, gr):.4f}")
        print(f"cos(good, terrible) = {cos(g, t):.4f}  (should be clearly lower)")

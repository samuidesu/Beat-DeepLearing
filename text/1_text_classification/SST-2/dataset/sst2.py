"""SST-2 (GLUE) sentiment corpus: download, parsing, Dataset, collate_fn.

Responsibilities of this file (the text mirror of the CNN projects'
dataset/voc.py):
  1. Download the GLUE SST-2 archive into config.DATA_ROOT and extract it
     (mirror fallback + resumable, like the VOC download).
  2. Parse the tsv splits into (sentence, label) pairs.
  3. Wrap them in a Dataset that yields (ids LongTensor, length, label).
  4. Provide the COLLATE FUNCTION that pads a list of variable-length
     sequences into one rectangular batch.

Point 4 is the structural novelty of text data. Segmentation crops were all
CROP_SIZE x CROP_SIZE, so torch's default collate could stack them; sentences
have different lengths, so somebody must pad. We pad DYNAMICALLY -- to the
longest sentence in THIS batch, not to config.MAX_LEN -- which keeps the RNN's
unrolled length as short as the batch allows (a batch of 9-token fragments
costs 9 steps, not 64). The lengths travel with the batch so the encoder can
pack them and the head can mask them; without that, the model would happily
read <pad> tokens as if they were words.

The data itself:
    train.tsv  67,349 rows "sentence \\t label"  (phrases from the treebank,
               many are short fragments -- that is the official GLUE train
               set, not an accident)
    dev.tsv       872 rows "sentence \\t label"  (complete sentences)
    test.tsv    1,821 rows "index \\t sentence"  -- NO labels: GLUE keeps them
               server-side, which is why every SST-2 number you see quoted
               (this project included) is a DEV number.

How to download: python dataset/sst2.py --download
(train.py also downloads automatically when the corpus is missing.)
"""

import argparse
import os
import sys

import torch
from torch.utils.data import Dataset

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.download import download_with_mirrors, extract_zip  # noqa: E402

try:
    from .vocab import Vocab, tokenize
except ImportError:  # running this file directly
    from vocab import Vocab, tokenize

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
_SST2_FILENAME = "SST-2.zip"
# Tried in order. The Facebook AI mirror is the fast, stable one used by most
# GLUE tooling; the Firebase link is the original download from the GLUE site.
_SST2_URLS = [
    "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations."
    "appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-"
    "cf6337f84ac8",
]
# Expected row counts per split (header excluded). Used INSTEAD of an md5:
# GLUE publishes no checksum and the two mirrors need not be byte-identical,
# but a truncated or wrong archive cannot possibly produce these counts.
_EXPECTED_ROWS = {"train": 67349, "dev": 872, "test": 1821}


def sst2_present() -> bool:
    """True when the three extracted tsv files exist under config.SST2_DIR."""
    return all(os.path.isfile(os.path.join(config.SST2_DIR, f"{s}.tsv"))
               for s in ("train", "dev", "test"))


def download_sst2():
    """Fetch + extract GLUE SST-2 into config.DATA_ROOT (idempotent)."""
    if sst2_present():
        print(f"SST-2 already present at {config.SST2_DIR}")
        return
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    archive = os.path.join(config.DATA_ROOT, _SST2_FILENAME)
    print("Downloading GLUE SST-2 (~7 MB)...")
    # 4 connections, not 16: the file is small enough that the per-connection
    # handshake would cost more than the parallelism saves.
    download_with_mirrors(_SST2_URLS, archive, md5=None, connections=4)
    # The zip already contains a top-level SST-2/ folder, so extract into
    # DATA_ROOT and the tsv files land exactly at config.SST2_DIR.
    extract_zip(archive, config.DATA_ROOT)
    if not sst2_present():
        raise RuntimeError(f"extraction did not produce {config.SST2_DIR}")
    os.remove(archive)   # the tsv files are what we keep; the zip is 7 MB of dead weight


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def read_split(split: str):
    """Read one tsv split into a list of (sentence, label) pairs.

    Input:
        split: "train" / "dev" / "test".
    Output:
        list of (sentence str, label int). The unlabeled test split gets
        label = -1 so downstream code can still batch it (the metrics simply
        must not be computed on it).
    """
    path = os.path.join(config.SST2_DIR, f"{split}.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/sst2.py --download`")

    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    header, rows = lines[0], lines[1:]
    labeled = "label" in header.split("\t")
    out = []
    for row in rows:
        parts = row.split("\t")
        if labeled:
            # "sentence \t label"
            out.append((parts[0].strip(), int(parts[-1])))
        else:
            # "index \t sentence" -- the id column is useless to us.
            out.append((parts[1].strip(), -1))

    expected = _EXPECTED_ROWS.get(split)
    if expected is not None and len(out) != expected:
        print(f"[sst2] WARNING: {split}.tsv has {len(out)} rows, "
              f"expected {expected} -- corrupt or non-standard copy?")
    return out


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SST2Dataset(Dataset):
    """One SST-2 split as (token ids, length, label) samples.

    The whole corpus is tokenized and encoded ONCE in __init__ (67k short
    sentences -> a fraction of a second, a few MB of python lists), so
    __getitem__ is a pure tensor construction and num_workers=0 costs nothing.
    Image datasets cannot do this -- decoding 10k JPEGs up front would blow up
    memory -- which is why this project runs with 0 dataloader workers while
    the segmentation ones used 4.

    Args:
        split: "train" / "dev" / "test".
        vocab: the Vocab to encode with. Pass the TRAIN vocab for every split.
        max_len: truncation length (config.MAX_LEN).
    """

    def __init__(self, split: str, vocab: Vocab, max_len: int = None):
        self.split = split
        self.vocab = vocab
        self.max_len = max_len or config.MAX_LEN

        pairs = read_split(split)
        self.sentences = [s for s, _ in pairs]
        self.labels = [y for _, y in pairs]
        self.tokens = [tokenize(s) for s in self.sentences]
        self.ids = [vocab.encode(t, self.max_len) for t in self.tokens]
        # A sentence must never be empty: an all-punctuation fragment could
        # tokenize to [] and pack_padded_sequence rejects length-0 sequences.
        self.ids = [ids if ids else [config.UNK_IDX] for ids in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Output: (ids [L] long, length int, label int)."""
        ids = self.ids[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), self.labels[idx]

    # ---- small helpers used by the report / logs ----
    def unk_rate(self) -> float:
        """Fraction of tokens that map to <unk> (vocabulary coverage check)."""
        total = sum(len(ids) for ids in self.ids)
        unks = sum(sum(1 for i in ids if i == config.UNK_IDX) for ids in self.ids)
        return unks / max(total, 1)

    def label_counts(self) -> dict:
        """Class histogram, e.g. {"negative": 29780, "positive": 37569}."""
        out = {name: 0 for name in config.CLASS_NAMES}
        for y in self.labels:
            if 0 <= y < len(config.CLASS_NAMES):
                out[config.CLASS_NAMES[y]] += 1
        return out


def collate_batch(batch):
    """Pad a list of samples into one rectangular batch.

    This is the text analogue of detection's custom collate_fn -- and it
    exists for the same reason: the default collate can only stack tensors of
    identical shape.

    Input:
        batch: list of (ids [L_i], length_i, label_i) from SST2Dataset.
    Output:
        ids:     [B, L_max] long, padded with config.PAD_IDX (0)
        lengths: [B] long, the TRUE length of each row (before padding)
        labels:  [B] long
    where L_max is the longest sentence IN THIS BATCH (dynamic padding).
    """
    seqs, lengths, labels = zip(*batch)
    max_len = max(lengths)

    ids = torch.full((len(seqs), max_len), config.PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = seq          # copy in; the tail stays <pad>
    return ids, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def build_vocab_from_train(min_freq=None, max_size=None) -> Vocab:
    """Build the vocabulary from the TRAIN split only (see vocab.py).

    Deterministic: same tsv + same min_freq/max_size always yields the same
    itos, so a rebuild in eval.py matches the ids the checkpoint was trained
    with even if vocab.json went missing.
    """
    pairs = read_split("train")
    return Vocab.build((tokenize(s) for s, _ in pairs),
                       min_freq=config.MIN_FREQ if min_freq is None else min_freq,
                       max_size=config.MAX_VOCAB_SIZE if max_size is None else max_size)


def build_datasets(vocab: Vocab = None):
    """Build (train_set, dev_set, vocab) with a shared train-built vocabulary.

    Input:
        vocab: an existing Vocab (eval.py passes the one saved next to the
            checkpoint); None builds it from train.tsv.
    """
    vocab = vocab or build_vocab_from_train()
    return SST2Dataset("train", vocab), SST2Dataset("dev", vocab), vocab


# ---- Quick self-test / downloader: run this file directly --------------------
# python dataset/sst2.py --download
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect the SST-2 corpus")
    p.add_argument("--download", action="store_true", help="download SST-2 first")
    args = p.parse_args()

    if args.download or not sst2_present():
        download_sst2()

    train_set, dev_set, vocab = build_datasets()
    print(f"\nvocab size:      {len(vocab)}")
    print(f"train sentences: {len(train_set)}  labels: {train_set.label_counts()}")
    print(f"dev sentences:   {len(dev_set)}  labels: {dev_set.label_counts()}")
    print(f"dev <unk> rate:  {dev_set.unk_rate():.4f} "
          f"(train words unseen in dev's sentences)")

    lens = [len(i) for i in train_set.ids]
    lens.sort()
    print(f"train length: mean={sum(lens)/len(lens):.1f} "
          f"median={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")

    print("\nfirst 3 train samples:")
    for i in range(3):
        ids, n, y = train_set[i]
        print(f"  [{config.CLASS_NAMES[y]}] len={n} ids={ids.tolist()[:12]}...")
        print(f"      {train_set.sentences[i]!r}")

    # Collate check: 3 different lengths -> one [3, L_max] tensor.
    batch = collate_batch([train_set[0], train_set[1], train_set[2]])
    ids, lengths, labels = batch
    print(f"\ncollated ids {tuple(ids.shape)} lengths={lengths.tolist()} "
          f"labels={labels.tolist()}")
    print(f"padded tail of row 0: {ids[0, lengths[0]:].tolist()} (expected all 0)")

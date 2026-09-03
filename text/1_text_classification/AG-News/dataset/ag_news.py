"""AG News topic corpus: download, parsing, splitting, Dataset, collate_fn.

Responsibilities of this file (the counterpart of the SST-2 project's
dataset/sst2.py):
  1. Download the AG News csv files into config.DATA_ROOT (mirror fallback).
  2. Parse them into (text, label) pairs.
  3. Carve a stratified VALIDATION split out of train.
  4. Wrap a split in a Dataset that yields (ids LongTensor, length, label).
  5. Provide the COLLATE FUNCTION that pads variable-length sequences into one
     rectangular batch.

Point 3 is what this project adds over SST-2. AG News publishes only train and
test, and its test labels are PUBLIC -- so if per-epoch evaluation and
best-checkpoint selection ran on test, the reported number would be the best
of N draws on the very set it claims to generalize to. Instead 5% of train is
held out with a fixed, training-seed-independent SPLIT_SEED, so:

    train portion (114,000)  gradient updates + vocabulary
    val portion   (  6,000)  per-epoch curves, best.pt selection
    test.csv      (  7,600)  read once, at the end, by eval.py

The raw data format is a headerless 3-column csv:

    "3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-
    sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green
    again."
     ^     ^                                               ^
     |     title                                           description
     class index, 1-based: 1=World 2=Sports 3=Business 4=Sci/Tech

Two parsing details that matter and are easy to miss:
  * the fields are properly quoted csv with ""-escaped quotes, so the `csv`
    module is required -- a str.split(",") would shred half the corpus;
  * the text carries scraping damage -- literal backslashes where the article
    had a line break, and half-eaten HTML entities ("Arsenal #39;s") in a
    quarter of all rows. _clean() below repairs both, and explains why.

How to download: python dataset/ag_news.py --download
(train.py also downloads automatically when the corpus is missing.)
"""

import argparse
import csv
import html
import os
import random
import re
import sys

import torch
from torch.utils.data import Dataset

# Make the project root importable (works both as a script and as a package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402
from utils.download import download_with_mirrors, extract_tar  # noqa: E402

try:
    from .vocab import Vocab, tokenize
except ImportError:  # running this file directly
    from vocab import Vocab, tokenize

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
# Primary source: the fast.ai mirror of Zhang et al.'s original release, one
# 11 MB tgz holding ag_news_csv/{classes,train,test}.csv.
_AG_TGZ_URLS = [
    "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz",
]
# Fallback: the two csv files served raw from a GitHub mirror. Used when the
# S3 bucket is unreachable -- same content, no archive step.
_AG_CSV_URLS = {
    "train.csv": ["https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
                  "master/data/ag_news_csv/train.csv"],
    "test.csv": ["https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
                 "master/data/ag_news_csv/test.csv"],
}
# Expected row counts. Used INSTEAD of an md5: the mirrors need not be
# byte-identical (line endings differ), but a truncated or wrong file cannot
# possibly produce these counts.
_EXPECTED_ROWS = {"train.csv": 120000, "test.csv": 7600}


def ag_news_present() -> bool:
    """True when both csv files exist under config.AG_NEWS_DIR."""
    return all(os.path.isfile(os.path.join(config.AG_NEWS_DIR, f))
               for f in ("train.csv", "test.csv"))


def download_ag_news():
    """Fetch AG News into config.AG_NEWS_DIR (idempotent).

    Tries the tgz first and falls back to the raw csv mirror, because the two
    failure modes are different: S3 can be blocked by region while GitHub is
    reachable, and vice versa.
    """
    if ag_news_present():
        print(f"AG News already present at {config.AG_NEWS_DIR}")
        return
    os.makedirs(config.AG_NEWS_DIR, exist_ok=True)

    try:
        archive = os.path.join(config.DATA_ROOT, "ag_news_csv.tgz")
        print("Downloading AG News (~11 MB)...")
        # 4 connections, not 16: at 11 MB the per-connection handshake would
        # cost more than the parallelism saves.
        download_with_mirrors(_AG_TGZ_URLS, archive, md5=None, connections=4)
        extract_tar(archive, config.DATA_ROOT)
        # The tgz unpacks to ag_news_csv/ (train.csv, test.csv, classes.txt
        # and a readme). Move EVERYTHING to config.AG_NEWS_DIR -- listing only
        # the files we care about would leave the source folder non-empty and
        # the cleanup below would fail.
        unpacked = os.path.join(config.DATA_ROOT, "ag_news_csv")
        for name in os.listdir(unpacked):
            os.replace(os.path.join(unpacked, name),
                       os.path.join(config.AG_NEWS_DIR, name))
        os.rmdir(unpacked)
        os.remove(archive)
    except Exception as e:
        print(f"  tgz route failed ({type(e).__name__}: {e}) -- trying raw csv mirror")
        for name, urls in _AG_CSV_URLS.items():
            download_with_mirrors(urls, os.path.join(config.AG_NEWS_DIR, name),
                                  md5=None, connections=4)

    if not ag_news_present():
        raise RuntimeError(f"AG News csv files missing from {config.AG_NEWS_DIR}")


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
# Mangled HTML entities. The corpus was scraped from web pages and the escape
# sequences survived -- but only half of them: "&lt;" and "&gt;" kept their
# ampersand while "&#39;" and "&quot;" lost theirs, so html.unescape() alone
# repairs nothing. The optional "&?" is what handles both spellings.
_ENTITY_RE = re.compile(r"&?(#\d+|quot|amp|apos|nbsp|lt|gt);")
# HTML tags that surface once the entities above are unescaped: <b>, <strong>,
# </p>, <br /> ... The length bound stops the pattern from eating a stretch of
# ordinary prose that merely contains "<" and ">".
_TAG_RE = re.compile(r"<[a-z/][^>]{0,20}>")


def _clean(text: str) -> str:
    r"""Repair the two scraping artifacts baked into the published csv files.

    1. LINE BREAKS AS LITERAL BACKSLASHES (11% of documents). AG News writes
       "dwindling\band of ultra-cynics" where the source article had a
       newline. Without the replacement the tokenizer emits "dwindlingband",
       an out-of-GloVe token where an ordinary word belongs.

    2. BROKEN HTML ENTITIES (25% of documents carry a bare "#39;"). Left
       as-is, "Arsenal #39;s record" tokenizes to
       ["arsenal", "#", "39", ";", "s", "record"] -- four tokens of noise
       around what should be one apostrophe. Corpus-wide that is 47k "#",
       45k "39", 87k ";" and 10k "quot" tokens, about 4% of everything the RNN
       has to walk through. Repairing them costs nothing in GloVe coverage
       (98.97% -> 98.83% of tokens -- the junk tokens were themselves "found"
       in GloVe, which is exactly why coverage is a poor quality signal here)
       and hands the model real contractions and possessives instead.

    Deliberately NOT done: lowercasing, stopword removal, stemming. The first
    belongs to the tokenizer (vocab.py), and the other two throw away signal a
    recurrent model is supposed to be able to use.
    """
    text = text.replace("\\", " ")
    text = _ENTITY_RE.sub(lambda m: html.unescape("&" + m.group(1) + ";"), text)
    return _TAG_RE.sub(" ", text)


def read_csv(name: str):
    """Read one csv file into a list of (text, label) pairs.

    Input:
        name: "train.csv" or "test.csv".
    Output:
        list of (text str, label int in [0, 4)). Text is
        "title. description" -- the standard concatenation for this benchmark;
        the title alone already carries most of the topic signal, and the lead
        paragraph supplies the rest.
    """
    path = os.path.join(config.AG_NEWS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/ag_news.py --download`")

    out = []
    # newline="" is required by the csv module so quoted fields containing
    # newlines are handled by it rather than by python's line splitting.
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label, title, description = row[0], row[1], row[2]
            # The csv label column is 1-based; torch wants 0-based ids.
            out.append((f"{_clean(title)}. {_clean(description)}", int(label) - 1))

    expected = _EXPECTED_ROWS.get(name)
    if expected is not None and len(out) != expected:
        print(f"[ag_news] WARNING: {name} has {len(out)} rows, "
              f"expected {expected} -- corrupt or non-standard copy?")
    return out


def read_split(split: str):
    """Read one logical split: "train" / "val" / "test".

    "train" and "val" are the two halves of the stratified split of train.csv;
    "test" is test.csv verbatim.
    """
    if split == "test":
        return read_csv("test.csv")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    train_rows, val_rows = split_train_val(read_csv("train.csv"))
    return train_rows if split == "train" else val_rows


def split_train_val(rows, val_ratio: float = None, seed: int = None):
    """Split rows into (train, val), stratified by class and deterministic.

    Stratified means each class contributes the SAME FRACTION to val, so the
    validation set stays 25/25/25/25 like the corpus. With a plain random
    split that balance would hold only approximately -- fine at 6,000 rows,
    but stratifying costs three lines and removes the caveat entirely.

    Deterministic via config.SPLIT_SEED, which is kept separate from
    config.SEED on purpose: changing the training seed to check run-to-run
    variance must not also change which documents are being scored.

    Input:
        rows: list of (text, label) from read_csv("train.csv").
        val_ratio / seed: default to config.VAL_RATIO / config.SPLIT_SEED.
    Output:
        (train_rows, val_rows), both in shuffled order.
    """
    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
    seed = config.SPLIT_SEED if seed is None else seed

    by_class = {}
    for row in rows:
        by_class.setdefault(row[1], []).append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label in sorted(by_class):
        bucket = by_class[label]
        rng.shuffle(bucket)
        n_val = int(round(len(bucket) * val_ratio))
        val_rows.extend(bucket[:n_val])
        train_rows.extend(bucket[n_val:])

    # Shuffle across classes too: the buckets were concatenated in label
    # order, and a class-sorted "train" set would make any accidental
    # dependence on ordering (a forgotten shuffle=True) invisible instead of
    # catastrophic.
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class AGNewsDataset(Dataset):
    """One AG News split as (token ids, length, label) samples.

    The whole split is tokenized and encoded ONCE in __init__ (120k short
    documents -> a few seconds and a few hundred MB of python lists), so
    __getitem__ is a pure tensor construction and num_workers=0 costs nothing.

    Args:
        split: "train" / "val" / "test".
        vocab: the Vocab to encode with. Pass the TRAIN vocab for every split.
        max_len: truncation length (config.MAX_LEN).
    """

    def __init__(self, split: str, vocab: Vocab, max_len: int = None):
        self.split = split
        self.vocab = vocab
        self.max_len = max_len or config.MAX_LEN

        pairs = read_split(split)
        self.texts = [t for t, _ in pairs]
        self.labels = [y for _, y in pairs]
        self.ids = [vocab.encode(tokenize(t), self.max_len) for t in self.texts]
        # A document must never be empty: pack_padded_sequence rejects
        # length-0 sequences, and a row of pure whitespace would produce one.
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
        """Class histogram, e.g. {"World": 28500, "Sports": 28500, ...}."""
        out = {name: 0 for name in config.CLASS_NAMES}
        for y in self.labels:
            if 0 <= y < len(config.CLASS_NAMES):
                out[config.CLASS_NAMES[y]] += 1
        return out


def collate_batch(batch):
    """Pad a list of samples into one rectangular batch.

    The default collate can only stack tensors of identical shape, and
    documents have different lengths, so somebody must pad. We pad
    DYNAMICALLY -- to the longest document in THIS batch, not to
    config.MAX_LEN -- which keeps the RNN's unrolled length as short as the
    batch allows. The lengths travel with the batch so the encoder can pack
    them and the head can mask them; without that the model would read <pad>
    tokens as if they were words.

    Input:
        batch: list of (ids [L_i], length_i, label_i) from AGNewsDataset.
    Output:
        ids:     [B, L_max] long, padded with config.PAD_IDX (0)
        lengths: [B] long, the TRUE length of each row (before padding)
        labels:  [B] long
    """
    seqs, lengths, labels = zip(*batch)
    max_len = max(lengths)

    ids = torch.full((len(seqs), max_len), config.PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = seq          # copy in; the tail stays <pad>
    return (ids,
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def build_vocab_from_train(min_freq=None, max_size=None) -> Vocab:
    """Build the vocabulary from the TRAIN portion only (see vocab.py).

    Deterministic: same csv + same split seed + same min_freq/max_size always
    yields the same itos, so a rebuild in eval.py matches the ids the
    checkpoint was trained with even if vocab.json went missing.
    """
    rows = read_split("train")
    return Vocab.build((tokenize(t) for t, _ in rows),
                       min_freq=config.MIN_FREQ if min_freq is None else min_freq,
                       max_size=config.MAX_VOCAB_SIZE if max_size is None else max_size)


# ---- Quick self-test / downloader: run this file directly --------------------
# python dataset/ag_news.py --download
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect the AG News corpus")
    p.add_argument("--download", action="store_true", help="download AG News first")
    args = p.parse_args()

    if args.download or not ag_news_present():
        download_ag_news()

    vocab = build_vocab_from_train()
    print(f"\nvocab size: {len(vocab)} (min_freq={config.MIN_FREQ})")

    for split in ("train", "val", "test"):
        ds = AGNewsDataset(split, vocab)
        lens = sorted(len(i) for i in ds.ids)
        print(f"\n{split:5} docs={len(ds):6}  labels={ds.label_counts()}")
        print(f"      <unk> rate={ds.unk_rate():.4f}  "
              f"len mean={sum(lens)/len(lens):.1f} median={lens[len(lens)//2]} "
              f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")

    # No leakage: train and val must not share a single document.
    train_texts = set(AGNewsDataset("train", vocab).texts)
    val_texts = AGNewsDataset("val", vocab).texts
    overlap = sum(1 for t in val_texts if t in train_texts)
    print(f"\nval documents also present in train: {overlap} (expected 0)")

    ds = AGNewsDataset("train", vocab)
    print("\nfirst 2 train samples:")
    for i in range(2):
        ids, n, y = ds[i]
        print(f"  [{config.CLASS_NAMES[y]}] len={n} ids={ids.tolist()[:10]}...")
        print(f"      {ds.texts[i][:100]!r}")

    ids, lengths, labels = collate_batch([ds[0], ds[1], ds[2]])
    print(f"\ncollated ids {tuple(ids.shape)} lengths={lengths.tolist()} "
          f"labels={labels.tolist()}")
    print(f"padded tail of the shortest row: "
          f"{ids[int(lengths.argmin()), int(lengths.min()):].tolist()[:10]} (expected 0s)")

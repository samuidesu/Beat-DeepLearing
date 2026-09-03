"""Central configuration for RNN topic classification on AG News.

SECOND TEXT PROJECT of this repo. Everything structural is inherited from the
SST-2 sibling -- same embedding/encoder/head split, same two-stage layered-LR
finetune, same per-epoch JSON log + curve plots. What changes is the TASK, and
the four differences below are the whole point of running a second dataset:

    SST-2                            AG News
    ------------------------------   ---------------------------------------
    2 classes (sentiment)            4 classes (topic: World/Sports/Biz/Tech)
    67k fragments, ~10 tokens each   120k news items, ~46 tokens each
    44/56 class balance              exactly 25/25/25/25
    dev is the reported number       test IS public and labeled

The last row has a methodological consequence. SST-2's test labels are
withheld, so "pick the best epoch on dev" and "report dev" were the same thing
and no leakage was possible. AG News hands you a labeled test set -- which
means selecting checkpoints on it would be scoring your own homework. So this
project carves a VALIDATION split out of train (VAL_RATIO below, stratified +
seeded) for per-epoch monitoring and best-checkpoint selection, and touches
test.csv exactly once, at the end, for the number that goes in the README.

The recurrent cell is selectable (config.CELL or train.py --cell):

    "rnn"  -- vanilla Elman RNN: h_t = tanh(W x_t + U h_{t-1})
    "lstm" -- long short-term memory (input/forget/output gates + cell state)
    "gru"  -- gated recurrent unit (2 gates, no separate cell state)

Each cell writes to its OWN output folder, so the three experiments never
overwrite each other and the comparison is single-variable. Documents here are
~4x longer than SST-2's fragments, which is exactly the regime where the gated
cells are supposed to pull away from the Elman RNN -- see the README.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../AG-News).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Corpus + word-vector location. AG News is ~30 MB of csv and lives with this
# project; GloVe is 862 MB zipped, so it is worth REUSING across text projects:
# scan the sibling projects under text/1_text_classification/ first (the SST-2
# project already downloaded it) and only fall back to our own dataset/data/
# when no sibling copy exists.
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_SIBLING_ROOT = os.path.dirname(PROJECT_ROOT)          # .../1_text_classification


def _sibling_data_dirs():
    """Every <sibling project>/dataset/data directory next to this project."""
    if not os.path.isdir(_SIBLING_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(_SIBLING_ROOT)):
        cand = os.path.join(_SIBLING_ROOT, name, "dataset", "data")
        if os.path.isdir(cand) and os.path.normpath(cand) != os.path.normpath(_LOCAL_DATA):
            out.append(cand)
    return out


DATA_ROOT = _LOCAL_DATA

# Extracted corpus folder: <DATA_ROOT>/ag_news/{train,test}.csv
AG_NEWS_DIR = os.path.join(DATA_ROOT, "ag_news")

# Word-vector width, and the GloVe filename DERIVED from it -- the two must
# agree, so build the name instead of writing it twice (the SST-2 project keeps
# them as two independent constants, a wart worth not repeating).
EMBED_DIM = 100
GLOVE_NAME = f"glove.6B.{EMBED_DIM}d.txt"

# Where glove.6B.<dim>d.txt lives. First sibling copy wins (normally
# ../SST-2/dataset/data/glove/), else our own data dir -> download.
GLOVE_DIR = os.path.join(DATA_ROOT, "glove")
for _cand in _sibling_data_dirs():
    if os.path.isfile(os.path.join(_cand, "glove", GLOVE_NAME)):
        GLOVE_DIR = os.path.join(_cand, "glove")
        break
GLOVE_PATH = os.path.join(GLOVE_DIR, GLOVE_NAME)

# Logs, curves, checkpoints, vocab -- ONE FOLDER PER CELL.
OUTPUT_DIR_RNN = os.path.join(PROJECT_ROOT, "outputs_rnn")
OUTPUT_DIR_LSTM = os.path.join(PROJECT_ROOT, "outputs_lstm")
OUTPUT_DIR_GRU = os.path.join(PROJECT_ROOT, "outputs_gru")


def output_dir_for_cell(cell: str) -> str:
    """Map a cell name ("rnn" / "lstm" / "gru") to its output folder."""
    return {"rnn": OUTPUT_DIR_RNN,
            "lstm": OUTPUT_DIR_LSTM,
            "gru": OUTPUT_DIR_GRU}[cell]


# -----------------------------------------------------------------------------
# Dataset: AG News (Zhang, Zhao & LeCun 2015 topic-classification benchmark)
# -----------------------------------------------------------------------------
# News titles + lead paragraphs from the AG corpus, in four topics. The
# published split:
#     train.csv  120,000 rows (30,000 per class)
#     test.csv     7,600 rows (1,900 per class)
# The csv label column is 1-based (1..4); this project subtracts 1 so ids line
# up with CLASS_NAMES and with everything torch expects.
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(CLASS_NAMES)  # 4

# Validation split carved out of train.csv for per-epoch monitoring and
# best-checkpoint selection. Stratified (equal share of each class) and seeded,
# so the same 6,000 rows are held out on every run and the three cells are
# compared on identical data. test.csv is NOT touched during training.
VAL_RATIO = 0.05
SPLIT_SEED = 1234   # deliberately separate from SEED: the data split must not
                    # move when the training seed changes

# Special tokens. Ids are pinned: <pad>=0 so padding_idx=0 works everywhere,
# <unk>=1 for words missing from the vocabulary at inference time.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

# Vocabulary is built from the TRAIN portion only (never from val/test).
# min_freq=2 rather than SST-2's 1: this corpus has ~18x the tokens and its raw
# type count is dominated by hapax legomena -- typos, one-off proper nouns,
# stock tickers -- that cannot be learned from a single occurrence anyway.
# Dropping them roughly halves the embedding table and, just as usefully, sends
# real training examples through <unk> so that vector gets trained instead of
# sitting at its random init.
MIN_FREQ = 2
MAX_VOCAB_SIZE = None  # None = no cap

# Truncate long documents to this many tokens. Measured on the cleaned train
# portion: mean 45.6, median 44, p99 97, max 253. 128 therefore truncates only
# 0.36% of documents while capping the RNN's unrolled depth (and with it the
# worst-case BPTT chain) on the long tail. Compare SST-2, where MAX_LEN=64 was
# pure insurance and never fired -- here it genuinely bites, just rarely.
MAX_LEN = 128

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Recurrent cell: "rnn" (vanilla Elman) / "lstm" / "gru". train.py --cell wins.
CELL = "lstm"

# Hidden width PER DIRECTION. Bidirectional doubles the feature the head sees.
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True

# Dropout on the embeddings, BETWEEN stacked RNN layers, and before the
# classifier. 0.5 is SST-2's value kept unchanged -- worth watching, because
# 114k documents overfit far less readily than 67k fragments and this may well
# be more regularization than the task needs. The curves will say.
DROPOUT = 0.5

# How the variable-length token features become ONE document vector:
#     "last" -- final hidden state (both directions concatenated)
#     "max"  -- element-wise max over time (masked)
#     "mean" -- masked average over time
# "last" stays the default so the cell comparison runs under the setting that
# stresses the recurrence hardest: with 43-token documents, everything the
# classifier sees has to have survived the whole walk.
POOLING = "last"

# GloVe pretrained embeddings = this project's "ImageNet backbone". Set False
# (or pass --no-glove) to train the word vectors from scratch.
USE_GLOVE = True

# -----------------------------------------------------------------------------
# Training -- same two-stage layered-LR protocol as SST-2
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"

# 128 rather than SST-2's 64: with 114k training documents that are 4x longer,
# an epoch is roughly 10x the work, and the larger batch buys back a good part
# of that wall clock on a task this easy.
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

# 0 = load in the main process. The whole corpus is tokenized into python lists
# in __init__, so a worker would only pay Windows' process-spawn cost.
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4

# Gradient-norm clipping. Not optional for RNNs: backprop through time
# multiplies by the same recurrent Jacobian at every step, and these sequences
# are ~4x longer than SST-2's, so the exponent in that product grows with them.
GRAD_CLIP = 5.0

# Label smoothing (0 disables): softens the CE target, a mild regularizer.
LABEL_SMOOTHING = 0.05

# Stage 1: FREEZE the embedding table; the from-scratch encoder + head learn
#          to read fixed GloVe vectors.
# Stage 2: unfreeze everything, three LR tiers (embedding slowest, head
#          fastest) so the pretrained vectors drift instead of being wrecked.
# Fewer epochs than SST-2 needed: AG News is an easier task with far more data
# per class, and the accuracy curve flattens within a handful of passes.
STAGE1_EPOCHS = 5
STAGE1_LR_HEAD = 1e-3
STAGE1_LR_ENCODER = 1e-3        # also from scratch -> same tier as the head

STAGE2_EPOCHS = 3
STAGE2_LR_HEAD = 3e-4
STAGE2_LR_ENCODER = 3e-4
STAGE2_LR_EMBEDDING = 5e-5      # pretrained words: gentle updates only

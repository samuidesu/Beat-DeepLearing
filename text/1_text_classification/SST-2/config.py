"""Central configuration for RNN text classification on SST-2 (GLUE).

FIRST TEXT PROJECT of this repo. The task changes from "label every pixel" to
"label every SENTENCE", so the whole data pipeline changes shape:

    images : fixed-size float tensors, augmented with crops/flips
    text   : VARIABLE-LENGTH id sequences, "preprocessing" is a vocab lookup

Nothing about the training frame changes though -- the same two-stage
layered-LR finetune, the same per-epoch JSON log + curve plots, the same
best-checkpoint-on-metric protocol as the segmentation projects. What plays
the role of the ImageNet-pretrained backbone here is the GloVe embedding
table: a big matrix of general-purpose word vectors that we first FREEZE
(stage 1: the RNN learns to read them) and then unfreeze at a small LR
(stage 2: the words themselves adapt to sentiment).

Model = embedding (backbone) -> RNN encoder (neck) -> pooling + linear (head).
The recurrent cell is selectable (config.CELL or train.py --cell):

    "rnn"  -- vanilla Elman RNN: h_t = tanh(W x_t + U h_{t-1})
    "lstm" -- long short-term memory (input/forget/output gates + cell state)
    "gru"  -- gated recurrent unit (2 gates, no separate cell state)

Each cell writes to its OWN output folder (outputs_rnn / outputs_lstm /
outputs_gru), so the three experiments never overwrite each other and the
comparison is single-variable -- same data, same vocab, same schedule.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../SST-2).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Corpus + word-vector location. SST-2 itself is tiny (~7 MB), but GloVe is
# 862 MB zipped, so it is worth REUSING across text projects: scan the sibling
# projects under text/1_text_classification/ first and only fall back to this
# project's own dataset/data/ (gitignored) when no sibling copy exists.
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


# SST-2 lives with this project (it IS this project's dataset); only the shared
# GloVe vectors are looked up across siblings.
DATA_ROOT = _LOCAL_DATA

# Extracted GLUE SST-2 folder: <DATA_ROOT>/SST-2/{train,dev,test}.tsv
SST2_DIR = os.path.join(DATA_ROOT, "SST-2")

# Where glove.6B.*.txt lives. First sibling copy wins, else our own data dir.
GLOVE_NAME = "glove.6B.100d.txt"          # must agree with EMBED_DIM below
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
# Dataset: SST-2 (Stanford Sentiment Treebank, 2-class GLUE version)
# -----------------------------------------------------------------------------
# Movie-review fragments labeled negative (0) / positive (1). The GLUE split:
#     train.tsv  67,349 labeled sentences (phrases, many are fragments)
#     dev.tsv       872 labeled sentences (full sentences)
#     test.tsv    1,821 UNLABELED sentences (the GLUE server scores them)
# Because test labels are withheld, "test accuracy" in every SST-2 paper and
# in this project means DEV accuracy -- that is the number to compare.
CLASS_NAMES = ["negative", "positive"]
NUM_CLASSES = len(CLASS_NAMES)  # 2

# Special tokens. Ids are pinned: <pad>=0 so padding_idx=0 works everywhere,
# <unk>=1 for words missing from the vocabulary at inference time.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

# Vocabulary built from the TRAIN split only (dev/test words that never appear
# in train map to <unk> -- building it over dev would leak evaluation data).
# min_freq=1 keeps all ~14.8k train types: the corpus is small, and with GloVe
# even the rare ones start from a real pretrained vector rather than noise.
MIN_FREQ = 1
MAX_VOCAB_SIZE = None  # None = no cap

# Truncate long sentences to this many tokens. SST-2's longest train sentence
# is ~52 tokens, so 64 truncates nothing here -- it is a safety net for the
# free-form text passed to predict.py, and it caps the RNN's unrolled depth.
MAX_LEN = 64

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Recurrent cell: "rnn" (vanilla Elman) / "lstm" / "gru". train.py --cell wins.
CELL = "lstm"

# Word-vector width. MUST match the GloVe file (glove.6B.100d.txt -> 100).
EMBED_DIM = 100

# Hidden width PER DIRECTION. Bidirectional doubles the feature the head sees.
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True

# Dropout on the embeddings, BETWEEN stacked RNN layers, and before the
# classifier. RNNs on a 67k-sentence corpus overfit fast; 0.5 is the classic
# text-classification value.
DROPOUT = 0.5

# How the variable-length token features become ONE sentence vector:
#     "last" -- final hidden state (both directions concatenated), the
#               textbook RNN classifier;
#     "max"  -- element-wise max over time (masked), usually a bit stronger;
#     "mean" -- masked average over time.
POOLING = "last"

# GloVe pretrained embeddings = this project's "ImageNet backbone". Set False
# (or pass --no-glove) to train word vectors from scratch; train.py then warns
# that stage 1's freeze is pointless and suggests --epochs-stage1 0.
USE_GLOVE = True

# -----------------------------------------------------------------------------
# Training -- same two-stage layered-LR protocol as the segmentation projects
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 64
# 0 = load in the main process. The whole corpus is already tokenized into
# python lists in RAM, so a worker would only pay Windows' process-spawn cost.
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4

# Gradient-norm clipping -- NEW compared to the CNN projects, and not optional
# here: backprop through time multiplies by the same recurrent Jacobian at
# every step, so gradients can explode. Clipping the global norm to 5 is the
# standard RNN safety belt.
GRAD_CLIP = 5.0

# Label smoothing (0 disables): softens the CE target, a mild regularizer that
# also stops the model from becoming absurdly overconfident on 2 classes.
LABEL_SMOOTHING = 0.05

# Stage 1: FREEZE the embedding table; the from-scratch encoder + head learn
#          to read fixed GloVe vectors.
# Stage 2: unfreeze everything, three LR tiers (embedding slowest, head
#          fastest) so the pretrained vectors drift instead of being wrecked.
STAGE1_EPOCHS = 8
STAGE1_LR_HEAD = 1e-3
STAGE1_LR_ENCODER = 1e-3        # also from scratch -> same tier as the head

STAGE2_EPOCHS = 4
STAGE2_LR_HEAD = 3e-4
STAGE2_LR_ENCODER = 3e-4
STAGE2_LR_EMBEDDING = 5e-5      # pretrained words: gentle updates only

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Dev is 872 sentences -- a full pass costs well under a second, so (unlike
# the segmentation projects' EVAL_MAX_BATCHES proxy) every epoch evaluates the
# WHOLE dev set and the per-epoch number IS the real number.
EVAL_BATCH_SIZE = 128

"""Central configuration for HRNetV2-W32 semantic segmentation on PASCAL VOC 2012.

HRNet's answer to the resolution problem, contrasted with this repo's earlier
segmenters:

    FCN:     downsample to stride 32, FUSE taps back up (FPN)   "lose, rebuild"
    DeepLab: never drop below stride 8 (dilation)               "never lose"
    HRNet:   keep a stride-4 branch ALIVE the whole way, and    "never lose,
             run parallel lower-resolution branches next to      and talk"
             it, exchanging information repeatedly.

The backbone maintains 4 parallel branches (strides 4/8/16/32, widths
32/64/128/256 for W32) with repeated cross-resolution fusion. The head is
selectable (config.HEAD or train.py --head):

    "simple" -- HRNetV2 head: upsample all 4 branches to stride 4, concat
                (480 ch), 1x1 mix, classify.
    "ocr"    -- OCR head (object-contextual representations): a soft-region
                aux prediction gathers one feature vector per CLASS, then each
                pixel attends over those class vectors to refine its feature
                before classification. Trains with an auxiliary loss.

Each head writes to its OWN output folder (outputs_simple/ vs outputs_ocr/),
so the two experiments never overwrite each other.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../HRNet).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# VOC data location. The dataset is IDENTICAL to the DeepLab/FCN projects, so
# reuse the first sibling copy that has VOC2012 (they also hold SBD next to it
# in <root>/sbd, which config.DATA_ROOT-relative code picks up automatically).
# Only if no sibling copy exists is data downloaded into this project's own
# dataset/data/ (gitignored).
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_CANDIDATES = [
    os.path.normpath(os.path.join(PROJECT_ROOT, "..", proj, "dataset", "data"))
    for proj in ("Deeplab", "DeeplabV3_plus", "FCN", "FCN_concat")
] + [
    os.path.normpath(os.path.join(
        PROJECT_ROOT, "..", "..", "[2]ObjectionDetection", proj,
        "PASCAL_VOC", "dataset", "data"))
    for proj in ("YOLO3", "FCOS")
]
DATA_ROOT = _LOCAL_DATA
for _cand in _CANDIDATES + [_LOCAL_DATA]:
    if os.path.isdir(os.path.join(_cand, "VOCdevkit", "VOC2012")):
        DATA_ROOT = _cand
        break

# Logs, curves, checkpoints. ONE FOLDER PER HEAD so a simple-head run and an
# OCR run sit side by side instead of overwriting each other's best.pt.
OUTPUT_DIR_SIMPLE = os.path.join(PROJECT_ROOT, "outputs_simple")
OUTPUT_DIR_OCR = os.path.join(PROJECT_ROOT, "outputs_ocr")


def output_dir_for_head(head: str) -> str:
    """Map a head name ("simple" / "ocr") to its output folder."""
    return {"simple": OUTPUT_DIR_SIMPLE, "ocr": OUTPUT_DIR_OCR}[head]


# -----------------------------------------------------------------------------
# Dataset: PASCAL VOC 2012 segmentation (21 classes) -- identical to DeepLab
# -----------------------------------------------------------------------------
VOC_SEG_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NUM_CLASSES = len(VOC_SEG_CLASSES)  # 21

# Pixel value marking "ignore" in the label pngs (VOC void contours + our mask
# padding): excluded from both the loss and the mIoU metric.
IGNORE_INDEX = 255

# SBD ("VOC aug") extra training data: train grows 1464 -> ~10.5k images.
# The sibling projects already downloaded it (DATA_ROOT/sbd), so this is free.
USE_SBD = True

# -----------------------------------------------------------------------------
# Image preprocessing -- identical protocol to the DeepLab experiments
# -----------------------------------------------------------------------------
# Random CROP_SIZE x CROP_SIZE crops (after random rescaling). Must be a
# multiple of 32: unlike DeepLab (whose dilated backbone stopped at stride 8),
# HRNet's lowest-resolution branch runs at stride 32, so every input dimension
# must divide by 32 (480 / 32 = 15).
CROP_SIZE = 480

# Random rescale factor range applied BEFORE cropping (DeepLab-standard).
SCALE_RANGE = (0.5, 2.0)

# Eval protocol: no resize; pad (right/bottom) to the next multiple of
# SIZE_DIVISOR, mask pad = IGNORE_INDEX. 32, NOT DeepLab's 8: the stride-32
# branch needs it (see CROP_SIZE note).
SIZE_DIVISOR = 32

# ImageNet normalization stats (the pretrained backbone expects these).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Which head to train: "simple" (HRNetV2 concat head) or "ocr" (OCR head with
# object-contextual attention + auxiliary loss). train.py --head overrides.
HEAD = "simple"

# HRNetV2-W32 branch widths at strides 4/8/16/32. "W32" = the width of the
# highest-resolution branch; each lower-resolution branch doubles it.
HRNET_CHANNELS = (32, 64, 128, 256)

# Blocks per branch inside every HighResolutionModule (official HRNet: 4).
HRNET_NUM_BLOCKS = 4

# Modules (fusion rounds) per stage: stage2 x1, stage3 x4, stage4 x3
# (the official HRNetV2 configuration).
HRNET_NUM_MODULES = (1, 4, 3)

# ImageNet-pretrained backbone weights. torchvision has no HRNet, so the
# official checkpoint must be fetched once by hand:
#   hrnetv2_w32_imagenet_pretrained.pth  (~110 MB)
# from the HRNet-Image-Classification releases (github.com/HRNet). Put it at
# the path below. If the file is missing the backbone trains FROM SCRATCH --
# train.py prints a loud warning, and stage 1's low-layer freeze should then
# be skipped (--epochs-stage1 0): freezing RANDOM weights helps nobody.
PRETRAINED_BACKBONE = os.path.join(
    PROJECT_ROOT, "model", "hrnetv2_w32_imagenet_pretrained.pth")

# --- OCR head knobs (only read when HEAD == "ocr") ---
OCR_MID_CHANNELS = 512   # pixel-feature width after the 3x3 entry conv
OCR_KEY_CHANNELS = 256   # query/key width inside the object attention
AUX_LOSS_WEIGHT = 0.4    # weight of the soft-region auxiliary CE loss

# Dropout inside the heads (0 disables).
HEAD_DROPOUT = 0.05

# -----------------------------------------------------------------------------
# Training -- same two-stage layered-LR protocol as the DeepLab experiments
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16  # of 480x480 crops; HRNet's stride-4 branch is memory-hungry,
                 # halve this if you hit OOM
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-3

# Stage 1: freeze the LOW backbone (stem + layer1); the parallel-branch stages
#          (whose fusion convs are new anyway) train with the head.
# Stage 2: unfreeze everything, three LR tiers (low slowest .. head fastest).
STAGE1_EPOCHS = 20
STAGE1_LR_HEAD = 1e-3           # heads + (new) transition/fusion-heavy stages
STAGE1_LR_BACKBONE_HIGH = 1e-4  # stage2/3/4 branches (pretrained, adapting)

STAGE2_EPOCHS = 60
STAGE2_LR_HEAD = 1e-4
STAGE2_LR_BACKBONE_HIGH = 3e-5
STAGE2_LR_BACKBONE_LOW = 1e-5   # stem + layer1

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Per-epoch val proxy cap (images, since the val loader is batch_size=1);
# None = all 1449. The final best.pt mIoU is always computed in full.
EVAL_MAX_BATCHES = 300

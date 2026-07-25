"""Central configuration for FCN semantic segmentation on PASCAL VOC 2012.

Semantic segmentation = classify EVERY PIXEL into one of NUM_CLASSES classes.
This project deliberately reuses the FCOS detector's backbone + FPN neck and
swaps only the task: the pyramid is fused into ONE stride-8 map, a small head
predicts 21 class logits per location, and the logits are upsampled 8x back to
input resolution -- the modern equivalent of the original FCN-8s (merge
stride-32/16/8 information, predict, upsample).

Notice everything that is GONE compared to the detection configs: no anchors,
no regression ranges, no center sampling, no NMS / confidence thresholds.
Dense per-pixel classification needs none of that machinery -- the knobs left
are the crop size, the scale-jitter range, and the same two-stage finetune
schedule that drove the YOLO3/FCOS results.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../FCN).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# VOC data location. The VOC2012 trainval archive ALREADY CONTAINS the
# segmentation labels (SegmentationClass/ pngs + ImageSets/Segmentation/ split
# lists), so the ~2 GB copy the detection projects downloaded serves this
# project as-is. Reuse the first candidate that has VOC2012; otherwise data is
# downloaded into this project's own dataset/data/ (gitignored).
_LOCAL_DATA = os.path.join(PROJECT_ROOT, "dataset", "data")
_YOLO3_DATA = os.path.normpath(
    os.path.join(
        PROJECT_ROOT, "..", "..", "[2]ObjectionDetection", "YOLO3", "PASCAL_VOC", "dataset", "data"
    )
)
_FCOS_DATA = os.path.normpath(
    os.path.join(
        PROJECT_ROOT, "..", "..", "[2]ObjectionDetection", "FCOS", "PASCAL_VOC", "dataset", "data"
    )
)
DATA_ROOT = _LOCAL_DATA
for _cand in (_YOLO3_DATA, _FCOS_DATA, _LOCAL_DATA):
    if os.path.isdir(os.path.join(_cand, "VOCdevkit", "VOC2012")):
        DATA_ROOT = _cand
        break

# Logs, curves, checkpoints go here.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# -----------------------------------------------------------------------------
# Dataset: PASCAL VOC 2012 segmentation (21 classes)
# -----------------------------------------------------------------------------
# Index 0 is BACKGROUND -- segmentation must give every pixel a label, so "none
# of the 20 objects" is itself a class (detection never needed this: there,
# background = simply predicting nothing). Indices 1..20 are the VOC classes in
# their official order; the label pngs store these exact ids per pixel.
# Do NOT reorder.
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

# Pixel value in the label pngs marking "ignore": the thin white contour VOC
# draws around every object (plus some ambiguous regions). These pixels are
# excluded from BOTH the loss (CrossEntropyLoss ignore_index) and the mIoU
# metric. We also reuse 255 as the mask PAD value, so padded borders are
# transparently ignored through the exact same mechanism.
IGNORE_INDEX = 255

# Optional extra training data: SBD ("VOC aug") adds ~9k VOC images that only
# have segmentation labels in the SBD release (train grows 1464 -> ~10.5k
# images) and typically lifts mIoU by several points. Off by default: it is a
# separate ~1.4 GB download (mirror is flaky) and reading its .mat masks needs
# scipy. See dataset/voc.py (SBDSegDataset) and the README.
USE_SBD = True

# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
# Training samples are random CROP_SIZE x CROP_SIZE crops (after random
# rescaling). Must be a multiple of 32 so the stride-32 grid is integer
# (480 / 32 = 15). 480 ~ VOC's max image side (500), the standard seg choice.
# Unlike detection we do NOT squash whole images to a square: the label is
# per-pixel, so cropping loses only context -- never label precision -- while
# resizing to a square would distort every object.
CROP_SIZE = 480

# Random rescale factor range applied BEFORE cropping: the segmentation
# counterpart of detection's RandomAffine scale jitter. (0.5, 2.0) is the
# DeepLab-standard range -- it forces the model to recognize every class at
# very different apparent sizes.
SCALE_RANGE = (0.5, 2.0)

# Eval protocol: images are NOT resized. They are padded (right/bottom) to the
# next multiple of SIZE_DIVISOR so every stride divides cleanly; the mask pad
# value is IGNORE_INDEX so padded pixels never count. mIoU is thus measured at
# the ORIGINAL resolution -- the official VOC protocol.
SIZE_DIVISOR = 32

# ImageNet normalization stats. The ResNet backbone was pretrained with these,
# so inputs must be normalized the same way (identical to the detection projects).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone architecture: "resnet18" or "resnet34" (same options as the
# detection projects; both tap C3/C4/C5 at channels 128/256/512).
# resnet34 = the best FCOS run's choice, kept for comparability.
BACKBONE = "resnet34"

# Width of the neck's single fused stride-8 output map (and the head input).
FPN_CHANNELS = 256

# -----------------------------------------------------------------------------
# Training (identical two-stage protocol to the YOLO3 / FCOS projects)
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16  # of 480x480 crops; halve it if you hit OOM
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-3  # same as the detection projects

# Two-stage finetuning schedule:
#   Stage 1: freeze the whole backbone, train only neck + head (warm them up so
#            they don't push garbage gradients into the pretrained weights).
#   Stage 2: unfreeze the backbone and finetune end-to-end with a smaller LR.
# The YOLO3/FCOS experiments both showed the frozen ImageNet backbone is the
# accuracy bottleneck -- stage 2 is where the score jumps; expect the same here.
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 60
STAGE2_LR_HEAD = 1e-4  # lr for neck + head in stage 2
STAGE2_LR_BACKBONE = 3e-5  # lr for the unfrozen backbone (the key knob)
# "all" unfreezes the ENTIRE backbone; a tuple like ("layer3", "layer4")
# unfreezes only those stages.
STAGE2_UNFREEZE = "all"

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Per-epoch monitoring cost control. The val loader runs batch_size=1 (each
# image keeps its own size), so this caps how many val IMAGES the per-epoch
# loss/mIoU proxy sees (None = all 1449, adds a couple of minutes per epoch).
# It is a biased-but-consistent proxy, same idea as FCOS's
# MAP_EVAL_MAX_BATCHES; the final best.pt mIoU is always computed in full.
EVAL_MAX_BATCHES = 300

"""Central configuration for DeepLab-v1 semantic segmentation on PASCAL VOC 2012.

Semantic segmentation = classify EVERY PIXEL into one of NUM_CLASSES classes.
This project is the DeepLab counterpart of the FCN project: instead of letting
the backbone downsample to stride 32 and rebuilding resolution with an FPN,
the strides in ResNet layer3/layer4 are REPLACED BY DILATION so the backbone
never drops below stride 8. A LargeFOV atrous context layer (the neck) and a
1x1 classifier + 8x bilinear upsample (the head) finish the job. Same data,
same loss, same metric as FCN -- the experiment isolates the backbone-
resolution question.

The two-stage finetune schedule DIFFERS from FCN's:
    Stage 1: freeze only the LOW backbone (stem/layer1/layer2); the dilation-
             modified layer3/layer4 train together with the new neck/head.
    Stage 2: unfreeze everything; LAYERED learning rates (low backbone
             slowest, high backbone middle, neck/head fastest).
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Absolute path to this project folder (.../Deeplab).
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
# rescaling). Must be a multiple of 8 so the stride-8 grid is integer -- the
# dilated backbone never goes below stride 8, so /32 divisibility (FCN's
# constraint) is no longer required. 480 ~ VOC's max image side (500).
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
# 8, not FCN's 32: the deepest stride in this network IS 8, so padding any
# further would only waste compute on ignored pixels.
SIZE_DIVISOR = 8

# ImageNet normalization stats. The ResNet backbone was pretrained with these,
# so inputs must be normalized the same way (identical to the detection projects).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone architecture: "resnet18" or "resnet34"; layer3/layer4 get their
# strides replaced by dilation (output stride 8) inside model/backbone.py.
# resnet34 = the FCN experiment's choice, kept for comparability.
BACKBONE = "resnet34"

# Neck (LargeFOV atrous context layer): 512 -> hidden -> out channels.
NECK_HIDDEN_CHANNELS = 256   # width of the rate-12 atrous 3x3 conv output
NECK_OUT_CHANNELS = 128      # width handed to the head's 1x1 classifier
ATROUS_RATE = 12             # dilation of the LargeFOV conv (DeepLab-v1 value)
NECK_DROPOUT = 0.1           # spatial dropout inside the neck; 0 disables

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"  # "auto" -> cuda > mps > cpu ; or force "cuda"/"cpu"/"mps"
BATCH_SIZE = 16  # of 480x480 crops; halve it if you hit OOM
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-3  # same as the detection projects

# Two-stage finetuning schedule (see the module docstring):
#   Stage 1: freeze stem/layer1/layer2 only. Two LR tiers -- the pretrained-
#            but-geometry-modified layer3/layer4 move slower than the
#            from-scratch neck/head.
#   Stage 2: unfreeze the whole backbone. Three LR tiers -- the deeper into
#            pristine pretrained territory, the smaller the LR.
STAGE1_EPOCHS = 20
STAGE1_LR_HEAD = 1e-3           # neck + head (training from scratch)
STAGE1_LR_BACKBONE_HIGH = 1e-4  # layer3 + layer4 (pretrained, adapting)

STAGE2_EPOCHS = 60
STAGE2_LR_HEAD = 1e-4           # neck + head
STAGE2_LR_BACKBONE_HIGH = 3e-5  # layer3 + layer4
STAGE2_LR_BACKBONE_LOW = 1e-5   # stem + layer1 + layer2 (barely nudge)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Per-epoch monitoring cost control. The val loader runs batch_size=1 (each
# image keeps its own size), so this caps how many val IMAGES the per-epoch
# loss/mIoU proxy sees (None = all 1449, adds a couple of minutes per epoch).
# It is a biased-but-consistent proxy, same idea as FCOS's
# MAP_EVAL_MAX_BATCHES; the final best.pt mIoU is always computed in full.
EVAL_MAX_BATCHES = 300

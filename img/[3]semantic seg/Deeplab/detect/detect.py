"""Random spot-check: predicted vs. ground-truth masks, side by side on disk.

Each run samples N (default 10) RANDOM images from the VOC2012 seg val split
(VOC's real "test" labels were never released -- val is the held-out set) and
writes TWO palette-colorized pngs per image into detect/results/:

    <id>_pred.png -- the model's predicted mask (VOC palette colors)
    <id>_gt.png   -- the ground-truth mask     (same palette)

The two files sort next to each other, so flipping between them in any image
viewer shows exactly where the prediction deviates. No seed by default: every
run draws a FRESH random sample (pass --seed to reproduce one).

This is the segmentation counterpart of the detection projects' detect.py.
The heavy lifting is all reused:
    sample_voc_val / load_voc_gt / segment_one  <- segment/segment.py
    colorize_mask (official VOC palette)        <- utils/viz.py

Usage (from the Deeplab project root):
    python detect/detect.py                # 10 random val images
    python detect/detect.py --n 5 --seed 0 # reproducible 5-image sample
"""

import os
import sys
import shutil
import argparse

import torch

# This file sits in Deeplab/detect/, so the project root is its parent's parent.
# Put it on sys.path so `import config`, `model`, `segment` ... resolve
# regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from model.deeplab import DeepLab  # noqa: E402
from dataset.transforms import get_eval_transforms  # noqa: E402
from utils.viz import colorize_mask  # noqa: E402
from train import get_device  # noqa: E402
from segment.segment import sample_voc_val, load_voc_gt, segment_one  # noqa: E402

# Output folder: Deeplab/detect/results/ (next to this file). A subfolder -- NOT
# detect/ itself -- because main() wipes it fresh each run, and wiping the
# folder this script lives in would delete the script.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    p = argparse.ArgumentParser(
        description="Save pred/GT mask pairs for N random VOC val images")
    p.add_argument("--n", type=int, default=10,
                   help="how many random val images to sample (default 10)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; default None = new random sample every run")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    # Fresh output folder each run so old samples never mix with new ones.
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    print(f"Device: {device}")

    # Build the model skeleton (pretrained=False: no ImageNet download needed,
    # every weight is about to be overwritten by the checkpoint) + load weights.
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    neck_hidden_channels=config.NECK_HIDDEN_CHANNELS,
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    atrous_rate=config.ATROUS_RATE,
                    neck_dropout=config.NECK_DROPOUT).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights: {args.weights}")

    # Eval preprocessing: pad to /8 + normalize, NO resize (predictions come
    # back at original resolution -- segment_one slices the padding off).
    transform = get_eval_transforms(config.IMAGENET_MEAN, config.IMAGENET_STD,
                                    config.SIZE_DIVISOR, config.IGNORE_INDEX)

    for path in sample_voc_val(args.n, args.seed):
        stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "2007_000033"

        # ---- Prediction: forward + argmax -> [H, W] class ids -> palette png.
        _, pred = segment_one(model, transform, device, path)
        colorize_mask(pred).save(os.path.join(args.out, f"{stem}_pred.png"))

        # ---- Ground truth: raw ids from SegmentationClass/<id>.png -> same
        # palette, so pred/gt colors are directly comparable.
        gt = load_voc_gt(path)  # never None here: val ids all have masks
        colorize_mask(gt).save(os.path.join(args.out, f"{stem}_gt.png"))

        # Per-image console summary: which classes each mask contains
        # (background dropped; 255 = void contours dropped from GT).
        pred_ids = set(int(c) for c in torch.unique(pred)) - {0}
        gt_ids = set(int(c) for c in set(gt.flatten().tolist())) - {0, 255}
        fmt = lambda ids: ", ".join(config.VOC_SEG_CLASSES[i] for i in sorted(ids)) or "-"
        print(f"{stem}:  pred=[{fmt(pred_ids)}]  gt=[{fmt(gt_ids)}]")

    print(f"\nSaved {args.n} pred/gt pairs to: {args.out}")


if __name__ == "__main__":
    main()

"""Inference entry point: run a trained DeepLab on image(s) and save colorized
segmentation masks (the counterpart of the detection projects' detect.py --
paint pixels instead of drawing boxes).

This script lives in Deeplab/segment/. Results are written to
Deeplab/segment/results/, which is wiped and recreated fresh on every run.

For each input image we save up to three files, named so they sort together:
    <id>_overlay.jpg -- prediction painted over the photo (the one to look at)
    <id>_pred.png    -- the raw colorized prediction mask (VOC palette)
    <id>_gt.png      -- the colorized ground-truth mask (only for VOC images
                        that have one in SegmentationClass/)

Usage (run from the Deeplab project root):
    python segment/segment.py --img path/to/image.jpg
    python segment/segment.py --voc-random 10        # 10 random VOC2012 val images
    python segment/segment.py --dir path/to/folder --alpha 0.7
"""

import os
import sys
import glob
import random
import shutil
import argparse

import numpy as np
import torch
from PIL import Image

# This file sits in Deeplab/segment/, so the project root is its parent's parent.
# Put it on sys.path so `import config`, `model`, `utils`, `train` ... resolve
# regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from model.deeplab import DeepLab  # noqa: E402
from dataset.transforms import get_eval_transforms  # noqa: E402
from utils.viz import colorize_mask, overlay_mask  # noqa: E402
from train import get_device  # noqa: E402  reuse the device picker

# Image extensions we accept when given a folder.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# Default output folder: Deeplab/segment/results/ (sits next to this file).
_SEGMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SEGMENT_DIR, "results")


def parse_args():
    p = argparse.ArgumentParser(description="Run DeepLab segmentation on images")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="path to a single image")
    src.add_argument("--dir", help="path to a folder of images")
    src.add_argument("--voc-random", type=int, metavar="N",
                     help="randomly sample N images from the VOC2012 seg val split")
    p.add_argument("--seed", type=int, default=None,
                   help="optional random seed for reproducible --voc-random sampling")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt")
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--alpha", type=float, default=0.55,
                   help="overlay tint strength for object pixels (0..1)")
    return p.parse_args()


def sample_voc_val(n, seed):
    """Randomly pick `n` image paths from the VOC2012 segmentation val split.

    Reads the official id list at
        <DATA_ROOT>/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
    (NOTE: Segmentation/, not detection's Main/ -- only ~2.9k of VOC2012's
    images carry segmentation masks) and maps ids to JPEGImages/<id>.jpg.

    Input:
        n: how many images to sample (capped at the split size, 1449).
        seed: optional RNG seed. None gives a different sample each run.
    Output:
        sorted list of image file paths (length <= n).
    """
    voc12 = os.path.join(config.DATA_ROOT, "VOCdevkit", "VOC2012")
    split_file = os.path.join(voc12, "ImageSets", "Segmentation", "val.txt")
    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    # A supplied seed makes sampling reproducible; a local RNG keeps the
    # global random state untouched.
    random.Random(seed).shuffle(ids)
    ids = ids[:n]
    img_dir = os.path.join(voc12, "JPEGImages")
    return sorted(os.path.join(img_dir, f"{i}.jpg") for i in ids)


def load_voc_gt(image_path):
    """Load the GT segmentation mask for a VOC image, if it exists.

    Maps the image path to its label by swapping folder and extension:
        .../VOC2012/JPEGImages/2007_000033.jpg
     -> .../VOC2012/SegmentationClass/2007_000033.png

    Output:
        [H, W] uint8 ndarray of class ids (0..20, 255), or None if the image
        has no segmentation annotation (most VOC images don't!).
    """
    gt_path = image_path.replace(
        os.sep + "JPEGImages" + os.sep, os.sep + "SegmentationClass" + os.sep)
    gt_path = os.path.splitext(gt_path)[0] + ".png"
    if not os.path.isfile(gt_path):
        return None
    # np.array on the palette png yields raw class ids (not colors).
    return np.array(Image.open(gt_path))


def gather_images(args):
    """Return the list of image paths to process from --img / --dir / --voc-random."""
    if args.img:
        return [args.img]
    if args.voc_random:
        return sample_voc_val(args.voc_random, args.seed)
    paths = []
    for ext in _IMG_EXTS:
        paths += glob.glob(os.path.join(args.dir, f"*{ext}"))
        paths += glob.glob(os.path.join(args.dir, f"*{ext.upper()}"))
    return sorted(paths)


@torch.no_grad()
def segment_one(model, transform, device, path):
    """Run segmentation on one image file.

    Input:
        model, transform, device; path to the image.
    Output:
        (image, pred) -- the original PIL image and the predicted mask as a
        [H, W] long tensor of class ids at the ORIGINAL resolution.
    """
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size

    # Preprocess: pad to /8 + normalize -- NO resize (unlike detection, no
    # coordinate rescaling is ever needed; we just slice the padding off the
    # prediction below). mask=None: pure inference has no label.
    img_t, _ = transform(image, None)
    img_t = img_t.unsqueeze(0).to(device)     # [1, 3, H', W']

    # Forward -> [1, 21, H', W'] logits at padded size; argmax -> class ids.
    logits = model(img_t)
    pred = logits.argmax(dim=1)[0].cpu()      # [H', W']

    # Padding was right/bottom only, so the original area is the top-left crop.
    pred = pred[:orig_h, :orig_w]
    return image, pred


def main():
    args = parse_args()
    device = get_device(args.device)
    # Start from a clean results folder: delete the previous one if it exists,
    # then recreate it, so each run's output isn't mixed with stale files.
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    # Build model + load weights.
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    neck_hidden_channels=config.NECK_HIDDEN_CHANNELS,
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    atrous_rate=config.ATROUS_RATE,
                    neck_dropout=config.NECK_DROPOUT).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights: {args.weights}")

    transform = get_eval_transforms(config.IMAGENET_MEAN, config.IMAGENET_STD,
                                    config.SIZE_DIVISOR, config.IGNORE_INDEX)

    paths = gather_images(args)
    if not paths:
        print("No images found.")
        return

    for path in paths:
        # File stem (e.g. "2007_000033"), used to name the output trio so the
        # files sort right next to each other when browsing.
        stem = os.path.splitext(os.path.basename(path))[0]

        # --- Prediction: overlay + raw colorized mask ---
        image, pred = segment_one(model, transform, device, path)
        overlay_mask(image, pred, alpha=args.alpha).save(
            os.path.join(args.out, f"{stem}_overlay.jpg"))
        colorize_mask(pred).save(os.path.join(args.out, f"{stem}_pred.png"))

        # Report which classes the model found (background excluded).
        found = sorted(set(int(c) for c in torch.unique(pred)) - {0})
        names = [config.VOC_SEG_CLASSES[c] for c in found]
        msg = f"{stem}: {', '.join(names) if names else 'background only'}"

        # --- Ground-truth mask (only if a VOC segmentation label exists) ---
        gt = load_voc_gt(path)
        if gt is not None:
            colorize_mask(gt).save(os.path.join(args.out, f"{stem}_gt.png"))
            msg += "  [gt saved]"

        print(msg)


if __name__ == "__main__":
    main()

"""PASCAL VOC 2012 segmentation dataset.

Responsibilities of this file (mirror of the detection projects' dataset/voc.py):
  1. Download VOC2012 into DATA_ROOT -- or transparently reuse the copy a
     detection project already downloaded (see config.DATA_ROOT): the VOC2012
     trainval archive already contains the segmentation labels. The download
     tries several mirrors (the official Oxford host goes down for weeks at a
     time), so it works on a fresh cloud machine.
  2. Wrap torchvision's VOCSegmentation and convert each sample into
        (image [3, H, W] float tensor, mask [H, W] long tensor)
     where mask holds per-pixel class ids 0..20, with 255 = ignore.
  3. Optionally add SBD ("VOC aug") extra training images (config.USE_SBD).

What is NOT here, compared to detection: no XML parsing (labels are plain
pngs), no box normalization, and NO custom collate_fn -- training crops are
all CROP_SIZE x CROP_SIZE so the default collate stacks them into
([B,3,S,S], [B,S,S]), and the val loader uses batch_size=1 (original sizes).
Segmentation's data plumbing is genuinely simpler than detection's.

About the label pngs: VOC stores masks as PALETTE images. Reading one with
np.array() yields the RAW class ids (0..20, 255) -- the famous colors you see
in an image viewer come from the png's palette table, not the pixel values.

Splits (ImageSets/Segmentation/): train = 1464 images, val = 1449. Only ~2.9k
of VOC2012's 17k images have segmentation masks -- that's why the SBD extra
labels exist and matter (see SBDSegDataset below).

How to download: python dataset/voc.py --download
(train.py also downloads automatically when the data is missing, so a fresh
cloud run needs no separate step: `python train.py` alone is enough.)
"""

import os
import sys

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import VOCSegmentation
from torchvision.datasets.utils import download_and_extract_archive

# Make the project root importable so `import config` works whether this file
# is run directly (python dataset/voc.py) or imported as a package (dataset.voc).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402

# Import our transforms (package-relative when imported, plain when run as script)
try:
    from .transforms import get_train_transforms, get_eval_transforms
except ImportError:
    from transforms import get_train_transforms, get_eval_transforms


def _build_transforms(train: bool):
    """Pick the joint (img, mask) pipeline for train vs. eval (see transforms.py)."""
    if train:
        return get_train_transforms(
            config.CROP_SIZE, config.SCALE_RANGE,
            config.IMAGENET_MEAN, config.IMAGENET_STD, config.IGNORE_INDEX)
    return get_eval_transforms(
        config.IMAGENET_MEAN, config.IMAGENET_STD,
        config.SIZE_DIVISOR, config.IGNORE_INDEX)


class VOCSegDataset(Dataset):
    """VOC2012 segmentation split returning (image tensor, mask tensor).

    Args:
        image_set (str): "train" (1464 images) / "val" (1449) / "trainval".
        train (bool): True -> random scale/crop/flip augmentation (fixed
            CROP_SIZE output); False -> pad-only eval pipeline (original size).
        download (bool): download the data if missing.
    """

    def __init__(self, image_set="train", train=True, download=False):
        # torchvision handles download, split lists and png loading; it yields
        # (PIL RGB image, PIL palette mask) pairs.
        self.voc = VOCSegmentation(
            root=config.DATA_ROOT, year="2012", image_set=image_set,
            download=download,
        )
        self.transforms = _build_transforms(train)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        """Return one processed sample.

        Output:
            img:  float tensor [3, H, W] (train: H=W=CROP_SIZE; eval: padded
                  original size).
            mask: long tensor [H, W] of class ids 0..20 (255 = ignore).
        """
        img, mask = self.voc[idx]           # (PIL image, PIL palette mask)
        return self.transforms(img, mask)


class SBDSegDataset(Dataset):
    """SBD "VOC aug" segmentation data (optional extra TRAINING images).

    SBD (Semantic Boundaries Dataset) provides segmentation masks for ~11k
    VOC2012 images that the official release left unlabeled. The standard
    "aug" recipe trains on VOC2012-train + SBD and evaluates on VOC2012-val.

    image_set "train_noval" is the crucial choice: it is SBD's train list with
    every VOC2012-VAL image REMOVED. Using plain "train"/"val" from SBD would
    leak evaluation images into training and inflate mIoU.

    Requires scipy (SBD masks ship as .mat files; torchvision reads them with
    scipy.io). Download is ~1.4 GB and the mirror is occasionally down -- if
    download=True fails, fetch benchmark.tgz manually and extract it under
    <DATA_ROOT>/sbd/.
    """

    def __init__(self, image_set="train_noval", download=False):
        from torchvision.datasets import SBDataset  # import lazily: needs scipy
        # mode="segmentation" -> targets are class-id masks (mode="boundaries"
        # would give edge maps, a different task).
        self.sbd = SBDataset(
            root=os.path.join(config.DATA_ROOT, "sbd"), image_set=image_set,
            mode="segmentation", download=download,
        )
        # SBD is training-only data here, so always the augmentation pipeline.
        self.transforms = _build_transforms(train=True)

    def __len__(self):
        return len(self.sbd)

    def __getitem__(self, idx):
        """Output: same contract as VOCSegDataset (img [3,S,S], mask [S,S])."""
        img, mask = self.sbd[idx]           # (PIL image, PIL mask)
        return self.transforms(img, mask)


def build_train_dataset():
    """The training set: VOC2012 seg train, plus SBD when config.USE_SBD.

    Output:
        a Dataset (possibly a ConcatDataset) of (img, mask) samples.
        1464 images without SBD, ~10.5k with it.
    """
    sets = [VOCSegDataset(image_set="train", train=True)]
    if config.USE_SBD:
        sets.append(SBDSegDataset(image_set="train_noval"))
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)


# -----------------------------------------------------------------------------
# Download (cloud-friendly: presence check + mirror fallback + resumable)
# -----------------------------------------------------------------------------
# The VOC2012 trainval archive (images + detection XMLs + segmentation pngs).
# The md5 is the official one (same value torchvision pins), so a corrupted or
# truncated download is detected instead of silently extracted.
_VOC2012_FILENAME = "VOCtrainval_11-May-2012.tar"
_VOC2012_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
# Tried in order. The official Oxford host is the canonical source but goes
# down for weeks at a time; pjreddie's mirror (the YOLO author's) serves the
# byte-identical archive and is what most people fall back to.
_VOC2012_URLS = [
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
]


def voc_seg_present() -> bool:
    """True if the VOC2012 SEGMENTATION data is already extracted in DATA_ROOT.

    We probe SegmentationClass/ (the label pngs) rather than just VOCdevkit/,
    so a partially-extracted archive doesn't count as present.
    """
    return os.path.isdir(os.path.join(
        config.DATA_ROOT, "VOCdevkit", "VOC2012", "SegmentationClass"))


def download_voc():
    """Download + extract VOC2012 trainval (~2 GB) into DATA_ROOT.

    Behavior (designed for fresh cloud machines):
      * If the data is already extracted (e.g. config.DATA_ROOT resolved to a
        detection project's copy), this returns immediately.
      * Otherwise the mirrors in _VOC2012_URLS are tried IN ORDER, so one dead
        host doesn't block training. The md5 check rejects corrupt files.
      * Re-running after an interrupted attempt is cheap:
        download_and_extract_archive skips the download when the tar already
        exists with the right md5 and just re-extracts.
    """
    if voc_seg_present():
        print(f"VOC2012 already present at: {config.DATA_ROOT} (skipping download)")
    else:
        os.makedirs(config.DATA_ROOT, exist_ok=True)
        last_err = None
        for url in _VOC2012_URLS:
            try:
                print(f"Downloading VOC2012 from: {url}")
                download_and_extract_archive(
                    url, download_root=config.DATA_ROOT,
                    filename=_VOC2012_FILENAME, md5=_VOC2012_MD5)
                last_err = None
                break
            except Exception as e:  # dead mirror / 404 / md5 mismatch -> next one
                print(f"  failed: {e}")
                last_err = e
        if last_err is not None:
            raise RuntimeError(
                "All VOC2012 mirrors failed. Download "
                f"{_VOC2012_FILENAME} manually (any mirror), place it in "
                f"{config.DATA_ROOT} and extract it there "
                "(tar -xf), then rerun.") from last_err

    # Optional SBD extra training data (config.USE_SBD).
    if config.USE_SBD:
        sbd_root = os.path.join(config.DATA_ROOT, "sbd")
        print(f"Downloading SBD into: {sbd_root} (~1.4 GB, needs scipy)...")
        from torchvision.datasets import SBDataset
        try:
            SBDataset(sbd_root, image_set="train_noval",
                      mode="segmentation", download=True)
        except Exception as e:
            # SBD has a single (flaky) official host and no good mirror; give
            # actionable instructions instead of a bare stack trace.
            raise RuntimeError(
                f"SBD download failed ({e}). Fetch benchmark.tgz manually "
                f"and extract it under {sbd_root}, or set config.USE_SBD "
                "= False to train on VOC2012 alone.") from e
    print("Done.")


# ---- Run directly ------------------------------------------------------------
#   python dataset/voc.py --download   # download the dataset
#   python dataset/voc.py              # offline self-test (no download needed)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        download_voc()
        raise SystemExit

    # ---- Offline self-test: exercise the (img, mask) pipeline + batching ----
    import numpy as np
    from PIL import Image

    # A fake VOC-like sample: 500x375 photo + palette-style mask with a
    # "dog" (12) region, a "person" (15) region and a 255 void strip.
    img = Image.new("RGB", (500, 375), color=(123, 116, 103))
    mask_arr = np.zeros((375, 500), dtype=np.uint8)
    mask_arr[50:200, 60:250] = 12
    mask_arr[210:340, 260:460] = 15
    mask_arr[200:210, :] = 255
    mask = Image.fromarray(mask_arr)

    train_tfm = _build_transforms(train=True)
    img_t, mask_t = train_tfm(img, mask)
    print("train sample:", tuple(img_t.shape), tuple(mask_t.shape),
          "(expected (3, 480, 480) (480, 480))")
    print("mask ids present:", sorted(torch.unique(mask_t).tolist()))

    # Default collate works because every training crop has the same size --
    # this replaces the detection projects' custom voc_collate_fn.
    from torch.utils.data import default_collate
    images, masks = default_collate([(img_t, mask_t), (img_t, mask_t)])
    print("batched:", tuple(images.shape), tuple(masks.shape),
          "(expected (2, 3, 480, 480) (2, 480, 480))")

    eval_tfm = _build_transforms(train=False)
    img_e, mask_e = eval_tfm(img, mask)
    print("eval sample:", tuple(img_e.shape), tuple(mask_e.shape),
          "(expected (3, 384, 512) (384, 512))")

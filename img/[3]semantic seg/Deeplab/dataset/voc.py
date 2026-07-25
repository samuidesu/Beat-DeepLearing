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

import hashlib
import math
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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

# SBD ("VOC aug") archive. torchvision's SBDataset(download=True) would fetch
# this itself, but with a SINGLE connection and NO resume -- and the Berkeley
# host throttles PER CONNECTION, so from far-away networks (e.g. mainland
# China) one stream crawls at ~50 kB/s and 1.4 GB takes 7+ hours. We instead
# fetch the archive ourselves with _download_segmented() below (N parallel
# byte-range connections, each resumable), then hand over to SBDataset: it
# sees the finished file, verifies the md5, SKIPS its own download and just
# extracts. The md5 is torchvision's pinned value, so we and it agree on what
# a "complete" archive is.
_SBD_FILENAME = "benchmark.tgz"
_SBD_MD5 = "82b4d87ceb2ed10f6038a1cba92111cb"
_SBD_URLS = [
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
]
# Parallel byte-range connections. The speedup comes from the throttle being
# per-connection: 16 throttled streams ~ 16x the single-stream speed. Raising
# this further gives diminishing returns and risks the server refusing.
_SBD_CONNECTIONS = 16


# -----------------------------------------------------------------------------
# Segmented (multi-connection) resumable downloader
# -----------------------------------------------------------------------------
def _md5_of(path: str) -> str:
    """md5 hex digest of a file, streamed in 1 MB blocks (constant memory).

    Input:  path to an existing file.
    Output: 32-char lowercase hex digest string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _probe_url(url: str):
    """Ask the server for the file size and whether byte ranges are supported.

    Sends a GET with "Range: bytes=0-0" (a 1-byte request):
      * status 206 + "Content-Range: bytes 0-0/<total>"  -> ranges supported,
        total parsed from after the "/".
      * status 200 -> server ignored the Range header: single stream only,
        total = Content-Length.

    Input:  url to probe.
    Output: (supports_ranges: bool, total_bytes: int).
    """
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status == 206:
            total = int(resp.headers["Content-Range"].rsplit("/", 1)[-1])
            return True, total
        return False, int(resp.headers.get("Content-Length", 0))


def _fetch_range(url, part_path, start, end, progress, retries: int = 3):
    """Download bytes [start, end] (inclusive) of `url` into `part_path`.

    Resumable: if part_path already holds k bytes from an earlier attempt,
    the request asks for bytes start+k..end, so finished bytes are NEVER
    re-downloaded -- a dropped connection costs nothing but the retry.

    Input:
        url, part_path: source and destination of this segment.
        start, end: absolute byte offsets of the segment within the file.
        progress: callable(n_new_bytes) -- called per chunk (thread-safe;
            tqdm.update holds an internal lock).
        retries: how many times to re-open a dropped connection.
    """
    expected = end - start + 1
    last_err = None
    for _ in range(retries):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have > expected:
            # Leftover from a run with a different split layout: start over.
            os.remove(part_path)
            have = 0
        if have == expected:
            return
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start + have}-{end}",
                         "User-Agent": "Mozilla/5.0"})
            # "ab" append mode is what makes the resume work: each retry
            # continues writing exactly where the last attempt stopped.
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(part_path, "ab") as f:
                for chunk in iter(lambda: resp.read(1 << 16), b""):
                    f.write(chunk)
                    progress(len(chunk))
            if os.path.getsize(part_path) == expected:
                return
        except Exception as e:  # timeout / reset / 503 -> retry from `have`
            last_err = e
    raise RuntimeError(
        f"segment {os.path.basename(part_path)} incomplete "
        f"after {retries} attempts") from last_err


def _download_segmented(url, dest, md5, connections: int = _SBD_CONNECTIONS):
    """Multi-connection resumable download of `url` into `dest`, md5-verified.

    How it works:
      1. Probe the server: total size + byte-range support (no ranges -> falls
         back to ONE resumable stream, still better than torchvision's).
      2. Split [0, total) into `connections` equal segments, download them in
         parallel threads into dest.part0..partN (each segment resumes its own
         partial file, so Ctrl-C + rerun loses nothing).
      3. Concatenate the parts into `dest`, delete the parts, verify the md5.

    Input:
        url: source URL.
        dest: final file path (e.g. .../sbd/benchmark.tgz).
        md5: expected hex digest; mismatch deletes the file and raises.
        connections: number of parallel range requests.
    """
    # Already fully downloaded? (md5, not just size, so a truncated or corrupt
    # file -- e.g. torchvision's aborted single-stream attempt -- never
    # passes; it is removed and the segmented download starts cleanly.)
    if os.path.exists(dest):
        print(f"  found existing {os.path.basename(dest)}, checking md5...")
        if _md5_of(dest) == md5:
            print("  already complete (md5 OK), skipping download")
            return
        print("  incomplete/corrupt -> removing and re-downloading")
        os.remove(dest)

    supports_ranges, total = _probe_url(url)
    if total <= 0:
        raise RuntimeError(f"server reported no file size for {url}")
    if not supports_ranges:
        connections = 1  # degrade gracefully: one (still resumable) stream

    # Split into `connections` contiguous segments covering [0, total).
    part_size = math.ceil(total / connections)
    ranges = [(i * part_size, min((i + 1) * part_size, total) - 1)
              for i in range(connections)]
    part_paths = [f"{dest}.part{i}" for i in range(len(ranges))]

    # Bytes already on disk from a previous interrupted run (for the progress
    # bar's starting point -- they will not be downloaded again).
    already = sum(
        min(os.path.getsize(p), e - s + 1)
        for p, (s, e) in zip(part_paths, ranges) if os.path.exists(p))

    try:
        from tqdm import tqdm
        bar = tqdm(total=total, initial=already, unit="B",
                   unit_scale=True, desc=f"  {connections} connections")
        progress = bar.update  # thread-safe (tqdm locks internally)
    except ImportError:
        bar = None

        def progress(n):  # no tqdm -> silent (sizes printed at the end)
            pass

    try:
        # One thread per segment; ThreadPoolExecutor's context manager joins
        # them, and result() re-raises the first segment failure, if any.
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_fetch_range, url, p, s, e, progress)
                for p, (s, e) in zip(part_paths, ranges)
            ]
            for f in futures:
                f.result()
    finally:
        if bar is not None:
            bar.close()

    # Stitch the segments together in order, then clean up.
    with open(dest, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
    for p in part_paths:
        os.remove(p)

    if _md5_of(dest) != md5:
        os.remove(dest)
        raise RuntimeError(
            f"md5 mismatch for {dest} -- download corrupt, removed; rerun")
    print(f"  downloaded {os.path.basename(dest)} ({total / 1e9:.2f} GB, md5 OK)")


def voc_seg_present() -> bool:
    """True if the VOC2012 SEGMENTATION data is already extracted in DATA_ROOT.

    We probe SegmentationClass/ (the label pngs) rather than just VOCdevkit/,
    so a partially-extracted archive doesn't count as present.
    """
    return os.path.isdir(os.path.join(
        config.DATA_ROOT, "VOCdevkit", "VOC2012", "SegmentationClass"))


def sbd_present() -> bool:
    """True if SBD is fully extracted and ready for SBDataset(download=False).

    Probes the three things SBDataset needs after its own download step:
      img/            extracted images
      cls/            extracted .mat class masks
      train_noval.txt the split list (a SEPARATE small download from JHU --
                      NOT inside benchmark.tgz, which is why a manually-placed
                      archive alone is not enough; the download step fetches it)

    A partial benchmark.tgz on disk does NOT count as present -- extraction
    must have completed. Kept in sync with the trigger in build_dataloaders so
    a missing SBD forces download_voc() even when VOC2012 already exists.
    """
    if not config.USE_SBD:
        return True  # SBD disabled -> nothing to download, treat as satisfied
    sbd_root = os.path.join(config.DATA_ROOT, "sbd")
    return all(os.path.exists(os.path.join(sbd_root, p))
               for p in ("img", "cls", "train_noval.txt"))


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
        # Fully extracted already? (img/ + cls/ + train_noval.txt) -> skip.
        if sbd_present():
            print(f"SBD already present at: {sbd_root} (skipping download)")
        else:
            print(f"Downloading SBD into: {sbd_root} (~1.4 GB, needs scipy)...")
            os.makedirs(sbd_root, exist_ok=True)
            archive = os.path.join(sbd_root, _SBD_FILENAME)

            # Step 1: fetch benchmark.tgz ourselves -- segmented + resumable,
            # ~connections-times faster than torchvision's single stream on
            # per-connection-throttled links (see _download_segmented).
            last_err = None
            for url in _SBD_URLS:
                try:
                    _download_segmented(url, archive, _SBD_MD5)
                    last_err = None
                    break
                except Exception as e:
                    print(f"  failed: {e}")
                    last_err = e
            if last_err is not None:
                raise RuntimeError(
                    "SBD download failed on all mirrors. Fetch it manually "
                    "with a multi-connection downloader, e.g.\n"
                    f"  aria2c -x16 -s16 -c '{_SBD_URLS[0]}' "
                    f"-d '{sbd_root}' -o {_SBD_FILENAME}\n"
                    "then rerun (the finished file is detected and reused), "
                    "or set config.USE_SBD = False to train on VOC2012 alone."
                ) from last_err

            # Step 2: hand over to torchvision. The archive is in place with
            # the right md5, so SBDataset skips its own download and only
            # extracts + arranges img/ cls/ train_noval.txt under sbd_root.
            from torchvision.datasets import SBDataset
            try:
                SBDataset(sbd_root, image_set="train_noval",
                          mode="segmentation", download=True)
            except Exception as e:
                raise RuntimeError(
                    f"SBD extraction failed ({e}). If the error is about an "
                    f"existing folder, delete {sbd_root} (KEEP a copy of "
                    f"{_SBD_FILENAME} elsewhere and put it back) and rerun."
                ) from e
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

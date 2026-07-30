"""Image + segmentation-mask JOINT transforms / augmentation.

THE key difference from the detection pipeline: the label is now an IMAGE (an
[H, W] map of class ids), not a list of boxes. Consequences:

  1. Every GEOMETRIC transform (scale, flip, crop, pad) must be applied to the
     image and the mask with EXACTLY the same parameters, or pixels and labels
     silently misalign.
  2. The mask must always be resampled with NEAREST interpolation. Bilinear
     would AVERAGE neighbouring class ids into meaningless in-between values
     (e.g. blending sheep=17 with person=15 yields 16 = pottedplant!).
     Class ids are categories, not quantities -- never interpolate them.
  3. Photometric transforms (ColorJitter, Normalize) touch only the image.

Every transform is a callable `t(img, mask) -> (img, mask)`; `mask` may be
None at pure inference (segment/segment.py) and is then passed through
untouched. Same Compose pattern as the detection projects' (img, boxes)
pipelines.

Pipelines (see get_train_transforms / get_eval_transforms):
  Train: RandomScale -> HFlip -> ColorJitter -> PadIfNeeded -> RandomCrop
         -> ToTensor -> Normalize            => fixed CROP_SIZE x CROP_SIZE
  Eval:  PadToMultiple(32) -> ToTensor -> Normalize   => original size kept

Padding rules (both pipelines): the IMAGE is padded with the ImageNet mean
color, which becomes ~0 ("neutral gray") after Normalize; the MASK is padded
with IGNORE_INDEX (255), so padded pixels are excluded from the loss AND the
mIoU metric through the exact same mechanism as VOC's void contours.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as tvt
from torchvision.transforms import InterpolationMode

# ImageNet mean as 0-255 ints: the pad color for images. After Normalize this
# value maps to ~0 in every channel, i.e. the "average pixel".
_MEAN_FILL = (124, 116, 104)


class Compose:
    """Chain several (img, mask) transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        # Apply each transform in order, threading img and mask through.
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomScale:
    """Rescale image AND mask by one random factor from `scale_range`.

    The segmentation counterpart of detection's RandomAffine scale jitter:
    varies every object's apparent size so the model can't overfit to one
    scale. Image uses BILINEAR (values are colors, interpolation is fine);
    mask uses NEAREST (values are class ids -- see module docstring).
    """

    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, img, mask):
        # One factor for both axes: aspect ratio is preserved (unlike the
        # detection projects' square resize).
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = img.size                       # PIL size is (width, height)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = TF.resize(img, [nh, nw], interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, [nh, nw], interpolation=InterpolationMode.NEAREST)
        return img, mask


class RandomHorizontalFlip:
    """Randomly mirror image AND mask left<->right together.

    Compare the detection version: there the image flipped and box centers
    were remapped cx -> 1-cx; here the label is an image too, so it simply
    flips the same way. No coordinate math at all.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)
        return img, mask


class ColorJitter:
    """Randomly perturb brightness/contrast/saturation/hue (IMAGE ONLY).

    Photometric changes cannot alter what class a pixel belongs to, so the
    mask is untouched -- same reasoning as boxes being untouched in detection.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = tvt.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, mask):
        return self.jitter(img), mask


class PadIfNeeded:
    """Pad right/bottom so both sides reach at least `size` (pre-crop safety).

    After RandomScale with factor < 1 an image can be smaller than the crop
    window (e.g. 500x375 * 0.5 = 250x188 < 480); pad it up so RandomCrop always
    has room. Image pad = ImageNet mean color, mask pad = ignore_index.
    """

    def __init__(self, size: int, ignore_index: int = 255):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        w, h = img.size
        pad_r = max(0, self.size - w)     # extend right edge
        pad_b = max(0, self.size - h)     # extend bottom edge
        if pad_r > 0 or pad_b > 0:
            # TF.pad padding order: [left, top, right, bottom].
            img = TF.pad(img, [0, 0, pad_r, pad_b], fill=_MEAN_FILL)
            if mask is not None:
                mask = TF.pad(mask, [0, 0, pad_r, pad_b], fill=self.ignore_index)
        return img, mask


class RandomCrop:
    """Cut one random size x size window -- the SAME window -- from img and mask.

    Run PadIfNeeded first so the window always fits. This is the step that
    fixes the training tensor size (=> the default DataLoader collate can
    stack samples; no custom collate_fn needed, unlike detection).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        # Sample the top-left corner ONCE; randint bounds are inclusive.
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        img = TF.crop(img, top, left, self.size, self.size)
        if mask is not None:
            mask = TF.crop(mask, top, left, self.size, self.size)
        return img, mask


class PadToMultiple:
    """Pad right/bottom to the next multiple of `divisor` (EVAL pipeline).

    The network downsamples by up to 32, so H and W must be divisible by 32
    for the feature grids (and the 8x-upsampled logits) to line up exactly.
    Instead of resizing (which would distort and require rescaling the
    prediction), we pad -- and because the mask pad value is ignore_index,
    padded pixels simply never count in loss or metric. Prediction for the
    original area can be recovered by slicing [:H, :W].
    """

    def __init__(self, divisor: int = 32, ignore_index: int = 255):
        self.divisor = divisor
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        w, h = img.size
        pad_r = (-w) % self.divisor       # e.g. w=500 -> pad 12 -> 512
        pad_b = (-h) % self.divisor       # e.g. h=375 -> pad 9  -> 384
        if pad_r > 0 or pad_b > 0:
            img = TF.pad(img, [0, 0, pad_r, pad_b], fill=_MEAN_FILL)
            if mask is not None:
                mask = TF.pad(mask, [0, 0, pad_r, pad_b], fill=self.ignore_index)
        return img, mask


class ToTensor:
    """PIL -> tensors. Image: float [3,H,W] in [0,1]. Mask: LONG [H,W] ids.

    The mask png is a PALETTE image: np.array() on it yields the RAW class ids
    (0..20 and 255) as uint8 -- NOT colors; the colors seen in a viewer come
    from the png's palette table. We convert ids to int64 ("long") because
    that is the dtype CrossEntropyLoss requires for class targets.
    NOTE: TF.to_tensor would be WRONG for the mask -- it rescales to [0,1]
    floats, destroying the integer ids.
    """

    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        if mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


class Normalize:
    """Normalize the image tensor with per-channel mean/std (IMAGE ONLY)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = TF.normalize(img, self.mean, self.std)
        return img, mask


def get_train_transforms(crop_size: int, scale_range, mean, std,
                         ignore_index: int = 255) -> Compose:
    """Training pipeline: scale jitter + flip + color + pad&crop + normalize.

    Input:
        crop_size: output size (square), multiple of 32.
        scale_range: (lo, hi) random rescale factor range.
        mean/std: per-channel normalization stats.
        ignore_index: mask pad value (excluded from loss/metric).
    Output:
        Compose mapping (PIL img, PIL mask) ->
        (float tensor [3,crop,crop], long tensor [crop,crop]).
    """
    return Compose([
        RandomScale(scale_range),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(),
        PadIfNeeded(crop_size, ignore_index),
        RandomCrop(crop_size),
        ToTensor(),
        Normalize(mean, std),
    ])


def get_eval_transforms(mean, std, divisor: int = 32,
                        ignore_index: int = 255) -> Compose:
    """Eval/inference pipeline: pad to /32 + normalize. NO resize, NO aug.

    Output keeps (near-)original resolution: (PIL img, PIL mask or None) ->
    (float tensor [3,H',W'], long tensor [H',W'] or None), where H'/W' are
    H/W rounded up to the next multiple of `divisor`.
    """
    return Compose([
        PadToMultiple(divisor, ignore_index),
        ToTensor(),
        Normalize(mean, std),
    ])


# ---- Quick self-test: run this file directly to verify shapes/dtypes ---------
# python dataset/transforms.py
if __name__ == "__main__":
    from PIL import Image

    # A fake 500x375 photo and a fake mask with a "person" (15) rectangle on
    # background (0), plus a 255 void border strip -- the shapes VOC really has.
    img = Image.new("RGB", (500, 375), color=(123, 116, 103))
    mask_arr = np.zeros((375, 500), dtype=np.uint8)
    mask_arr[100:300, 150:350] = 15          # person block
    mask_arr[95:100, 150:350] = 255          # void contour strip
    mask = Image.fromarray(mask_arr)         # mode "L" behaves like VOC's "P"

    train_tfm = get_train_transforms(480, (0.5, 2.0),
                                     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_t, mask_t = train_tfm(img, mask)
    print("train img:", tuple(img_t.shape), img_t.dtype, "(expected (3, 480, 480) float32)")
    print("train mask:", tuple(mask_t.shape), mask_t.dtype, "(expected (480, 480) int64)")
    print("train mask values:", sorted(torch.unique(mask_t).tolist()),
          "(subset of [0, 15, 255])")

    eval_tfm = get_eval_transforms([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_e, mask_e = eval_tfm(img, mask)
    print("eval img:", tuple(img_e.shape), "(expected (3, 384, 512): 375->384, 500->512)")
    print("eval mask:", tuple(mask_e.shape), "(expected (384, 512))")
    # Padded border must be ignore (255), so it never affects loss/metric.
    print("eval mask bottom-right value:", int(mask_e[-1, -1]), "(expected 255)")

    # Inference path: mask=None must pass through without errors.
    img_only, none_mask = eval_tfm(img, None)
    print("mask=None passthrough:", tuple(img_only.shape), none_mask)

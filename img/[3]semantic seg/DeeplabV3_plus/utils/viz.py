"""Visualization: colorize class-id masks with the official VOC palette.

Detection drew boxes; segmentation paints masks. VOC label pngs are PALETTE
images: the pixel VALUE is the class id, and the famous colors (dark-red
aeroplane, pink person, ...) come from the png's palette table. We regenerate
that exact palette, so our predictions render identically to the GT masks
people are used to seeing.

Palette generation is the official bit-trick from the VOC devkit: the class
id's bits are distributed, 3 at a time, into the R/G/B channels from the most
significant bit down. It maps 0 -> black (background), 1 -> (128,0,0),
2 -> (0,128,0), ..., and conveniently 255 -> (224,224,192), the cream "void"
color of the ignore contours.
"""

import numpy as np
import torch
from PIL import Image


def voc_palette() -> list:
    """Build the 256-entry VOC color palette.

    Output:
        flat list of 768 ints [R0, G0, B0, R1, G1, B1, ...], the format
        PIL's Image.putpalette expects.
    """
    palette = [0] * (256 * 3)
    for cid in range(256):
        c = cid
        r = g = b = 0
        # Spread the id's bits over R/G/B, MSB first: bit 3k -> R, 3k+1 -> G,
        # 3k+2 -> B, each landing at bit position (7 - k) of its channel.
        for k in range(8):
            r |= ((c >> 0) & 1) << (7 - k)
            g |= ((c >> 1) & 1) << (7 - k)
            b |= ((c >> 2) & 1) << (7 - k)
            c >>= 3
        palette[3 * cid: 3 * cid + 3] = [r, g, b]
    return palette


# Build once at import; every colorize call reuses it.
_PALETTE = voc_palette()


def _to_uint8_array(mask) -> np.ndarray:
    """Accept a [H, W] tensor or ndarray of class ids, return uint8 ndarray."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=np.uint8)


def colorize_mask(mask) -> Image.Image:
    """Turn a class-id mask into a VOC-colored palette image.

    Input:
        mask: [H, W] tensor/ndarray of class ids (0..20, 255 = void).
    Output:
        PIL image in "P" (palette) mode -- same format as VOC's own label
        pngs; save it as .png to keep the ids losslessly.
    """
    img = Image.fromarray(_to_uint8_array(mask), mode="P")
    img.putpalette(_PALETTE)
    return img


def overlay_mask(image: Image.Image, mask, alpha: float = 0.55) -> Image.Image:
    """Blend the colorized mask over the photo (predictions "painted on").

    Background (0) and void (255) pixels keep the PURE photo, so only actual
    object regions get tinted -- much easier to judge than a full-frame blend.

    Input:
        image: the original PIL RGB photo.
        mask:  [H, W] class ids, SAME size as the image.
        alpha: tint strength for object pixels (0 = photo, 1 = solid color).
    Output:
        PIL RGB image.
    """
    arr = _to_uint8_array(mask)
    color = colorize_mask(arr).convert("RGB")
    image = image.convert("RGB")
    blended = Image.blend(image, color, alpha)

    # Composite: take `blended` where the mask is a real object class,
    # the untouched photo elsewhere. The "L" image is the selection mask
    # (255 = take blended).
    is_object = ((arr != 0) & (arr != 255)).astype(np.uint8) * 255
    return Image.composite(blended, image, Image.fromarray(is_object, mode="L"))


# ---- Quick self-test: run this file directly ---------------------------------
# python utils/viz.py
if __name__ == "__main__":
    # First few classes must match the canonical VOC colors.
    pal = voc_palette()
    print("class 0 (background):", pal[0:3], "(expected [0, 0, 0])")
    print("class 1 (aeroplane): ", pal[3:6], "(expected [128, 0, 0])")
    print("class 15 (person):   ", pal[45:48], "(expected [192, 128, 128])")
    print("id 255 (void):       ", pal[765:768], "(expected [224, 224, 192])")

    # Colorize + overlay a synthetic mask; check modes/sizes survive.
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[20:80, 30:90] = 15                    # a person block
    mask[:, 150:] = 255                        # a void strip
    photo = Image.new("RGB", (160, 100), color=(90, 120, 90))

    colored = colorize_mask(torch.as_tensor(mask))
    over = overlay_mask(photo, mask, alpha=0.55)
    print("colorized:", colored.mode, colored.size, "(expected P (160, 100))")
    print("overlay:  ", over.mode, over.size, "(expected RGB (160, 100))")
    # Overlay must keep the photo untouched where mask==0.
    print("corner pixel kept:", over.getpixel((0, 0)), "(expected (90, 120, 90))")

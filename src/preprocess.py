"""Reusable image geometry transforms for model input.

Modes (all produce an exact ``size x size`` RGB image):

- ``squash`` — stretch to ``(size, size)``, ignore aspect ratio (MegaLoc default).
- ``letterbox`` — fit inside ``(size, size)`` keeping aspect; pad with ``pad_color``.
- ``center_crop`` — scale so the short side covers ``size``, then crop the center.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from PIL import Image

# ImageNet mean as 0–255 RGB — natural fill for letterbox when the model later
# normalizes with ImageNet stats (padded pixels become ~zeros after Normalize).
IMAGENET_PAD_RGB = (124, 116, 104)


class TransformMode(str, Enum):
    SQUASH = "squash"
    LETTERBOX = "letterbox"
    CENTER_CROP = "center_crop"


def _as_mode(mode: TransformMode | str) -> TransformMode:
    if isinstance(mode, TransformMode):
        return mode
    try:
        return TransformMode(str(mode).strip().lower())
    except ValueError as e:
        raise ValueError(
            f"unknown transform mode {mode!r}; expected "
            f"{', '.join(m.value for m in TransformMode)}"
        ) from e


def squash(img: Image.Image, size: int) -> Image.Image:
    """Stretch to exact square, discarding aspect ratio."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.Resampling.BICUBIC)


def letterbox(
    img: Image.Image,
    size: int,
    *,
    pad_color: Iterable[int] = IMAGENET_PAD_RGB,
) -> Image.Image:
    """Fit inside a square canvas, pad the short axis (aspect preserved)."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError(f"invalid image size {w}x{h}")
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), tuple(pad_color))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Scale short side to ``size``, then center-crop to a square.

    Upscales if the source is smaller than ``size`` on either axis so the crop
    always yields exactly ``size x size``.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError(f"invalid image size {w}x{h}")
    # Cover the square: short side becomes `size`, long side stays >= size.
    scale = max(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def transform_image(
    img: Image.Image,
    size: int,
    mode: TransformMode | str = TransformMode.SQUASH,
    *,
    pad_color: Iterable[int] = IMAGENET_PAD_RGB,
) -> Image.Image:
    """Apply a geometry mode and return an RGB ``size x size`` image."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    m = _as_mode(mode)
    if m is TransformMode.SQUASH:
        return squash(rgb, size)
    if m is TransformMode.LETTERBOX:
        return letterbox(rgb, size, pad_color=pad_color)
    if m is TransformMode.CENTER_CROP:
        return center_crop(rgb, size)
    raise ValueError(f"unhandled transform mode: {m}")


def open_image(data: bytes | Image.Image) -> Image.Image:
    """Decode bytes to a PIL image, or pass a PIL image through."""
    if isinstance(data, Image.Image):
        return data
    from io import BytesIO

    with Image.open(BytesIO(data)) as im:
        # Load fully before the context closes the buffer.
        im.load()
        return im

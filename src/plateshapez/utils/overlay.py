from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def calculate_center_position(background: Image.Image, overlay: Image.Image) -> Tuple[int, int]:
    """Calculate position to center overlay on background."""
    bg_w, bg_h = background.size
    ov_w, ov_h = overlay.size
    x = (bg_w - ov_w) // 2
    y = (bg_h - ov_h) // 2
    return (x, y)


def paste_overlay(
    background: Image.Image, overlay: Image.Image, position: Tuple[int, int] | None = None
) -> Image.Image:
    """Paste overlay onto background at specified position (or center if None)."""
    result = background.copy()

    if position is None:
        position = calculate_center_position(background, overlay)

    # Ensure overlay has alpha channel for proper compositing
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    result.paste(overlay, position, overlay)
    return result


def get_overlay_region(
    overlay: Image.Image, position: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Get the region (x, y, width, height) occupied by overlay at position."""
    x, y = position
    w, h = overlay.size
    return (x, y, w, h)


def isolate_neutral_layer(
    rendered: Image.Image, neutral: int = 128, *, alpha_gain: float = 2.0
) -> Image.Image:
    """Drop a flat neutral-grey background, keeping only the perturbations.

    The perturbations are re-rendered onto a flat neutral-grey canvas (with no
    plate or vehicle underneath), so additive noise is centred on grey instead
    of clipping against the plate's white/black pixels.

    Transparency is scaled by how far each pixel deviates from the neutral grey,
    so faint noise stays see-through (a *clear* speckle rather than a solid grey
    block) while strong signal such as drawn shapes is fully opaque. Pixels the
    perturbations never touched are completely transparent.

    Args:
        rendered: A neutral-grey canvas with the perturbations applied to it.
        neutral: The flat grey level the canvas was filled with.
        alpha_gain: How quickly opacity ramps up with deviation from ``neutral``.
            1.0 maps the maximum possible deviation to full opacity; higher
            values make weak noise more visible.

    Returns:
        An RGBA :class:`PIL.Image.Image` the same size as ``rendered``.
    """
    arr = np.asarray(rendered.convert("RGB")).astype(np.int16)
    deviation = np.abs(arr - neutral).max(axis=-1)
    alpha = np.clip(deviation * alpha_gain, 0, 255).astype(np.uint8)

    opaque = alpha > 0
    out = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    out[opaque, :3] = arr[opaque].astype(np.uint8)
    out[..., 3] = alpha
    return Image.fromarray(out, "RGBA")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode."""
    return image.convert("RGB") if image.mode != "RGB" else image


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    return image.convert("RGBA") if image.mode != "RGBA" else image

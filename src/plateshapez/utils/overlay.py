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
    rendered: Image.Image, neutral: int = 128, *, contrast: float = 3.0
) -> Image.Image:
    """Drop a flat neutral-grey background, keeping only the perturbations.

    The perturbations are re-rendered onto a flat neutral-grey canvas (with no
    plate or vehicle underneath), so additive noise is centred on grey instead
    of clipping against the plate's white/black pixels — the layer therefore
    carries no plate text or border, only the real noise and shapes.

    The deviation from neutral grey is amplified by ``contrast`` so the noise
    reads as clearly visible speckle (not a flat grey wash), and every touched
    pixel is fully opaque. Pixels the perturbations never changed stay
    completely transparent.

    Args:
        rendered: A neutral-grey canvas with the perturbations applied to it.
        neutral: The flat grey level the canvas was filled with.
        contrast: Multiplier on each pixel's deviation from ``neutral``; higher
            values make low-intensity noise more visible.

    Returns:
        An RGBA :class:`PIL.Image.Image` the same size as ``rendered``.
    """
    arr = np.asarray(rendered.convert("RGB")).astype(np.int16)
    deviation = arr - neutral
    changed = np.any(deviation != 0, axis=-1)

    visible = np.clip(neutral + deviation * contrast, 0, 255).astype(np.uint8)
    out = np.zeros((*arr.shape[:2], 4), dtype=np.uint8)
    out[changed, :3] = visible[changed]
    out[changed, 3] = 255
    return Image.fromarray(out, "RGBA")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode."""
    return image.convert("RGB") if image.mode != "RGB" else image


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    return image.convert("RGBA") if image.mode != "RGBA" else image

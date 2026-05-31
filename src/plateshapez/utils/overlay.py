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


def extract_perturbation_delta(
    before: Image.Image, after: Image.Image, *, midpoint: int = 128
) -> Image.Image:
    """Isolate a single perturbation's signal (e.g. speckles/noise) with no plate.

    Captures the *change* a perturbation made by taking the signed per-pixel
    difference ``after - before`` and rendering it centred on a neutral mid-grey
    (``midpoint``). Pixels the perturbation did not touch stay fully transparent,
    so the result is a clean image of just that perturbation — the noise shows as
    grey speckle and drawn shapes as silhouettes — with the underlying plate and
    background removed entirely.

    Args:
        before: The composite immediately before this perturbation was applied.
        after: The composite immediately after this perturbation was applied.
        midpoint: Neutral grey level the signed delta is centred on.

    Returns:
        An RGBA :class:`PIL.Image.Image` the same size as ``after``.
    """
    before_arr = np.asarray(before.convert("RGB"), dtype=np.int16)
    after_arr = np.asarray(after.convert("RGB"), dtype=np.int16)

    if before_arr.shape == after_arr.shape:
        delta = after_arr - before_arr
        changed = np.any(delta != 0, axis=-1)
    else:
        # Geometric perturbations may change the canvas size; fall back to
        # showing the whole result relative to the neutral mid-grey.
        delta = after_arr - midpoint
        changed = np.ones(after_arr.shape[:2], dtype=bool)

    vis = np.clip(midpoint + delta, 0, 255).astype(np.uint8)
    out = np.zeros((*after_arr.shape[:2], 4), dtype=np.uint8)
    out[changed, :3] = vis[changed]
    out[changed, 3] = 255
    return Image.fromarray(out, "RGBA")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode."""
    return image.convert("RGB") if image.mode != "RGB" else image


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    return image.convert("RGBA") if image.mode != "RGBA" else image

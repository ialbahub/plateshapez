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


def extract_perturbation_layer(base: Image.Image, perturbed: Image.Image) -> Image.Image:
    """Isolate the perturbations (patterns and noise) onto a transparent layer.

    Returns an RGBA image that carries the perturbed RGB values wherever they
    differ from the clean ``base`` composite, and is fully transparent
    everywhere else. Compositing the returned layer back over ``base``
    reproduces ``perturbed`` exactly, so the layer is a lossless representation
    of just the adversarial patterns and noise, with the background removed.

    Args:
        base: The clean composite (background + overlay) before perturbations.
        perturbed: The composite after perturbations were applied.

    Returns:
        An RGBA :class:`PIL.Image.Image` of the same size as ``perturbed``.
    """
    base_arr = np.array(base.convert("RGB"))
    pert_arr = np.array(perturbed.convert("RGB"))

    if base_arr.shape == pert_arr.shape:
        changed = np.any(base_arr != pert_arr, axis=-1)
    else:
        # Geometric perturbations could change the canvas size; in that case the
        # whole perturbed image is treated as the perturbation layer.
        changed = np.ones(pert_arr.shape[:2], dtype=bool)

    layer = np.zeros((*pert_arr.shape[:2], 4), dtype=np.uint8)
    layer[..., :3] = pert_arr
    layer[..., 3] = np.where(changed, 255, 0).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode."""
    return image.convert("RGB") if image.mode != "RGB" else image


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    return image.convert("RGBA") if image.mode != "RGBA" else image

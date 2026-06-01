"""Recommended adversarial attack presets for defeating ALPR.

Validated against Fast-ALPR (the engine the ALPRovingGround suite uses):

* A **flat** semi-transparent grid/mesh over the plate defeats the ALPR
  *detector* (Fast-ALPR returns nothing on 6/6 plates) and PP-OCR/Tesseract,
  while the characters stay human-readable. This is the recommended *output*
  attack — it needs no viewing-angle change.
* A perspective/viewing-angle skew also defeats detection, but that is a
  capture condition for **testing** robustness, not something baked into the
  plate image. Use :func:`perspective_attack` only to test, not to ship.

``ALPR_DEFEAT`` is the default (flat grid). All parameters are configurable.
"""

from __future__ import annotations

from typing import Any

# The validated flat output attack: a fine grid that breaks plate detection.
ALPR_DEFEAT: list[dict[str, Any]] = [
    {"name": "grid", "params": {"spacing": 14, "width": 2, "alpha": 150}},
]


def grid_attack(
    spacing: int = 14,
    width: int = 2,
    alpha: int = 150,
    color: list[int] | None = None,
    diagonal: bool = False,
    noise: int = 0,
) -> list[dict[str, Any]]:
    """Build the flat grid attack (optionally add noise to disrupt recognition).

    Args:
        spacing: Pixels between grid lines (smaller = denser, harder to read).
        width: Line thickness in pixels.
        alpha: Line opacity 0-255 (lower = fainter / more readable).
        color: RGB line colour (defaults to light grey).
        diagonal: Use a diagonal mesh instead of axis-aligned.
        noise: Optional Gaussian noise added under the grid; 0 to disable.

    Returns:
        A perturbation list for ``DatasetGenerator(perturbations=...)``.
    """
    perts: list[dict[str, Any]] = []
    if noise > 0:
        perts.append({"name": "noise", "params": {"intensity": noise}})
    perts.append({
        "name": "grid",
        "params": {
            "spacing": spacing,
            "width": width,
            "alpha": alpha,
            "color": color or [245, 245, 245],
            "diagonal": diagonal,
        },
    })
    return perts


def perspective_attack(
    strength: float = 0.30, tilt: str = "v", noise: int = 18
) -> list[dict[str, Any]]:
    """Perspective skew — a TEST condition (viewing angle), not an output attack.

    Useful to check whether a detector survives steep angles; reliable detector
    defeat starts around 0.28 (vertical) / 0.36 (horizontal). Pair with
    ``confine_to_plate=False`` since it is geometric.
    """
    perts: list[dict[str, Any]] = []
    if noise > 0:
        perts.append({"name": "noise", "params": {"intensity": noise}})
    perts.append({"name": "perspective", "params": {"strength": strength, "tilt": tilt}})
    return perts

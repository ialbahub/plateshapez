"""Recommended adversarial attack presets for defeating ALPR.

Validated against Fast-ALPR (the engine the ALPRovingGround suite uses):
surface patterns (noise/shapes/glare/blur) do not reliably defeat the detector,
but a perspective/viewing-angle skew makes detection fail 100% of the time while
the plate stays human-readable — independent of plate colour or markings.

``ALPR_DEFEAT`` is the default attack: a steep vertical perspective skew plus a
touch of noise. Because perspective is geometric, generate it with
``confine_to_plate=False`` so the warp is not clipped back to the plate shape.
"""

from __future__ import annotations

from typing import Any

# The validated, reliable ALPR-defeating recipe (perspective is the key).
ALPR_DEFEAT: list[dict[str, Any]] = [
    {"name": "noise", "params": {"intensity": 18}},
    {"name": "perspective", "params": {"strength": 0.30, "tilt": "v"}},
]


def perspective_attack(
    strength: float = 0.30, tilt: str = "v", noise: int = 18
) -> list[dict[str, Any]]:
    """Build a perspective-skew attack (optionally with a little noise).

    Args:
        strength: Corner displacement as a fraction of the plate size. Reliable
            detector defeat starts around 0.28 (vertical) / 0.36 (horizontal).
        tilt: ``"v"`` (vertical / forward-back) or ``"h"`` (horizontal / yaw).
        noise: Gaussian noise intensity added before the warp; 0 to disable.

    Returns:
        A perturbation list usable as ``DatasetGenerator(perturbations=...)``
        (pair with ``confine_to_plate=False``).
    """
    perts: list[dict[str, Any]] = []
    if noise > 0:
        perts.append({"name": "noise", "params": {"intensity": noise}})
    perts.append({"name": "perspective", "params": {"strength": strength, "tilt": tilt}})
    return perts

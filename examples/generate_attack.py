#!/usr/bin/env python3
"""Generate an ALPR-defeating dataset with the perspective attack baked in.

Uses the validated ``ALPR_DEFEAT`` preset (perspective skew + light noise) as
the default perturbation. Perspective is geometric, so confinement is disabled.

Run with: uv run python examples/generate_attack.py
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from plateshapez.attacks import perspective_attack
from plateshapez.pipeline import DatasetGenerator
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background


def main() -> None:
    root = Path("dataset/attack_demo")
    ind = root / "in"
    (ind / "backgrounds").mkdir(parents=True, exist_ok=True)
    (ind / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background(size=(1600, 1100)).save(
        ind / "backgrounds" / "vehicle.jpg", "JPEG", quality=92
    )
    for text in ("4J9T7W", "AA9AFX", "BXR4821"):
        create_plate_overlay(text, size=(1200, 600)).save(ind / "overlays" / f"{text}.png")

    DatasetGenerator(
        bg_dir=ind / "backgrounds",
        overlay_dir=ind / "overlays",
        out_dir=root / "dataset",
        perturbations=cast(
            "list[DatasetGenerator.PerturbationConf]",
            perspective_attack(strength=0.30, tilt="v", noise=18),
        ),
        random_seed=7,
        confine_to_plate=False,  # perspective is geometric — do not clip it
        save_perturbation_layer=False,
    ).run(n_variants=3)
    print(f"✅ ALPR-attack dataset in {root / 'dataset' / 'images'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a plate-attack demo: plate, no-plate, and merged full versions.

Builds a vehicle background and a Michigan plate overlay with the library's
generators, runs the dataset pipeline (perturbations clipped to the plate), and
writes three full-resolution views to ``<out>/full/``:

* ``plate.png``    - the clean, readable plate composited on the vehicle
* ``no_plate.png`` - the isolated perturbations only (no plate), on white
* ``merged.png``   - the real perturbed composite (plate + perturbation)

Run with: uv run python examples/plate_attack_demo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from plateshapez.pipeline import DatasetGenerator
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position, ensure_rgb, ensure_rgba


def build_inputs(root: Path, plate_text: str) -> None:
    """Create the vehicle background and plate overlay under ``root``."""
    (root / "backgrounds").mkdir(parents=True, exist_ok=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(root / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    create_plate_overlay(plate_text).save(root / "overlays" / f"MI_{plate_text}.png")


def write_full_versions(root: Path, plate_text: str) -> Path:
    """Run the generator and assemble the plate/no_plate/merged trio."""
    ds = root / "dataset"
    DatasetGenerator(
        bg_dir=root / "backgrounds",
        overlay_dir=root / "overlays",
        out_dir=ds,
        perturbations=[
            {"name": "shapes", "params": {"num_shapes": 35, "min_size": 6, "max_size": 22}},
            {"name": "noise", "params": {"intensity": 35}},
        ],
        random_seed=4977,
        confine_to_plate=True,
    ).run(n_variants=1)

    stem = f"vehicle_MI_{plate_text}_000"
    full = root / "full"
    full.mkdir(exist_ok=True)

    # plate: clean composite (no perturbation)
    bg = ensure_rgb(Image.open(root / "backgrounds" / "vehicle.jpg"))
    ov = ensure_rgba(Image.open(root / "overlays" / f"MI_{plate_text}.png"))
    clean = bg.copy()
    clean.paste(ov, calculate_center_position(bg, ov), ov)
    clean.save(full / "plate.png")

    # no_plate: isolated perturbation layer flattened onto white
    layer = Image.open(ds / "perturbations" / f"{stem}.png").convert("RGBA")
    white = Image.new("RGB", layer.size, (255, 255, 255))
    white.paste(layer, (0, 0), layer)
    white.save(full / "no_plate.png")

    # merged: the real perturbed composite produced by the pipeline
    Image.open(ds / "images" / f"{stem}.png").convert("RGB").save(full / "merged.png")
    return full


def main() -> None:
    root = Path("dataset/plate_attack_demo")
    plate_text = "4J9T7W"
    build_inputs(root, plate_text)
    full = write_full_versions(root, plate_text)
    print(f"✅ Wrote {full / 'plate.png'}, {full / 'no_plate.png'}, {full / 'merged.png'}")


if __name__ == "__main__":
    main()

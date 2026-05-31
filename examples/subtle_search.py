#!/usr/bin/env python3
"""Search for the most subtle perturbation that still drives OCR scores down.

For several "subtle" recipes, generate variants, then measure two things per
image: visibility (mean absolute pixel change over the plate, % of 255) and OCR
defeat (the lowest score across all engines = the score even the strongest
engine gets). Reports recipes sorted by OCR defeat so you can pick the subtlest
one that still works.

Run with: uv run python examples/subtle_search.py
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from plateshapez.ocr import (
    EasyOCREngine,
    OCREngine,
    RapidOCREngine,
    TesseractEngine,
    character_score,
    crop_plate,
    normalize_plate,
)
from plateshapez.pipeline import DatasetGenerator
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position, ensure_rgb, ensure_rgba

PLATE = "Q0D1I5"
K = 10  # variants per recipe

# label -> perturbation list (kept deliberately subtle)
RECIPES: dict[str, list] = {
    "noise40": [{"name": "noise", "params": {"intensity": 40}}],
    "noise70": [{"name": "noise", "params": {"intensity": 70}}],
    "scratches+noise": [
        {"name": "texture", "params": {"type": "scratches", "intensity": 0.6}},
        {"name": "noise", "params": {"intensity": 25}},
    ],
    "grey-specks": [
        {
            "name": "shapes",
            "params": {
                "num_shapes": 24,
                "min_size": 4,
                "max_size": 14,
                "color": [110, 110, 110, 255],
            },
        },
        {"name": "noise", "params": {"intensity": 20}},
    ],
    "translucent-veil": [
        {
            "name": "shapes",
            "params": {
                "num_shapes": 12,
                "min_size": 20,
                "max_size": 70,
                "color": [235, 235, 235, 70],
            },
        },
        {"name": "noise", "params": {"intensity": 20}},
    ],
    "translucent-specks+scratch": [
        {
            "name": "shapes",
            "params": {
                "num_shapes": 20,
                "min_size": 4,
                "max_size": 16,
                "color": [235, 235, 235, 110],
            },
        },
        {"name": "texture", "params": {"type": "scratches", "intensity": 0.5}},
        {"name": "noise", "params": {"intensity": 18}},
    ],
}


def build_engines() -> dict[str, OCREngine]:
    engines: dict[str, OCREngine] = {"tesseract": TesseractEngine()}
    for name, ctor in (("easyocr", EasyOCREngine), ("rapidocr", RapidOCREngine)):
        try:
            engines[name] = ctor()
        except Exception as exc:  # pragma: no cover
            print(f"  (skipping {name}: {exc})")
    return engines


def main() -> None:
    from pathlib import Path

    root = Path("dataset/subtle_search")
    ind = root / "in"
    (ind / "backgrounds").mkdir(parents=True, exist_ok=True)
    (ind / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(ind / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    create_plate_overlay(PLATE).save(ind / "overlays" / f"{PLATE}.png")

    bg = ensure_rgb(Image.open(ind / "backgrounds" / "vehicle.jpg"))
    ov = ensure_rgba(Image.open(ind / "overlays" / f"{PLATE}.png"))
    bx, by = calculate_center_position(bg, ov)
    ow, oh = ov.size
    clean = bg.copy()
    clean.paste(ov, (bx, by), ov)
    clean_arr = np.array(clean).astype(np.int16)
    mask = np.zeros((bg.height, bg.width), bool)
    mask[by : by + oh, bx : bx + ow] = np.array(ov.split()[-1]) > 0

    engines = build_engines()
    expected = normalize_plate(PLATE)
    print(f"engines: {', '.join(engines)}\n")
    print(f"{'recipe':28} {'visibility%':>11} {'OCR worst (mean/min)':>22}")

    results = []
    for label, perts in RECIPES.items():
        out = root / label
        DatasetGenerator(
            bg_dir=ind / "backgrounds",
            overlay_dir=ind / "overlays",
            out_dir=out,
            perturbations=perts,
            random_seed=100,
            save_perturbation_layer=False,
        ).run(n_variants=K)

        vis_list, worst_list = [], []
        for label_path in sorted((out / "labels").glob("*.json")):
            meta = json.loads(label_path.read_text())
            img = Image.open(out / "images" / f"{label_path.stem}.png").convert("RGB")
            delta = np.abs(np.array(img).astype(np.int16) - clean_arr)[mask]
            vis_list.append(float(delta.mean()) / 255 * 100)
            crop = crop_plate(img, meta)
            scores = [
                character_score(expected, normalize_plate(e.read(crop))) for e in engines.values()
            ]
            worst_list.append(max(scores))
        vis = float(np.mean(vis_list))
        mean_worst, min_worst = float(np.mean(worst_list)), float(np.min(worst_list))
        results.append((label, vis, mean_worst, min_worst))
        print(f"{label:28} {vis:>10.2f} {np.mean(worst_list):>12.0f} /{np.min(worst_list):>4.0f}")

    results.sort(key=lambda r: (r[2], r[1]))  # lowest mean OCR, then most subtle
    best = results[0]
    print(f"\nBest OCR-defeat: '{best[0]}'  (visibility {best[1]:.2f}%, mean worst {best[2]:.0f})")
    subtle = sorted(results, key=lambda r: r[1])[0]
    print(f"Most subtle:     '{subtle[0]}'  (visibility {subtle[1]:.2f}%, worst {subtle[2]:.0f})")


if __name__ == "__main__":
    main()

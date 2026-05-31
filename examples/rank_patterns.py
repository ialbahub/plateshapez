#!/usr/bin/env python3
"""Rank perturbation patterns by how well they defeat OCR, keep the top N.

Generates many symbols+noise variants of one plate, reads each with every
available OCR engine (Tesseract, EasyOCR, RapidOCR/PP-OCR), scores how readable
the plate remains to the *best* engine, and keeps the patterns with the lowest
best-engine score (i.e. the strongest adversarial patterns). Builds a montage
of the top-N patterns and copies them to ``top_patterns/``.

Run with: uv run python examples/rank_patterns.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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

PLATE = "Q0D1I5"
N_VARIANTS = 120
TOP_N = 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def build_engines() -> dict[str, OCREngine]:
    engines: dict[str, OCREngine] = {"tesseract": TesseractEngine()}
    for name, ctor in (("easyocr", EasyOCREngine), ("rapidocr", RapidOCREngine)):
        try:
            engines[name] = ctor()
        except Exception as exc:  # pragma: no cover - optional backend
            print(f"  (skipping {name}: {exc})")
    return engines


def main() -> None:
    root = Path("dataset/pattern_rank")
    ind, ds = root / "in", root / "ds"
    (ind / "backgrounds").mkdir(parents=True, exist_ok=True)
    (ind / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(ind / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    create_plate_overlay(PLATE).save(ind / "overlays" / f"{PLATE}.png")

    DatasetGenerator(
        bg_dir=ind / "backgrounds",
        overlay_dir=ind / "overlays",
        out_dir=ds,
        perturbations=[
            {"name": "shapes", "params": {"num_shapes": 18, "min_size": 10, "max_size": 44}},
            {"name": "noise", "params": {"intensity": 32}},
        ],
        random_seed=7,
        confine_to_plate=True,
    ).run(n_variants=N_VARIANTS)

    engines = build_engines()
    expected = normalize_plate(PLATE)
    print(f"scoring {N_VARIANTS} patterns with: {', '.join(engines)}")

    scored = []
    for label_path in sorted((ds / "labels").glob("*.json")):
        import json

        meta = json.loads(label_path.read_text())
        img = Image.open(ds / "images" / f"{label_path.stem}.png").convert("RGB")
        crop = crop_plate(img, meta)
        per_engine = {
            name: character_score(expected, normalize_plate(eng.read(crop)))
            for name, eng in engines.items()
        }
        # "best" adversarial pattern = lowest score even for the strongest engine.
        worst_case = max(per_engine.values())
        scored.append((worst_case, per_engine, label_path.stem))

    scored.sort(key=lambda r: r[0])
    top = scored[:TOP_N]

    top_dir = root / "top_patterns"
    top_dir.mkdir(exist_ok=True)
    print(f"\nTop {TOP_N} patterns (lowest best-engine score = hardest for OCR):")
    for rank, (worst, per_engine, stem) in enumerate(top, 1):
        shutil.copy(ds / "perturbations" / f"{stem}.png", top_dir / f"rank{rank:02d}_{stem}.png")
        engs = " ".join(f"{k}={v:.0f}" for k, v in per_engine.items())
        print(f"  {rank:2d}. worst={worst:5.1f}  ({engs})")

    _montage(ds, top, root / "top20_patterns.png")
    print(f"\n✅ Montage: {root / 'top20_patterns.png'} | patterns copied to {top_dir}")


def _montage(ds: Path, top: list, path: Path) -> None:
    """Grid of the top patterns (perturbation layer over white) with scores."""
    cols, cell = 5, 230
    rows = (len(top) + cols - 1) // cols
    pad, header = 12, 30
    W = cols * (cell + pad) + pad
    H = rows * (cell // 2 + header + pad) + pad
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 13)
    for i, (worst, per_engine, stem) in enumerate(top):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell // 2 + header + pad)
        layer = Image.open(ds / "perturbations" / f"{stem}.png").convert("RGBA")
        tile = Image.new("RGB", layer.size, (255, 255, 255))
        tile.paste(layer, (0, 0), layer)
        tile = tile.resize((cell, cell // 2))
        sheet.paste(tile, (x, y + header))
        d.text((x, y), f"#{i + 1}  worst={worst:.0f}", fill=(150, 0, 0), font=f)
        d.text((x, y + 14), stem.replace("vehicle_", ""), fill=(60, 60, 60), font=f)
    sheet.save(path)


if __name__ == "__main__":
    main()

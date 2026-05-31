#!/usr/bin/env python3
"""Rank perturbation patterns by how well they defeat OCR — with the reads.

Generates symbols+noise variants for several plates, reads each composite with
every available OCR engine (Tesseract, EasyOCR, RapidOCR/PP-OCR), and records
*what each engine returned*. Ranks patterns by the lowest score even for the
strongest engine and keeps the top-N. Pattern/noise layers are saved as
genuinely transparent PNGs (opacity scales with perturbation strength).

Outputs under ``dataset/pattern_rank/``:
  reads.csv            - every image: expected + each engine's read + score
  top_patterns/*.png   - the top-N transparent pattern PNGs
  top_patterns.png     - montage: perturbed plate, pattern, and the OCR reads

Run with: uv run python examples/rank_patterns.py
"""

from __future__ import annotations

import csv
import json
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

PLATES = ["Q0D1I5", "B8G6S5"]
VARIANTS_PER_PLATE = 50
TOP_N = 20
ALPHA_GAIN = 5.0  # transparent pattern: opacity scales with perturbation strength
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def build_engines() -> dict[str, OCREngine]:
    engines: dict[str, OCREngine] = {"tesseract": TesseractEngine()}
    for name, ctor in (("easyocr", EasyOCREngine), ("rapidocr", RapidOCREngine)):
        try:
            engines[name] = ctor()
        except Exception as exc:  # pragma: no cover - optional backend
            print(f"  (skipping {name}: {exc})")
    return engines


def checker(size: tuple[int, int], s: int = 14) -> Image.Image:
    bg = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(bg)
    for yy in range(0, size[1], s):
        for xx in range(0, size[0], s):
            if (xx // s + yy // s) % 2:
                d.rectangle((xx, yy, xx + s, yy + s), fill=(205, 205, 205))
    return bg


def main() -> None:
    root = Path("dataset/pattern_rank")
    ind, ds = root / "in", root / "ds"
    (ind / "backgrounds").mkdir(parents=True, exist_ok=True)
    (ind / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(ind / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    for text in PLATES:
        create_plate_overlay(text).save(ind / "overlays" / f"{text}.png")

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
        perturbation_alpha_gain=ALPHA_GAIN,  # transparent pattern PNGs
    ).run(n_variants=VARIANTS_PER_PLATE)

    engines = build_engines()
    print(f"scoring {len(PLATES) * VARIANTS_PER_PLATE} patterns with: {', '.join(engines)}")

    scored = []
    rows_csv = []
    for label_path in sorted((ds / "labels").glob("*.json")):
        meta = json.loads(label_path.read_text())
        expected = normalize_plate(Path(meta["overlay"]).stem)
        img = Image.open(ds / "images" / f"{label_path.stem}.png").convert("RGB")
        crop = crop_plate(img, meta)
        reads, scores = {}, {}
        for name, eng in engines.items():
            pred = normalize_plate(eng.read(crop))
            reads[name] = pred
            scores[name] = character_score(expected, pred)
        worst = max(scores.values())  # strongest engine still gets this score
        scored.append((worst, reads, scores, expected, label_path.stem))
        rows_csv.append(
            {"image": label_path.stem, "expected": expected,
             **{f"read_{k}": v for k, v in reads.items()},
             **{f"score_{k}": round(scores[k], 1) for k in scores}}
        )

    with open(root / "reads.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_csv)

    scored.sort(key=lambda r: r[0])
    top = scored[:TOP_N]

    top_dir = root / "top_patterns"
    top_dir.mkdir(exist_ok=True)
    print(f"\nTop {TOP_N} patterns (lowest best-engine score; engine reads shown):")
    for rank, (worst, reads, scores, expected, stem) in enumerate(top, 1):
        shutil.copy(ds / "perturbations" / f"{stem}.png", top_dir / f"rank{rank:02d}_{stem}.png")
        reads_str = "  ".join(f"{k}:{reads[k] or '∅'}({scores[k]:.0f})" for k in reads)
        print(f"  {rank:2d}. {expected} worst={worst:4.0f}  {reads_str}")

    _montage(ds, top, root / "top_patterns.png")
    print(f"\n✅ reads.csv, transparent patterns in {top_dir}, montage {root / 'top_patterns.png'}")


def _montage(ds: Path, top: list, path: Path) -> None:
    """One row per top pattern: perturbed plate | transparent pattern | OCR reads."""
    thumb_w, row_h, pad, text_w = 200, 96, 10, 470
    W = pad * 4 + thumb_w * 2 + text_w
    H = pad + len(top) * (row_h + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 14)
    fm = ImageFont.truetype(MONO, 13)
    for i, (worst, reads, scores, expected, stem) in enumerate(top):
        y = pad + i * (row_h + pad)
        perturbed = Image.open(ds / "images" / f"{stem}.png").convert("RGB")
        perturbed = perturbed.resize((thumb_w, row_h))
        sheet.paste(perturbed, (pad, y))
        layer = Image.open(ds / "perturbations" / f"{stem}.png").convert("RGBA")
        tile = checker(layer.size)
        tile.paste(layer, (0, 0), layer)
        sheet.paste(tile.resize((thumb_w, row_h)), (pad * 2 + thumb_w, y))
        tx = pad * 3 + thumb_w * 2
        head = f"#{i + 1}  expected {expected} (worst {worst:.0f})"
        d.text((tx, y), head, fill=(150, 0, 0), font=f)
        for j, k in enumerate(reads):
            d.text((tx, y + 22 + j * 20),
                   f"{k:9}: {reads[k] or '(nothing)':9}  score {scores[k]:5.1f}",
                   fill=(20, 20, 20), font=fm)
    sheet.save(path)


if __name__ == "__main__":
    main()

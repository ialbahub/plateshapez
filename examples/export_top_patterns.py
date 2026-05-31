#!/usr/bin/env python3
"""Export the top-N OCR-defeating patterns for a plate at full print DPI.

Generates a pool of symbols+noise variants at ``PPI`` pixels/inch (true 12x6 in,
2:1), scores each with every available OCR engine, ranks by the lowest score
even for the strongest engine, and exports the top-N as full-DPI files:
the transparent pattern PNG and the perturbed plate (both 12x6 in @ PPI, with
DPI metadata). Also writes a preview montage and zips the exports.

Run with: uv run python examples/export_top_patterns.py
"""

from __future__ import annotations

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

PLATE = "Q0D1I5"
PPI = 300
POOL = 40
TOP_N = 20
ALPHA_GAIN = 5.0
OCR_WIDTH = 720  # downscale plate crop for fast, consistent OCR scoring
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def build_engines() -> dict[str, OCREngine]:
    engines: dict[str, OCREngine] = {"tesseract": TesseractEngine()}
    for name, ctor in (("easyocr", EasyOCREngine), ("rapidocr", RapidOCREngine)):
        try:
            engines[name] = ctor()
        except Exception as exc:  # pragma: no cover
            print(f"  (skipping {name}: {exc})")
    return engines


def main() -> None:
    w, h = 12 * PPI, 6 * PPI
    scale = PPI / 100  # vs the 100-ppi reference sizes
    root = Path("dataset/top20_300dpi")
    ind, ds = root / "in", root / "ds"
    (ind / "backgrounds").mkdir(parents=True, exist_ok=True)
    (ind / "overlays").mkdir(parents=True, exist_ok=True)

    create_vehicle_background(size=(w + 200, h + 200)).save(
        ind / "backgrounds" / "vehicle.jpg", "JPEG", quality=95
    )
    create_plate_overlay(PLATE, size=(w, h)).save(ind / "overlays" / f"{PLATE}.png")

    DatasetGenerator(
        bg_dir=ind / "backgrounds",
        overlay_dir=ind / "overlays",
        out_dir=ds,
        perturbations=[
            {
                "name": "shapes",
                "params": {
                    "num_shapes": 18,
                    "min_size": int(10 * scale),
                    "max_size": int(44 * scale),
                },
            },
            {"name": "noise", "params": {"intensity": 32}},
        ],
        random_seed=7,
        confine_to_plate=True,
        perturbation_alpha_gain=ALPHA_GAIN,
    ).run(n_variants=POOL)

    engines = build_engines()
    expected = normalize_plate(PLATE)
    print(f"scoring {POOL} hi-res patterns with: {', '.join(engines)}")

    scored = []
    for label_path in sorted((ds / "labels").glob("*.json")):
        meta = json.loads(label_path.read_text())
        img = Image.open(ds / "images" / f"{label_path.stem}.png").convert("RGB")
        crop = crop_plate(img, meta)
        crop = crop.resize((OCR_WIDTH, int(OCR_WIDTH * crop.height / crop.width)))
        reads = {n: normalize_plate(e.read(crop)) for n, e in engines.items()}
        scores = {n: character_score(expected, reads[n]) for n in reads}
        scored.append((max(scores.values()), reads, scores, label_path.stem, meta))

    scored.sort(key=lambda r: r[0])
    top = scored[:TOP_N]

    pat_dir = root / "top_patterns"
    plate_dir = root / "top_perturbed"
    for d in (pat_dir, plate_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    print(f"\nTop {TOP_N} ({PLATE}) at {PPI} PPI ({w}x{h}); engine reads:")
    for rank, (worst, reads, scores, stem, meta) in enumerate(top, 1):
        bx, by = meta["overlay_position"]
        ow, oh = meta["overlay_size"]
        box = (bx, by, bx + ow, by + oh)
        layer = Image.open(ds / "perturbations" / f"{stem}.png").convert("RGBA").crop(box)
        plate = Image.open(ds / "images" / f"{stem}.png").convert("RGB").crop(box)
        layer.save(pat_dir / f"rank{rank:02d}_pattern.png", dpi=(PPI, PPI))
        plate.save(plate_dir / f"rank{rank:02d}_perturbed.png", dpi=(PPI, PPI))
        reads_str = "  ".join(f"{k}:{reads[k] or '∅'}({scores[k]:.0f})" for k in reads)
        print(f"  {rank:2d}. worst={worst:4.0f}  {reads_str}")

    _montage(top, ds, root / "top20_preview.png")
    shutil.make_archive(str(root / "top20_patterns_300dpi"), "zip", pat_dir)
    shutil.make_archive(str(root / "top20_perturbed_300dpi"), "zip", plate_dir)
    print(f"\n✅ {pat_dir} + {plate_dir} (full DPI), zips + {root / 'top20_preview.png'}")


def _montage(top: list, ds: Path, path: Path) -> None:
    cols, tw, th, pad, hdr = 4, 260, 130, 12, 36
    rows = (len(top) + cols - 1) // cols
    W = cols * (tw + pad) + pad
    H = rows * (th + hdr + pad) + pad
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 13)
    fm = ImageFont.truetype(MONO, 11)
    for i, (worst, reads, scores, stem, meta) in enumerate(top):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + hdr + pad)
        plate = Image.open(ds / "images" / f"{stem}.png").convert("RGB")
        bx, by = meta["overlay_position"]
        ow, oh = meta["overlay_size"]
        plate = plate.crop((bx, by, bx + ow, by + oh)).resize((tw, th))
        sheet.paste(plate, (x, y + hdr))
        d.text((x, y), f"#{i + 1}  worst={worst:.0f}", fill=(150, 0, 0), font=f)
        reads_str = " ".join(f"{k[:4]}:{reads[k] or '-'}" for k in reads)
        d.text((x, y + 18), reads_str[:46], fill=(40, 40, 40), font=fm)
    sheet.save(path)


if __name__ == "__main__":
    main()

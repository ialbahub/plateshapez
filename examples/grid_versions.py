#!/usr/bin/env python3
"""Generate configurable grid-attack versions with clean/pattern/merged + scores.

For each (fully configurable) version, writes three images — the clean plate,
the generated pattern alone (no plate, transparent), and the merged result —
plus the scores from every engine (Fast-ALPR is the real ALPR; Tesseract/
EasyOCR/RapidOCR are recognition checks on a perfect crop). Builds an overview
montage and a zip. Add/edit entries in VERSIONS to explore more.

Run with: uv run python examples/grid_versions.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from plateshapez.ocr import (
    EasyOCREngine,
    FastALPREngine,
    OCREngine,
    RapidOCREngine,
    TesseractEngine,
    character_score,
    normalize_plate,
)
from plateshapez.perturbations.grid import GridPerturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position

PLATE = "Q0D1I5"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Every parameter is configurable; add rows to explore more versions.
VERSIONS: list[dict] = [
    {"name": "fine_faint", "grid": dict(spacing=22, width=2, alpha=110)},
    {"name": "fine_strong", "grid": dict(spacing=18, width=3, alpha=160)},
    {"name": "dense", "grid": dict(spacing=12, width=2, alpha=140)},
    {"name": "diagonal", "grid": dict(spacing=20, width=2, alpha=140, diagonal=True)},
    {"name": "dark_mesh", "grid": dict(spacing=18, width=3, alpha=170, color=[10, 10, 10])},
    {"name": "grid_plus_noise", "grid": dict(spacing=18, width=2, alpha=140), "noise": 45},
]


def main() -> None:
    root = Path("dataset/grid_versions")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bg = create_vehicle_background(size=(1600, 1100))
    plate = create_plate_overlay(PLATE, size=(1200, 600))
    pos = calculate_center_position(bg, plate)
    x, y = pos
    w, h = plate.size
    region = (x, y, w, h)
    box = (x, y, x + w, y + h)
    clean = bg.copy()
    clean.paste(plate, pos, plate)
    clean.crop(box).save(root / "clean.png")

    alpr = FastALPREngine()
    ocr: dict[str, OCREngine] = {
        "tesseract": TesseractEngine(),
        "easyocr": EasyOCREngine(),
        "rapidocr": RapidOCREngine(),
    }
    expected = normalize_plate(PLATE)

    def band(img: Image.Image) -> Image.Image:
        return img.crop((x, y + int(h * 0.30), x + w, y + int(h * 0.72)))

    summary = []
    for v in VERSIONS:
        d = root / v["name"]
        d.mkdir()
        grid = GridPerturbation(**v["grid"])

        # pattern alone (no plate, transparent)
        transparent = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        grid.apply(transparent, region).crop(box).save(d / "pattern.png")

        # merged = clean plate + grid (+ optional noise)
        merged = clean.copy()
        if v.get("noise"):
            merged = NoisePerturbation(intensity=v["noise"]).apply(merged, region)
        merged = grid.apply(merged, region)
        merged.crop(box).save(d / "merged.png")
        shutil.copy(root / "clean.png", d / "clean.png")

        # scores: Fast-ALPR on full scene; OCR on the perfect band crop
        full = bg.copy()
        full.paste(plate, pos, plate)
        if v.get("noise"):
            full = NoisePerturbation(intensity=v["noise"]).apply(full, region)
        full = grid.apply(full, region)
        fa = normalize_plate(alpr.read(full))
        reads = {"fast_alpr": fa}
        reads.update({n: normalize_plate(e.read(band(full))) for n, e in ocr.items()})
        scores = {n: (character_score(expected, r) if r else 0.0) for n, r in reads.items()}
        (d / "score.json").write_text(json.dumps({"reads": reads, "scores": scores}, indent=2))
        summary.append((v["name"], reads, scores))
        line = "  ".join(f"{n}:{reads[n] or 'NONE'}({scores[n]:.0f})" for n in reads)
        print(f"{v['name']:16} {line}")

    _montage(root, summary)
    shutil.make_archive(str(root / "grid_versions"), "zip", root)
    print(f"\n✅ versions (clean/pattern/merged/score) in {root}; zip {root / 'grid_versions.zip'}")


def _montage(root: Path, summary: list) -> None:
    cw, ch, pad, hdr = 300, 150, 14, 56
    cols = 3  # clean | pattern | merged
    rows = len(summary)
    W = pad + cols * (cw + pad)
    H = pad + rows * (ch + hdr + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 14)
    fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)

    def chk(im: Image.Image) -> Image.Image:
        bgc = Image.new("RGB", im.size, (255, 255, 255))
        bgc.paste(im, (0, 0), im if im.mode == "RGBA" else None)
        return bgc

    for i, (name, reads, scores) in enumerate(summary):
        y = pad + i * (ch + hdr + pad)
        d.text((pad, y), name, fill=(120, 0, 120), font=f)
        line = "  ".join(f"{n[:4]}:{reads[n] or 'NONE'}" for n in reads)
        d.text((pad, y + 20), line[:90], fill=(20, 20, 20), font=fm)
        for j, fname in enumerate(("clean.png", "pattern.png", "merged.png")):
            im = Image.open(root / name / fname).convert("RGBA").resize((cw, ch))
            sheet.paste(chk(im), (pad + j * (cw + pad), y + hdr))
    sheet.save(root / "overview.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble the best-performing perturbation versions from each technique.

For one plate, builds the strongest example of each approach we validated:
  v1_strong_occlusion  - white blobs (lowest OCR, most visible)
  v2_subtle_optimized  - optimizer winner (grey specks + scratch + noise)
  v3_perspective       - viewing-angle warp (kills OCR, stays human-readable)
  v4_translucent       - faint translucent specks (subtlest)
Each version is saved at true plate size (1200x600, 2:1) as the perturbed plate
plus a transparent pattern PNG, labelled with its 3-engine OCR reads. Writes a
montage and a zip.

Run with: uv run python examples/best_versions.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plateshapez.ocr import (
    EasyOCREngine,
    OCREngine,
    RapidOCREngine,
    TesseractEngine,
    character_score,
    normalize_plate,
)
from plateshapez.perturbations.base import Perturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.perturbations.perspective import PerspectivePerturbation
from plateshapez.perturbations.shapes import ShapesPerturbation
from plateshapez.perturbations.texture import TexturePerturbation
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position, isolate_neutral_layer

PLATE = "Q0D1I5"
NEUTRAL = 128
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# name -> (list of (PerturbationClass, kwargs), perspective kwargs or None, alpha_gain)
Version = tuple[list[tuple[type[Perturbation], dict]], dict | None, float | None]
VERSIONS: dict[str, Version] = {
    "v1_strong_occlusion": (
        [(ShapesPerturbation, dict(num_shapes=18, min_size=10, max_size=44, color="white")),
         (NoisePerturbation, dict(intensity=32))],
        None, None),
    "v2_subtle_optimized": (
        [(ShapesPerturbation, dict(num_shapes=18, min_size=4, max_size=40,
                                   color=[100, 100, 100, 200])),
         (TexturePerturbation, dict(type="scratches", intensity=0.5)),
         (NoisePerturbation, dict(intensity=60))],
        None, 5.0),
    "v3_perspective": (
        [(ShapesPerturbation, dict(num_shapes=10, min_size=4, max_size=16,
                                   color=[235, 235, 235, 120])),
         (NoisePerturbation, dict(intensity=22))],
        dict(strength=0.26, tilt="h"), 5.0),
    "v4_translucent": (
        [(ShapesPerturbation, dict(num_shapes=20, min_size=4, max_size=16,
                                   color=[235, 235, 235, 110])),
         (TexturePerturbation, dict(type="scratches", intensity=0.5)),
         (NoisePerturbation, dict(intensity=18))],
        None, 5.0),
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
    root = Path("dataset/best_versions")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bg = create_vehicle_background()
    plate = create_plate_overlay(PLATE)
    pos = calculate_center_position(bg, plate)
    ow, oh = plate.size
    box = (pos[0], pos[1], pos[0] + ow, pos[1] + oh)
    region = (pos[0], pos[1], ow, oh)
    mask = np.zeros((bg.height, bg.width), bool)
    mask[pos[1] : pos[1] + oh, pos[0] : pos[0] + ow] = np.array(plate.split()[-1]) > 0

    clean = bg.copy()
    clean.paste(plate, pos, plate)
    clean.crop(box).save(root / "clean.png")

    engines = build_engines()
    expected = normalize_plate(PLATE)
    band = (0.30, 0.72)

    def ocr_reads(img_full: Image.Image) -> dict[str, str]:
        crop = img_full.crop((pos[0], pos[1] + int(oh * band[0]),
                              pos[0] + ow, pos[1] + int(oh * band[1])))
        return {n: normalize_plate(e.read(crop)) for n, e in engines.items()}

    summary = []
    for name, (perts, persp, again) in VERSIONS.items():
        vdir = root / name
        vdir.mkdir()
        random.seed(42)
        np.random.seed(42)
        comp = bg.copy()
        comp.paste(plate, pos, plate)
        base = comp.copy()
        for cls, kw in perts:
            comp = cls(**kw).apply(comp, region)
        arr, barr = np.array(comp), np.array(base)
        arr[~mask] = barr[~mask]
        comp = Image.fromarray(arr)
        if persp is not None:
            comp = PerspectivePerturbation(**persp).apply(comp, region)
        comp.crop(box).save(vdir / "perturbed.png")

        # transparent pattern (drawn perturbations only, neutral replay)
        random.seed(42)
        np.random.seed(42)
        neutral = Image.new("RGB", bg.size, (NEUTRAL,) * 3)
        for cls, kw in perts:
            neutral = cls(**kw).apply(neutral, region)
        narr = np.array(neutral)
        narr[~mask] = NEUTRAL
        isolate_neutral_layer(Image.fromarray(narr), NEUTRAL, alpha_gain=again).crop(box).save(
            vdir / "pattern.png"
        )

        reads = ocr_reads(comp)
        scores = {n: character_score(expected, reads[n]) for n in reads}
        summary.append((name, reads, scores, max(scores.values())))
        print(f"{name:22} worst={max(scores.values()):4.0f}  "
              + "  ".join(f"{n}:{reads[n] or '_'}({scores[n]:.0f})" for n in reads))

    _montage(root, summary)
    # Write the archive outside the directory being zipped (avoid self-inclusion).
    shutil.make_archive(str(root.parent / "best_versions"), "zip", root)
    print(f"\n✅ {root} + {root.parent / 'best_versions.zip'}")


def _montage(root: Path, summary: list) -> None:
    tw, th, pad = 360, 180, 12
    rows = len(summary)
    W = pad * 3 + tw * 2
    H = pad + rows * (th + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 15)
    fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    for i, (name, reads, scores, worst) in enumerate(summary):
        y = pad + i * (th + pad)
        perturbed = Image.open(root / name / "perturbed.png").convert("RGB").resize((tw, th))
        sheet.paste(perturbed, (pad, y))
        tx = pad * 2 + tw
        d.text((tx, y), f"{name}  (worst {worst:.0f})", fill=(150, 0, 0), font=f)
        for j, n in enumerate(reads):
            d.text((tx, y + 24 + j * 20), f"{n:9}: {reads[n] or '(none)':9} {scores[n]:5.0f}",
                   fill=(20, 20, 20), font=fm)
    sheet.save(root / "overview.png")


if __name__ == "__main__":
    main()

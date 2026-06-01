#!/usr/bin/env python3
"""Scan 2000 perturbation examples (any colour/technique), tier them, keep top 5.

Randomly samples across every technique (shapes of any colour/size/opacity,
noise, scratches, glare, blur, perspective, and combinations), scores each by
how low it drives OCR (Tesseract + RapidOCR; lower = better defeat), sorts the
results into tiers, and keeps the 5 that beat OCR hardest. The top 5 are
re-rendered at plate size with the perturbed plate, a transparent NO-PLATE
pattern, and validated with EasyOCR. Writes a montage + zip.

Run with: uv run python examples/scan_2000.py
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
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
from plateshapez.perturbations.blur import BlurPerturbation
from plateshapez.perturbations.glare import GlarePerturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.perturbations.perspective import PerspectivePerturbation
from plateshapez.perturbations.shapes import ShapesPerturbation
from plateshapez.perturbations.texture import TexturePerturbation
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position, isolate_neutral_layer

PLATE = "Q0D1I5"
N = 2000
SCAN_W, SCAN_H = 600, 300  # small plate for a fast scan
NEUTRAL = 128
BAND = (0.30, 0.72)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

COLORS = [
    [0, 0, 0, 255], [255, 255, 255, 255], [128, 128, 128, 255], [90, 90, 90, 255],
    [210, 40, 40, 255], [40, 110, 210, 255], [225, 205, 40, 255], [40, 160, 90, 255],
    [235, 235, 235, 120], [235, 235, 235, 70], [20, 20, 20, 140],
]


@dataclass
class Cfg:
    perts: list[tuple[type[Perturbation], dict]] = field(default_factory=list)
    persp: dict | None = None
    label: str = ""


def sample_cfg(rng: random.Random, sw: int, sh: int) -> Cfg:
    techniques = rng.sample(
        ["shapes", "noise", "scratch", "glare", "blur"],
        rng.randint(1, 3),
    )
    perts: list[tuple[type[Perturbation], dict]] = []
    used = []
    if "shapes" in techniques:
        color = rng.choice(COLORS)
        if rng.random() < 0.2:  # also try fully random colours
            color = [rng.randint(0, 255) for _ in range(3)] + [rng.choice([120, 200, 255])]
        smax = int(sh * rng.choice([0.05, 0.1, 0.18, 0.3]))
        perts.append((ShapesPerturbation, dict(
            num_shapes=rng.choice([8, 14, 20, 30]), min_size=2, max_size=max(6, smax),
            color=color)))
        used.append(f"shapes(c{color[:3]}a{color[3]},m{smax})")
    if "noise" in techniques:
        n = rng.choice([20, 35, 55, 80])
        perts.append((NoisePerturbation, dict(intensity=n)))
        used.append(f"noise{n}")
    if "scratch" in techniques:
        s = rng.choice([0.3, 0.5, 0.7])
        perts.append((TexturePerturbation, dict(type="scratches", intensity=s)))
        used.append(f"scratch{s}")
    if "glare" in techniques:
        g = rng.choice([0.5, 0.7, 0.9])
        perts.append((GlarePerturbation, dict(intensity=g, spread=0.45)))
        used.append(f"glare{g}")
    if "blur" in techniques:
        b = rng.choice([2, 3, 5])
        perts.append((BlurPerturbation, dict(radius=b)))
        used.append(f"blur{b}")
    persp = None
    if rng.random() < 0.4:
        st = rng.choice([0.12, 0.2, 0.28])
        persp = dict(strength=st, tilt=rng.choice(["h", "v"]))
        used.append(f"persp{st}{persp['tilt']}")
    return Cfg(perts=perts, persp=persp, label=" + ".join(used))


def build(
    bg: Image.Image,
    plate: Image.Image,
    pos: tuple[int, int],
    mask: np.ndarray,
    cfg: Cfg,
) -> Image.Image:
    ow, oh = plate.size
    region = (pos[0], pos[1], ow, oh)
    comp = bg.copy()
    comp.paste(plate, pos, plate)
    base = comp.copy()
    for cls, kw in cfg.perts:
        comp = cls(**kw).apply(comp, region)
    arr, barr = np.array(comp), np.array(base)
    arr[~mask] = barr[~mask]
    comp = Image.fromarray(arr)
    if cfg.persp is not None:
        comp = PerspectivePerturbation(**cfg.persp).apply(comp, region)
    return comp


def pattern(
    bg_size: tuple[int, int],
    plate: Image.Image,
    pos: tuple[int, int],
    mask: np.ndarray,
    cfg: Cfg,
) -> Image.Image:
    ow, oh = plate.size
    region = (pos[0], pos[1], ow, oh)
    neutral = Image.new("RGB", bg_size, (NEUTRAL,) * 3)
    for cls, kw in cfg.perts:
        neutral = cls(**kw).apply(neutral, region)
    narr = np.array(neutral)
    narr[~mask] = NEUTRAL
    return isolate_neutral_layer(Image.fromarray(narr), NEUTRAL, alpha_gain=5.0)


def make_scene(
    w: int, h: int
) -> tuple[Image.Image, Image.Image, tuple[int, int], np.ndarray]:
    bg = create_vehicle_background(size=(int(w * 1.33), int(h * 1.4)))
    plate = create_plate_overlay(PLATE, size=(w, h))
    pos = calculate_center_position(bg, plate)
    ow, oh = plate.size
    mask = np.zeros((bg.height, bg.width), bool)
    mask[pos[1]:pos[1] + oh, pos[0]:pos[0] + ow] = np.array(plate.split()[-1]) > 0
    return bg, plate, pos, mask


def band_crop(img: Image.Image, plate: Image.Image, pos: tuple[int, int]) -> Image.Image:
    ow, oh = plate.size
    return img.crop((pos[0], pos[1] + int(oh * BAND[0]), pos[0] + ow, pos[1] + int(oh * BAND[1])))


def main() -> None:
    root = Path("dataset/scan2000")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bg, plate, pos, mask = make_scene(SCAN_W, SCAN_H)
    expected = normalize_plate(PLATE)
    fast: dict[str, OCREngine] = {"tesseract": TesseractEngine(), "rapidocr": RapidOCREngine()}

    rng = random.Random(0)
    scored = []
    for i in range(N):
        cfg = sample_cfg(rng, SCAN_W, SCAN_H)
        comp = build(bg, plate, pos, mask, cfg)
        crop = band_crop(comp, plate, pos)
        sc = {n: character_score(expected, normalize_plate(e.read(crop))) for n, e in fast.items()}
        scored.append((max(sc.values()), sc, cfg))
        if (i + 1) % 250 == 0:
            print(f"  scanned {i + 1}/{N}")

    scored.sort(key=lambda r: r[0])
    tiers = {"S (<25)": 0, "A (25-44)": 0, "B (45-64)": 0, "C (65-84)": 0, "D (85+)": 0}
    for worst, _, _ in scored:
        key = ("S (<25)" if worst < 25 else "A (25-44)" if worst < 45 else
               "B (45-64)" if worst < 65 else "C (65-84)" if worst < 85 else "D (85+)")
        tiers[key] += 1
    print("\nTiers (by best-engine OCR score; lower = better defeat):")
    for k, v in tiers.items():
        print(f"  {k:12} {v:5d}  ({v / N:.0%})")

    top = scored[:5]
    print("\nTop 5 (validating with EasyOCR):")
    try:
        easy: OCREngine | None = EasyOCREngine()
    except Exception:
        easy = None
    engines = dict(fast) if easy is None else {**fast, "easyocr": easy}

    big_bg, big_plate, big_pos, big_mask = make_scene(1200, 600)
    box = (big_pos[0], big_pos[1], big_pos[0] + 1200, big_pos[1] + 600)
    rows = []
    for rank, (worst, sc, cfg) in enumerate(top, 1):
        d = root / f"rank{rank}"
        d.mkdir()
        comp = build(big_bg, big_plate, big_pos, big_mask, cfg)
        comp.crop(box).save(d / "perturbed.png")
        pattern(big_bg.size, big_plate, big_pos, big_mask, cfg).crop(box).save(d / "pattern.png")
        crop = band_crop(comp, big_plate, big_pos)
        reads = {n: normalize_plate(e.read(crop)) for n, e in engines.items()}
        scores = {n: character_score(expected, reads[n]) for n in reads}
        rows.append((rank, cfg.label, reads, scores, max(scores.values())))
        print(f"  #{rank} worst={max(scores.values()):4.0f}  {cfg.label}")
        print("       " + "  ".join(f"{n}:{reads[n] or '_'}({scores[n]:.0f})" for n in reads))

    big_plate_clean = big_bg.copy()
    big_plate_clean.paste(big_plate, big_pos, big_plate)
    big_plate_clean.crop(box).save(root / "clean.png")
    _montage(root, rows)
    shutil.make_archive(str(root / "top5"), "zip", root)
    print(f"\n✅ tiers above; top-5 (perturbed + no-plate pattern) in {root}")
    print(f"   zip: {root / 'top5.zip'}")


def _montage(root: Path, rows: list) -> None:
    tw, th, pad = 320, 160, 12
    W = pad * 3 + tw * 2
    H = pad + len(rows) * (th + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 13)
    fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    for i, (rank, label, reads, scores, worst) in enumerate(rows):
        y = pad + i * (th + pad)
        img = Image.open(root / f"rank{rank}" / "perturbed.png").convert("RGB").resize((tw, th))
        sheet.paste(img, (pad, y))
        tx = pad * 2 + tw
        d.text((tx, y), f"#{rank}  worst {worst:.0f}", fill=(150, 0, 0), font=f)
        d.text((tx, y + 20), label[:52], fill=(80, 0, 80), font=fm)
        for j, n in enumerate(reads):
            d.text((tx, y + 40 + j * 18), f"{n:9}:{reads[n] or '_':8} {scores[n]:4.0f}",
                   fill=(20, 20, 20), font=fm)
    sheet.save(root / "overview.png")


if __name__ == "__main__":
    main()

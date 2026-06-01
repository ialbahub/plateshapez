#!/usr/bin/env python3
"""Scan 2000 examples against the REAL ALPR (Fast-ALPR) and tier them.

This is the test that matters: Fast-ALPR (YOLO detector + OCR) is the engine the
ALPRovingGround suite uses, and it includes plate **detection**. We sample 2000
perturbations across every technique and colour (plus perspective up to steep
angles), score each by what Fast-ALPR returns on the full scene (empty = the
detector found no plate = a full defeat), tier the results, and keep the 5 that
beat it with the least perturbation. Top 5 get a perturbed plate + a transparent
no-plate pattern. Writes a montage + zip.

Needs the ocr-fastalpr extra. Run with: uv run python examples/scan_alpr.py
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plateshapez.ocr import FastALPREngine, character_score, normalize_plate
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
NEUTRAL = 128
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = [
    [0, 0, 0, 255],
    [255, 255, 255, 255],
    [128, 128, 128, 255],
    [90, 90, 90, 255],
    [210, 40, 40, 255],
    [40, 110, 210, 255],
    [225, 205, 40, 255],
    [235, 235, 235, 120],
    [20, 20, 20, 140],
]


@dataclass
class Cfg:
    perts: list[tuple[type[Perturbation], dict]] = field(default_factory=list)
    persp: dict | None = None
    label: str = ""
    vis: float = 0.0


def sample_cfg(rng: random.Random) -> Cfg:
    perts: list[tuple[type[Perturbation], dict]] = []
    used = []
    for tech in rng.sample(["shapes", "noise", "scratch", "glare", "blur"], rng.randint(0, 2)):
        if tech == "shapes":
            color = rng.choice(COLORS)
            perts.append(
                (
                    ShapesPerturbation,
                    dict(
                        num_shapes=rng.choice([10, 20, 40]),
                        min_size=4,
                        max_size=rng.choice([20, 44, 80]),
                        color=color,
                    ),
                )
            )
            used.append(f"shapes{color[:3]}a{color[3]}")
        elif tech == "noise":
            n = rng.choice([35, 60, 90])
            perts.append((NoisePerturbation, dict(intensity=n)))
            used.append(f"noise{n}")
        elif tech == "scratch":
            s = rng.choice([0.5, 0.7])
            perts.append((TexturePerturbation, dict(type="scratches", intensity=s)))
            used.append(f"scratch{s}")
        elif tech == "glare":
            g = rng.choice([0.7, 0.95])
            perts.append((GlarePerturbation, dict(intensity=g, spread=0.5)))
            used.append(f"glare{g}")
        elif tech == "blur":
            b = rng.choice([3, 8, 15, 25])
            perts.append((BlurPerturbation, dict(radius=b)))
            used.append(f"blur{b}")
    persp = None
    if rng.random() < 0.55:
        st = rng.choice([0.18, 0.24, 0.28, 0.32, 0.38, 0.44])
        persp = dict(strength=st, tilt=rng.choice(["h", "v"]))
        used.append(f"persp{st}{persp['tilt']}")
    if not perts and persp is None:
        persp = dict(strength=0.3, tilt="v")
        used.append("persp0.3v")
    return Cfg(perts=perts, persp=persp, label=" + ".join(used) or "clean")


def make_scene(
    w: int, h: int
) -> tuple[Image.Image, Image.Image, tuple[int, int], np.ndarray]:
    bg = create_vehicle_background(size=(int(w * 1.33), int(h * 1.4)))
    plate = create_plate_overlay(PLATE, size=(w, h))
    pos = calculate_center_position(bg, plate)
    ow, oh = plate.size
    mask = np.zeros((bg.height, bg.width), bool)
    mask[pos[1] : pos[1] + oh, pos[0] : pos[0] + ow] = np.array(plate.split()[-1]) > 0
    return bg, plate, pos, mask


def build(
    bg: Image.Image,
    plate: Image.Image,
    pos: tuple[int, int],
    mask: np.ndarray,
    cfg: Cfg,
    clean_arr: np.ndarray | None = None,
) -> Image.Image:
    ow, oh = plate.size
    region = (pos[0], pos[1], ow, oh)
    comp = bg.copy()
    comp.paste(plate, pos, plate)
    base = comp.copy()
    for cls, kw in cfg.perts:
        comp = cls(**kw).apply(comp, region)
    arr = np.array(comp)
    arr[~mask] = np.array(base)[~mask]
    comp = Image.fromarray(arr)
    if clean_arr is not None:
        cfg.vis = float(np.abs(arr.astype(np.int16) - clean_arr)[mask].mean()) / 255 * 100
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


def subtlety(cfg: Cfg) -> float:
    """Lower = subtler: penalise perspective angle and surface visibility."""
    angle = cfg.persp["strength"] if cfg.persp else 0.0
    return angle * 100 + cfg.vis


def main() -> None:
    root = Path("dataset/scan_alpr")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bg, plate, pos, mask = make_scene(900, 450)
    clean = bg.copy()
    clean.paste(plate, pos, plate)
    clean_arr = np.array(clean).astype(np.int16)
    expected = normalize_plate(PLATE)
    alpr = FastALPREngine()

    rng = random.Random(0)
    scored = []
    for i in range(N):
        cfg = sample_cfg(rng)
        comp = build(bg, plate, pos, mask, cfg, clean_arr)
        read = normalize_plate(alpr.read(comp))
        score = character_score(expected, read) if read else 0.0
        scored.append((score, read, cfg))
        if (i + 1) % 250 == 0:
            print(f"  scanned {i + 1}/{N}")

    defeated = [s for s in scored if s[0] == 0.0]
    tiers = {
        "defeated (not read)": len(defeated),
        "degraded (<60)": sum(1 for s in scored if 0 < s[0] < 60),
        "partial (60-84)": sum(1 for s in scored if 60 <= s[0] < 85),
        "read (85+)": sum(1 for s in scored if s[0] >= 85),
    }
    print("\nTiers vs Fast-ALPR (the real ALPR):")
    for k, v in tiers.items():
        print(f"  {k:22} {v:5d}  ({v / N:.0%})")

    # Top 5 = defeats Fast-ALPR with the least perturbation (subtlest).
    defeated.sort(key=lambda s: subtlety(s[2]))
    top = defeated[:5] if defeated else sorted(scored, key=lambda s: s[0])[:5]

    big_bg, big_plate, big_pos, big_mask = make_scene(1200, 600)
    box = (big_pos[0], big_pos[1], big_pos[0] + 1200, big_pos[1] + 600)
    rows = []
    print("\nTop 5 (defeat Fast-ALPR with least perturbation):")
    for rank, (score, read, cfg) in enumerate(top, 1):
        d = root / f"rank{rank}"
        d.mkdir()
        comp = build(big_bg, big_plate, big_pos, big_mask, cfg)
        comp.crop(box).save(d / "perturbed.png")
        pattern(big_bg.size, big_plate, big_pos, big_mask, cfg).crop(box).save(d / "pattern.png")
        rows.append((rank, cfg.label, score, read))
        print(f"  #{rank} fast_alpr_score={score:.0f} read={read or '(none)'!r}  {cfg.label}")

    clean.crop((pos[0], pos[1], pos[0] + 900, pos[1] + 450)).save(root / "clean.png")
    _montage(root, rows)
    shutil.make_archive(str(root / "top5_alpr"), "zip", root)
    print(f"\n✅ tiers above; top-5 in {root}; zip {root / 'top5_alpr.zip'}")


def _montage(root: Path, rows: list) -> None:
    tw, th, pad = 360, 180, 12
    sheet = Image.new("RGB", (pad * 3 + tw * 2, pad + len(rows) * (th + pad)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 14)
    fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    for i, (rank, label, score, read) in enumerate(rows):
        y = pad + i * (th + pad)
        img = Image.open(root / f"rank{rank}" / "perturbed.png").convert("RGB").resize((tw, th))
        sheet.paste(img, (pad, y))
        tx = pad * 2 + tw
        d.text((tx, y), f"#{rank}  Fast-ALPR: {read or 'NOT DETECTED'}", fill=(150, 0, 0), font=f)
        d.text((tx, y + 26), f"score {score:.0f}", fill=(20, 20, 20), font=fm)
        d.text((tx, y + 46), label[:54], fill=(80, 0, 80), font=fm)
    sheet.save(root / "overview.png")


if __name__ == "__main__":
    main()

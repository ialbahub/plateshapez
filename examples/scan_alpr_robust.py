#!/usr/bin/env python3
"""Multi-variant robustness scan vs Fast-ALPR — find configs that RELIABLY win.

The single-variant scan over-counts flukes (a config that defeats one random
image but not others). This evaluates a deduplicated set of candidate configs
across K variants each, ranks by reliable defeat rate (fraction of variants the
detector finds nothing), tiers them, and keeps the top-5 that defeat Fast-ALPR
most reliably with the least perturbation.

Needs the ocr-fastalpr extra. Run with: uv run python examples/scan_alpr_robust.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from scan_alpr import Cfg, build, make_scene, pattern, subtlety  # type: ignore[import-not-found]

from plateshapez.ocr import FastALPREngine, normalize_plate
from plateshapez.perturbations.blur import BlurPerturbation
from plateshapez.perturbations.glare import GlarePerturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.perturbations.shapes import ShapesPerturbation
from plateshapez.perturbations.texture import TexturePerturbation

K = 10  # variants per candidate
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def candidates() -> list[Cfg]:
    cfgs: list[Cfg] = []
    # Perspective grid (the validated winner family).
    for st in (0.18, 0.24, 0.28, 0.32, 0.40):
        for tilt in ("v", "h"):
            cfgs.append(Cfg(perts=[], persp={"strength": st, "tilt": tilt},
                            label=f"persp {st}{tilt}"))
    # Perspective + light noise.
    cfgs.append(Cfg(perts=[(NoisePerturbation, dict(intensity=18))],
                    persp={"strength": 0.30, "tilt": "v"}, label="persp0.30v + noise18"))
    # Surface-only controls (expected unreliable / ineffective).
    cfgs += [
        Cfg(perts=[(NoisePerturbation, dict(intensity=60))], label="noise60"),
        Cfg(perts=[(NoisePerturbation, dict(intensity=90))], label="noise90"),
        Cfg(perts=[(BlurPerturbation, dict(radius=8))], label="blur8"),
        Cfg(perts=[(BlurPerturbation, dict(radius=15))], label="blur15"),
        Cfg(perts=[(BlurPerturbation, dict(radius=8)), (NoisePerturbation, dict(intensity=60))],
            label="blur8+noise60"),
        Cfg(perts=[(ShapesPerturbation,
                    dict(num_shapes=18, min_size=10, max_size=44, color="white"))],
            label="shapes white"),
        Cfg(perts=[(TexturePerturbation, dict(type="scratches", intensity=0.7))],
            label="scratch0.7"),
        Cfg(perts=[(GlarePerturbation, dict(intensity=0.95, spread=0.5))], label="glare0.95"),
    ]
    return cfgs


def main() -> None:
    root = Path("dataset/scan_alpr_robust")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bg, plate, pos, mask = make_scene(900, 450)
    alpr = FastALPREngine()

    results = []
    for cfg in candidates():
        defeats = 0
        for seed in range(K):
            import random as _r

            import numpy as _np

            _r.seed(seed)
            _np.random.seed(seed)
            comp = build(bg, plate, pos, mask, Cfg(cfg.perts, cfg.persp, cfg.label), None)
            read = normalize_plate(alpr.read(comp))
            if not read:
                defeats += 1
        rate = defeats / K
        results.append((rate, cfg))
        print(f"  {cfg.label:24} reliable defeat {defeats}/{K} ({rate:.0%})")

    tiers = {"robust (>=90%)": 0, "frequent (50-89%)": 0,
             "occasional (10-49%)": 0, "never (<10%)": 0}
    for rate, _ in results:
        key = ("robust (>=90%)" if rate >= 0.9 else "frequent (50-89%)" if rate >= 0.5
               else "occasional (10-49%)" if rate >= 0.1 else "never (<10%)")
        tiers[key] += 1
    print("\nTiers by reliable defeat rate:")
    for k, v in tiers.items():
        print(f"  {k:22} {v}")

    results.sort(key=lambda r: (-r[0], subtlety(r[1])))
    top = results[:5]

    big_bg, big_plate, big_pos, big_mask = make_scene(1200, 600)
    box = (big_pos[0], big_pos[1], big_pos[0] + 1200, big_pos[1] + 600)
    rows = []
    print("\nTop 5 (most reliable, least perturbation):")
    for rank, (rate, cfg) in enumerate(top, 1):
        d = root / f"rank{rank}"
        d.mkdir()
        comp = build(big_bg, big_plate, big_pos, big_mask, cfg, None)
        comp.crop(box).save(d / "perturbed.png")
        pattern(big_bg.size, big_plate, big_pos, big_mask, cfg).crop(box).save(d / "pattern.png")
        rows.append((rank, cfg.label, rate))
        print(f"  #{rank} {rate:.0%} reliable  {cfg.label}")

    _montage(root, rows)
    shutil.make_archive(str(root / "top5_robust"), "zip", root)
    print(f"\n✅ tiers above; robust top-5 in {root}; zip {root / 'top5_robust.zip'}")


def _montage(root: Path, rows: list) -> None:
    tw, th, pad = 360, 180, 12
    sheet = Image.new("RGB", (pad * 3 + tw * 2, pad + len(rows) * (th + pad)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 15)
    for i, (rank, label, rate) in enumerate(rows):
        y = pad + i * (th + pad)
        img = Image.open(root / f"rank{rank}" / "perturbed.png").convert("RGB").resize((tw, th))
        sheet.paste(img, (pad, y))
        tx = pad * 2 + tw
        d.text((tx, y), f"#{rank}  {rate:.0%} reliable defeat", fill=(0, 120, 0), font=f)
        d.text((tx, y + 26), label, fill=(40, 40, 40), font=f)
    sheet.save(root / "overview.png")


if __name__ == "__main__":
    main()

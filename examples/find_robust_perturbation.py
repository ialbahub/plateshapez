#!/usr/bin/env python3
"""Find a perturbation that defeats OCR at any perspective yet stays readable.

Generates 100 variations of one plate across several perturbation "recipes" and
viewing perspectives, reads each with multiple OCR engines (Tesseract + EasyOCR),
and reports which recipe defeats *every* engine at *every* perspective while
leaving the characters humanly visible (low occlusion of the character band).

Then writes, for the winning recipe:
  clean.png            - the plate, no perturbation
  perturbed.png        - the plate with the perturbation (mid perspective)
  perturbation_only.png - just the noise + symbols, transparent PNG (no plate)

Run with: uv run python examples/find_robust_perturbation.py
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from plateshapez.ocr import (
    EasyOCREngine,
    OCREngine,
    TesseractEngine,
    character_score,
    normalize_plate,
)
from plateshapez.perturbations.base import Perturbation
from plateshapez.perturbations.glare import GlarePerturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.perturbations.perspective import PerspectivePerturbation
from plateshapez.perturbations.shapes import ShapesPerturbation
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position, isolate_neutral_layer

PLATE = "Q0D1I5"
BAND = (0.28, 0.74)
FAIL_BELOW = 60.0  # OCR "fails" when its character score is below this
NEUTRAL = 128


@dataclass
class Recipe:
    name: str
    factory: Callable[[], list[Perturbation]]  # fresh perturbation instances per variant


RECIPES = [
    Recipe("noise", lambda: [NoisePerturbation(intensity=40)]),
    Recipe("symbols", lambda: [ShapesPerturbation(num_shapes=18, min_size=10, max_size=44)]),
    Recipe(
        "symbols+noise",
        lambda: [
            ShapesPerturbation(num_shapes=18, min_size=10, max_size=44),
            NoisePerturbation(intensity=35),
        ],
    ),
    Recipe("glare", lambda: [GlarePerturbation(intensity=0.85, spread=0.5)]),
]

# (label, perspective kwargs or None)
PERSPECTIVES = [
    ("front", None),
    ("h-mild", {"strength": 0.15, "tilt": "h"}),
    ("v-mild", {"strength": 0.15, "tilt": "v"}),
    ("h-steep", {"strength": 0.28, "tilt": "h"}),
    ("v-steep", {"strength": 0.28, "tilt": "v"}),
]
VARIANTS = 5  # 4 recipes x 5 perspectives x 5 = 100


def plate_mask(plate: Image.Image, pos: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    bx, by = pos
    ow, oh = plate.size
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[by : by + oh, bx : bx + ow] = np.array(plate.split()[-1]) > 0
    return mask


def make_variant(
    bg: Image.Image,
    plate: Image.Image,
    pos: tuple[int, int],
    recipe: Recipe,
    persp: dict | None,
    seed: int,
) -> Image.Image:
    """Composite the plate, apply the recipe (clipped to the plate), then warp."""
    random.seed(seed)
    np.random.seed(seed)
    comp = bg.copy()
    comp.paste(plate, pos, plate)
    x, y = pos
    region = (x, y, plate.size[0], plate.size[1])

    base = comp.copy()
    for pert in recipe.factory():
        comp = pert.apply(comp, region)
    # clip noise/symbols to the plate's opaque pixels
    mask = plate_mask(plate, pos, bg.size)
    arr, barr = np.array(comp), np.array(base)
    arr[~mask] = barr[~mask]
    comp = Image.fromarray(arr)

    if persp is not None:
        comp = PerspectivePerturbation(**persp).apply(comp, region)
    return comp


def perturbation_layer(plate: Image.Image, recipe: Recipe, seed: int) -> Image.Image:
    """Render just the recipe's noise+symbols on neutral grey -> transparent PNG."""
    random.seed(seed)
    np.random.seed(seed)
    neutral = Image.new("RGB", plate.size, (NEUTRAL,) * 3)
    region = (0, 0, plate.size[0], plate.size[1])
    for pert in recipe.factory():
        neutral = pert.apply(neutral, region)
    # keep only inside the plate shape
    arr = np.array(neutral)
    alpha = np.array(plate.split()[-1]) > 0
    arr[~alpha] = NEUTRAL
    return isolate_neutral_layer(Image.fromarray(arr), NEUTRAL)


def band_crop(img: Image.Image, pos: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    bx, by = pos
    ow, oh = size
    return img.crop((bx, by + int(oh * BAND[0]), bx + ow, by + int(oh * BAND[1])))


def main() -> None:
    out = "dataset/robust_search"
    import os

    os.makedirs(out, exist_ok=True)

    bg = create_vehicle_background()
    plate = create_plate_overlay(PLATE)
    pos = calculate_center_position(bg, plate)
    expected = normalize_plate(PLATE)

    engines: dict[str, OCREngine] = {
        "tesseract": TesseractEngine(),
        "easyocr": EasyOCREngine(),
    }

    # recipe -> list of (both_fail, scores per engine, persp_label)
    stats: dict[str, list[tuple[bool, dict[str, float], str]]] = {r.name: [] for r in RECIPES}
    total = 0
    for recipe in RECIPES:
        for plabel, persp in PERSPECTIVES:
            for v in range(VARIANTS):
                seed = 1000 + total
                comp = make_variant(bg, plate, pos, recipe, persp, seed)
                crop = band_crop(comp, pos, plate.size)
                scores = {
                    name: character_score(expected, normalize_plate(eng.read(crop)))
                    for name, eng in engines.items()
                }
                both_fail = all(s < FAIL_BELOW for s in scores.values())
                stats[recipe.name].append((both_fail, scores, plabel))
                total += 1

    print(f"\n===== {total} variations of plate {PLATE} =====")
    print(f"OCR engines: {', '.join(engines)} | 'defeated' = every engine scores < {FAIL_BELOW}\n")
    print(f"{'recipe':16} {'defeat-all-OCR':>15} {'mean tess':>10} {'mean easy':>10}")
    ranked = []
    for r in RECIPES:
        rows = stats[r.name]
        defeat_rate = sum(b for b, _, _ in rows) / len(rows)
        mt = np.mean([s["tesseract"] for _, s, _ in rows])
        me = np.mean([s["easyocr"] for _, s, _ in rows])
        # robust across perspective => defeated at every perspective label
        per_persp: dict[str, list[bool]] = {pl: [] for pl, _ in PERSPECTIVES}
        for b, _, pl in rows:
            per_persp[pl].append(b)
        all_persp = all(all(v) for v in per_persp.values())
        ranked.append((r.name, defeat_rate, all_persp, mt, me))
        flag = "  <-- every perspective" if all_persp else ""
        print(f"{r.name:16} {defeat_rate:>14.0%} {mt:>10.1f} {me:>10.1f}{flag}")

    # Winner: defeats OCR at every perspective if possible, else highest defeat rate.
    robust = [x for x in ranked if x[2]]
    winner = (robust or sorted(ranked, key=lambda x: -x[1]))[0]
    wname = winner[0]
    wrecipe = next(r for r in RECIPES if r.name == wname)
    print(f"\nWINNER: '{wname}' (defeat-all-OCR {winner[1]:.0%}, every-perspective={winner[2]})")

    # Deliverables for the winner.
    clean = bg.copy()
    clean.paste(plate, pos, plate)
    clean.save(f"{out}/clean.png")
    perturbed = make_variant(bg, plate, pos, wrecipe, {"strength": 0.15, "tilt": "h"}, seed=42)
    perturbed.save(f"{out}/perturbed.png")
    perturbation_layer(plate, wrecipe, seed=42).save(f"{out}/perturbation_only.png")
    print(f"wrote {out}/clean.png, perturbed.png, perturbation_only.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optimize for a subtle-but-OCR-defeating perturbation via iterative search.

Searches the perturbation parameter space (shape count/size/opacity/greylevel,
noise intensity, scratch intensity) to minimize an objective that trades OCR
readability against visibility:

    objective = mean_worst_OCR_score + VIS_WEIGHT * visibility%

It runs a random round, then hill-climbs (jitters the best configs) for a few
rounds — "continuously improving". The fast engines (Tesseract + RapidOCR) drive
the search; the top finalists are then validated with EasyOCR too. Saves the
winner's clean/perturbed/pattern at 300 DPI and prints the analysis.

Run with: uv run python examples/optimize_subtle.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from plateshapez.ocr import (
    EasyOCREngine,
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
K = 4  # variants per config
VIS_WEIGHT = 1.5  # OCR points traded per 1% visibility
RANDOM_CONFIGS = 16
REFINE_ROUNDS = [4, 3, 2]  # how many top configs to jitter each refine round

CHOICES = {
    "count": [0, 6, 10, 14, 18, 24, 30],
    "smin": [2, 3, 4, 6, 8],
    "smax": [8, 14, 20, 28, 40],
    "lum": [70, 100, 130, 170, 210, 235],
    "alpha": [50, 80, 120, 160, 200, 255],
    "noise": [0, 15, 25, 40, 60, 80],
    "scratch": [0.0, 0.3, 0.5, 0.7],
}


def sample(rng: random.Random) -> dict:
    cfg = {k: rng.choice(v) for k, v in CHOICES.items()}
    cfg["smax"] = max(cfg["smax"], cfg["smin"] + 4)
    if cfg["count"] == 0 and cfg["noise"] == 0 and cfg["scratch"] == 0.0:
        cfg["noise"] = 40  # avoid a no-op config
    return cfg


def jitter(cfg: dict, rng: random.Random) -> dict:
    out = dict(cfg)
    for k in rng.sample(list(CHOICES), 3):  # nudge 3 params to a neighbour value
        opts = CHOICES[k]
        i = opts.index(cfg[k]) if cfg[k] in opts else 0
        out[k] = opts[max(0, min(len(opts) - 1, i + rng.choice([-1, 1])))]
    out["smax"] = max(out["smax"], out["smin"] + 4)
    return out


def to_perturbations(cfg: dict) -> list:
    perts: list = []
    if cfg["count"] > 0:
        perts.append(
            {
                "name": "shapes",
                "params": {
                    "num_shapes": cfg["count"],
                    "min_size": cfg["smin"],
                    "max_size": cfg["smax"],
                    "color": [cfg["lum"], cfg["lum"], cfg["lum"], cfg["alpha"]],
                },
            }
        )
    if cfg["scratch"] > 0:
        perts.append(
            {"name": "texture", "params": {"type": "scratches", "intensity": cfg["scratch"]}}
        )
    if cfg["noise"] > 0:
        perts.append({"name": "noise", "params": {"intensity": cfg["noise"]}})
    return perts


def main() -> None:
    root = Path("dataset/optimize")
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

    expected = normalize_plate(PLATE)
    search_engines = {"tesseract": TesseractEngine(), "rapidocr": RapidOCREngine()}

    cache: dict[str, dict] = {}
    counter = [0]

    def evaluate(cfg: dict) -> dict:
        key = json.dumps(cfg, sort_keys=True)
        if key in cache:
            return cache[key]
        out = root / f"cfg{counter[0]:03d}"
        counter[0] += 1
        DatasetGenerator(
            bg_dir=ind / "backgrounds",
            overlay_dir=ind / "overlays",
            out_dir=out,
            perturbations=to_perturbations(cfg),
            random_seed=100,
            save_perturbation_layer=False,
        ).run(n_variants=K)
        vis, worst = [], []
        for lp in sorted((out / "labels").glob("*.json")):
            meta = json.loads(lp.read_text())
            img = Image.open(out / "images" / f"{lp.stem}.png").convert("RGB")
            vis.append(
                float(np.abs(np.array(img).astype(np.int16) - clean_arr)[mask].mean()) / 255 * 100
            )
            crop = crop_plate(img, meta)
            sc = [
                character_score(expected, normalize_plate(e.read(crop)))
                for e in search_engines.values()
            ]
            worst.append(max(sc))
        res = {"cfg": cfg, "vis": float(np.mean(vis)), "worst": float(np.mean(worst))}
        res["objective"] = res["worst"] + VIS_WEIGHT * res["vis"]
        cache[key] = res
        return res

    rng = random.Random(0)
    results = [evaluate(sample(rng)) for _ in range(RANDOM_CONFIGS)]
    results.sort(key=lambda r: r["objective"])
    print(
        f"round0 (random {RANDOM_CONFIGS}): best objective {results[0]['objective']:.1f} "
        f"(worst {results[0]['worst']:.0f}, vis {results[0]['vis']:.2f}%)"
    )

    for rnd, topn in enumerate(REFINE_ROUNDS, 1):
        seeds = results[:topn]
        for s in seeds:
            for _ in range(3):
                results.append(evaluate(jitter(s["cfg"], rng)))
        results.sort(key=lambda r: r["objective"])
        print(
            f"round{rnd}: best objective {results[0]['objective']:.1f} "
            f"(worst {results[0]['worst']:.0f}, vis {results[0]['vis']:.2f}%)  [{len(cache)} configs tried]"
        )

    # Validate the top finalists with EasyOCR too (full 3-engine worst).
    try:
        easy = EasyOCREngine()
    except Exception:
        easy = None
    print("\nTop 8 (objective ascending):")
    print(f"{'worst':>6}{'vis%':>7}{'obj':>7}  params")
    for r in results[:8]:
        c = r["cfg"]
        params = (
            f"shapes n={c['count']} {c['smin']}-{c['smax']}px lum{c['lum']} a{c['alpha']} | "
            f"noise {c['noise']} | scratch {c['scratch']}"
        )
        print(f"{r['worst']:6.0f}{r['vis']:7.2f}{r['objective']:7.1f}  {params}")

    best = results[0]
    print(f"\nWINNER objective {best['objective']:.1f}: {best['cfg']}")
    if easy is not None:
        # recompute winner worst across all three engines on a fresh sample
        out = root / "winner_check"
        DatasetGenerator(
            bg_dir=ind / "backgrounds",
            overlay_dir=ind / "overlays",
            out_dir=out,
            perturbations=to_perturbations(best["cfg"]),
            random_seed=200,
        ).run(n_variants=K)
        engines3 = {
            "tesseract": search_engines["tesseract"],
            "rapidocr": search_engines["rapidocr"],
            "easyocr": easy,
        }
        for lp in sorted((out / "labels").glob("*.json"))[:3]:
            meta = json.loads(lp.read_text())
            crop = crop_plate(Image.open(out / "images" / f"{lp.stem}.png").convert("RGB"), meta)
            reads = {n: normalize_plate(e.read(crop)) for n, e in engines3.items()}
            sc = {n: character_score(expected, reads[n]) for n in reads}
            print("  reads:", "  ".join(f"{n}:{reads[n] or '_'}({sc[n]:.0f})" for n in reads))


if __name__ == "__main__":
    main()

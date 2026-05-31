#!/usr/bin/env python3
"""Generate many perturbed plates and sort them by OCR read success/failure.

Builds a vehicle background and several plate overlays, generates many perturbed
variants, then runs an OCR engine over them and splits the images into
``ocr_eval/success`` and ``ocr_eval/failure`` folders (plus a ``results.csv``).

Requires the OCR extra and the tesseract binary:
    uv sync --extra ocr   # and: apt-get install tesseract-ocr
Run with:
    uv run python examples/ocr_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

from plateshapez.ocr import TesseractEngine, evaluate_dataset
from plateshapez.pipeline import DatasetGenerator
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background

PLATES = ["4J9T7W", "AA9AFX", "BXR4821", "7TLK092", "9WPD513", "3HVN806"]
VARIANTS = 40  # 6 plates x 40 = 240 images


def build_inputs(root: Path) -> None:
    (root / "backgrounds").mkdir(parents=True, exist_ok=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(root / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    for text in PLATES:
        # Name the overlay by the plate text so it is the OCR ground truth.
        create_plate_overlay(text).save(root / "overlays" / f"{text}.png")


def main() -> None:
    root = Path("dataset/ocr_benchmark")
    build_inputs(root)

    DatasetGenerator(
        bg_dir=root / "backgrounds",
        overlay_dir=root / "overlays",
        out_dir=root / "dataset",
        perturbations=[
            {"name": "shapes", "params": {"num_shapes": 16, "min_size": 8, "max_size": 42}},
            {"name": "noise", "params": {"intensity": 24}},
        ],
        random_seed=2025,
        save_perturbation_layer=False,
        confine_to_plate=True,
    ).run(n_variants=VARIANTS)

    summary = evaluate_dataset(
        root / "dataset",
        TesseractEngine(),
        success_threshold=100.0,  # exact plate read
        out_dir=root / "ocr_eval",
    )

    print("\n===== OCR benchmark =====")
    print(f"engine        : {summary['engine']}")
    print(f"images        : {summary['total']}")
    print(f"read success  : {summary['success']}  ({summary['read_accuracy']:.1%})")
    print(f"read failure  : {summary['failure']}")
    print(f"mean char score: {summary['mean_score']:.1f}/100")
    print(f"sorted into   : {root / 'ocr_eval' / 'success'} | {root / 'ocr_eval' / 'failure'}")
    print(f"per-image CSV : {summary['results_csv']}")

    misreads = [r for r in summary["results"] if not r.success][:8]
    if misreads:
        print("\nsample misreads (expected -> predicted, score):")
        for r in misreads:
            print(f"  {r.expected:8s} -> {r.predicted or '∅':8s}  {r.score:5.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sweep perturbation intensity and chart OCR accuracy vs. strength.

Finds the "breaking point" of each perturbation by generating datasets across a
range of intensities, reading them with an OCR engine, and plotting read
accuracy against intensity. Writes ``sweep.csv`` and ``sweep_chart.png``.

Run with: uv run python examples/ocr_intensity_sweep.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from plateshapez.ocr import TesseractEngine, sweep_intensity
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background

PLATES = ["4J9T7W", "AA9AFX", "BXR4821"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Each curve: (label, perturbation, param, values, base_params, colour)
SWEEPS: list[tuple[str, str, str, Sequence[float], dict, tuple[int, int, int]]] = [
    (
        "perspective",
        "perspective",
        "strength",
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        {"tilt": "h"},
        (200, 70, 50),
    ),
    ("blur", "blur", "radius", [0, 1, 2, 3, 4, 6, 8, 10], {"type": "gaussian"}, (40, 90, 200)),
    (
        "shapes",
        "shapes",
        "num_shapes",
        [0, 5, 10, 20, 30, 45, 60, 80],
        {"min_size": 10, "max_size": 48},
        (40, 160, 70),
    ),
]


def build_inputs(root: Path) -> None:
    (root / "backgrounds").mkdir(parents=True, exist_ok=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    create_vehicle_background().save(root / "backgrounds" / "vehicle.jpg", "JPEG", quality=92)
    for text in PLATES:
        create_plate_overlay(text).save(root / "overlays" / f"{text}.png")


def draw_chart(curves: dict[str, list[dict]], colours: dict[str, tuple], path: Path) -> None:
    """Render a simple line chart of read accuracy vs. perturbation intensity."""
    W, H = 900, 520
    ml, mr, mt, mb = 90, 220, 40, 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title_f = ImageFont.truetype(FONT, 20)
    f = ImageFont.truetype(FONT, 14)
    px0, py0, px1, py1 = ml, mt, W - mr, H - mb

    d.text((ml, 8), "OCR read score vs perturbation intensity", fill=(0, 0, 0), font=title_f)
    # Axes + gridlines (0-100%).
    d.rectangle((px0, py0, px1, py1), outline=(0, 0, 0), width=2)
    for pct in range(0, 101, 20):
        y = py1 - (py1 - py0) * pct / 100
        d.line([px0, y, px1, y], fill=(225, 225, 225))
        d.text((px0 - 52, y - 8), f"{pct:3d}", fill=(0, 0, 0), font=f)
    d.text((10, mt - 2), "score", fill=(0, 0, 0), font=f)
    d.text(
        (ml, py1 + 30), "intensity (normalised 0 = none -> 1 = max tested)", fill=(0, 0, 0), font=f
    )

    legend_y = mt + 10
    for label, rows in curves.items():
        colour = colours[label]
        n = len(rows)
        pts = []
        for i, r in enumerate(rows):
            x = px0 + (px1 - px0) * (i / max(1, n - 1))
            y = py1 - (py1 - py0) * (r["mean_score"] / 100.0)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=colour, width=3)
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour)
        d.text((px1 + 16, legend_y), f"{label}", fill=colour, font=f)
        legend_y += 22

    img.save(path)


def main() -> None:
    root = Path("dataset/ocr_sweep")
    build_inputs(root)
    engine = TesseractEngine()

    curves: dict[str, list[dict]] = {}
    colours: dict[str, tuple] = {}
    rows_csv: list[dict] = []
    for label, pert, param, values, base, colour in SWEEPS:
        rows = sweep_intensity(
            root / "backgrounds",
            root / "overlays",
            engine,
            perturbation=pert,
            param=param,
            values=values,
            base_params=base,
            n_variants=8,
            work_dir=root / "work" / label,
        )
        curves[label] = rows
        colours[label] = colour
        print(f"\n{label} sweep ({param}):")
        for r in rows:
            print(
                f"  {param}={r['value']:>4}: accuracy {r['read_accuracy']:5.1%}  "
                f"mean score {r['mean_score']:5.1f}  (n={r['total']})"
            )
            rows_csv.append({"sweep": label, "param": param, **r})

    with open(root / "sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sweep", "param", "value", "read_accuracy", "mean_score", "total"]
        )
        writer.writeheader()
        writer.writerows(rows_csv)

    draw_chart(curves, colours, root / "sweep_chart.png")
    print(f"\n✅ Wrote {root / 'sweep.csv'} and {root / 'sweep_chart.png'}")


if __name__ == "__main__":
    main()

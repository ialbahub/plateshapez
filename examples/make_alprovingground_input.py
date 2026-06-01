#!/usr/bin/env python3
"""Build an ALPRovingGround-ready input folder of perturbed plate scenes.

ALPRovingGround (github.com/bennjordan/ALPRovingGround) runs a folder of images
through Fast-ALPR and writes alpr_results.csv. This produces that folder: full
vehicle+plate scenes for several plates, each with perspective ATTACKS (expected
to defeat the detector) and CONTROLS (expected to be read), named so you can
correlate them with the CSV. Drop the folder into ALPRbatch.py on your machine.

Run with: uv run python examples/make_alprovingground_input.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from plateshapez.perturbations.blur import BlurPerturbation
from plateshapez.perturbations.noise import NoisePerturbation
from plateshapez.perturbations.perspective import PerspectivePerturbation
from plateshapez.perturbations.shapes import ShapesPerturbation
from plateshapez.synthetic import create_plate_overlay, create_vehicle_background
from plateshapez.utils.overlay import calculate_center_position

PLATES = ["4J9T7W", "AA9AFX", "BXR4821", "7TLK092", "9WPD513"]

Group = dict[str, tuple[list, dict | None]]
# name -> (surface perturbations applied to the plate, perspective kwargs or None)
ATTACKS: Group = {
    "attack_persp_v0.28": ([], dict(strength=0.28, tilt="v")),
    "attack_persp_v0.34": ([], dict(strength=0.34, tilt="v")),
    "attack_persp_h0.40": ([], dict(strength=0.40, tilt="h")),
    "attack_persp_v0.30_noise": ([(NoisePerturbation, dict(intensity=18))],
                                 dict(strength=0.30, tilt="v")),
}
CONTROLS: Group = {
    "control_clean": ([], None),
    "control_noise60": ([(NoisePerturbation, dict(intensity=60))], None),
    "control_blur8": ([(BlurPerturbation, dict(radius=8))], None),
    "control_shapes_white": (
        [(ShapesPerturbation, dict(num_shapes=18, min_size=10, max_size=44, color="white"))],
        None,
    ),
}


def main() -> None:
    out = Path("dataset/alprovinggound_input")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bg = create_vehicle_background(size=(1600, 1100))
    count = 0
    for text in PLATES:
        plate = create_plate_overlay(text, size=(1200, 600))
        pos = calculate_center_position(bg, plate)
        ow, oh = plate.size
        region = (pos[0], pos[1], ow, oh)
        mask = np.zeros((bg.height, bg.width), bool)
        mask[pos[1] : pos[1] + oh, pos[0] : pos[0] + ow] = np.array(plate.split()[-1]) > 0

        for group in (ATTACKS, CONTROLS):
            for name, (perts, persp) in group.items():
                random.seed(1)
                np.random.seed(1)
                comp = bg.copy()
                comp.paste(plate, pos, plate)
                base = comp.copy()
                for cls, kw in perts:
                    comp = cls(**kw).apply(comp, region)
                arr = np.array(comp)
                arr[~mask] = np.array(base)[~mask]
                comp = Image.fromarray(arr)
                if persp is not None:
                    comp = PerspectivePerturbation(**persp).apply(comp, region)
                comp.save(out / f"{text}__{name}.jpg", "JPEG", quality=90)
                count += 1

    (out / "README.txt").write_text(
        "ALPRovingGround input set\n"
        "=========================\n"
        "Run ALPRbatch.py and select THIS folder.\n\n"
        "Filenames: <PLATE>__<technique>.jpg\n"
        "  attack_*  : perspective skew — expected to DEFEAT detection (no/empty read)\n"
        "  control_* : clean/noise/blur/shapes — expected to be DETECTED and read\n\n"
        "Check annotated_output/ and alpr_results.csv: the attack_* rows should show\n"
        "no detection or a wrong read; control_* should be read (modulo font confusion).\n"
    )
    shutil.make_archive(str(out), "zip", out)
    print(f"✅ {count} images in {out} (+ README); zip {out}.zip")


if __name__ == "__main__":
    main()

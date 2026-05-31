import random
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image

from plateshapez.perturbations.base import PERTURBATION_REGISTRY
from plateshapez.utils.io import iter_backgrounds, iter_overlays, save_image, save_metadata
from plateshapez.utils.overlay import (
    calculate_center_position,
    ensure_rgb,
    ensure_rgba,
    isolate_neutral_layer,
)

# Flat grey the perturbation layer is rendered on so additive noise is centred
# without clipping against the plate's white/black pixels.
NEUTRAL_GREY = 128


class DatasetGenerator:
    """Main class for generating adversarial license plate datasets.

    Orchestrates the process of combining background vehicle images with license plate
    overlays and applying configurable perturbations to create training datasets for
    adversarial robustness research.
    """

    class PerturbationConf(TypedDict, total=False):
        name: str
        params: dict[str, Any]

    def __init__(
        self,
        bg_dir: str | Path,
        overlay_dir: str | Path,
        out_dir: str | Path,
        perturbations: list["DatasetGenerator.PerturbationConf"] | None = None,
        random_seed: int | None = None,
        save_metadata: bool = True,
        save_perturbation_layer: bool = True,
        confine_to_plate: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the dataset generator.

        Args:
            bg_dir: Directory containing background vehicle images (JPG)
            overlay_dir: Directory containing license plate overlays (PNG with alpha)
            out_dir: Output directory for generated dataset
            perturbations: List of perturbation configurations to apply
            random_seed: Random seed for reproducible generation
            save_metadata: Whether to save JSON metadata files
            save_perturbation_layer: Whether to also save the perturbations
                (patterns and noise) as a separate transparent image per variant
            confine_to_plate: Clip every perturbation to the plate's opaque pixels
                (the overlay's alpha) so nothing spills past the plate edges or
                rounded corners onto the vehicle. Disable for whole-image
                ("global" scope) perturbations.
            verbose: Enable verbose logging output
        """
        self.bg_dir: Path = Path(bg_dir)
        self.ov_dir: Path = Path(overlay_dir)
        self.out_dir: Path = Path(out_dir)
        self.img_dir: Path = self.out_dir / "images"
        self.label_dir: Path = self.out_dir / "labels"
        self.pert_dir: Path = self.out_dir / "perturbations"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.perturbations: list[DatasetGenerator.PerturbationConf] = perturbations or []
        self.random_seed: int | None = random_seed
        self.save_metadata: bool = save_metadata
        self.save_perturbation_layer: bool = save_perturbation_layer
        self.confine_to_plate: bool = confine_to_plate
        if self.save_perturbation_layer:
            self.pert_dir.mkdir(parents=True, exist_ok=True)
        self.verbose: bool = verbose

    def run(self, n_variants: int = 5) -> None:
        """Generate dataset with deterministic seeding."""
        # Deterministic seeding for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            # Also seed PIL's internal random for consistent image operations
            # PIL uses Python's random module internally

        backgrounds = list(iter_backgrounds(self.bg_dir))
        overlays = list(iter_overlays(self.ov_dir))

        if not backgrounds:
            raise ValueError(f"No background images found in {self.bg_dir}")
        if not overlays:
            raise ValueError(f"No overlay images found in {self.ov_dir}")

        total_images = 0
        for bg_path in backgrounds:
            try:
                bg = ensure_rgb(Image.open(bg_path))
            except (IOError, OSError, ValueError) as e:
                if self.verbose:
                    print(f"Warning: Could not load background {bg_path}: {e}")
                continue

            for ov_path in overlays:
                try:
                    overlay = ensure_rgba(Image.open(ov_path))
                except (IOError, OSError, ValueError) as e:
                    if self.verbose:
                        print(f"Warning: Could not load overlay {ov_path}: {e}")
                    continue
                position = calculate_center_position(bg, overlay)
                ow, oh = overlay.size
                bx, by = position

                # Boolean mask of the plate's opaque pixels in full-image
                # coordinates, used to clip perturbations to the plate shape.
                plate_mask: np.ndarray | None = None
                if self.confine_to_plate:
                    plate_mask = np.zeros((bg.height, bg.width), dtype=bool)
                    ov_alpha = np.array(overlay.split()[-1]) > 0
                    plate_mask[by : by + oh, bx : bx + ow] = ov_alpha

                for i in range(n_variants):
                    # Create composite image
                    img = bg.copy()
                    img.paste(overlay, position, overlay)

                    # Deterministic file stem shared by the composite, its label,
                    # and the isolated perturbation image.
                    stem = f"{bg_path.stem}_{ov_path.stem}_{i:03d}"

                    # Clean composite kept so perturbations can be clipped back
                    # to the plate shape after they are applied.
                    base = img.copy() if plate_mask is not None else None

                    # A flat neutral-grey canvas the same perturbations are
                    # re-rendered onto, so the saved layer has no plate underneath.
                    neutral = (
                        Image.new("RGB", img.size, (NEUTRAL_GREY,) * 3)
                        if self.save_perturbation_layer
                        else None
                    )

                    # Apply perturbations
                    applied: list[dict[str, Any]] = []
                    for perturbation_conf in self.perturbations:
                        name = perturbation_conf["name"]
                        if name not in PERTURBATION_REGISTRY:
                            raise ValueError(f"Unknown perturbation: {name}")

                        cls = PERTURBATION_REGISTRY[name]
                        pert = cls(**perturbation_conf.get("params", {}))

                        if neutral is not None:
                            # Replay the exact same randomness onto the neutral
                            # canvas so its perturbation matches the composite.
                            py_state = random.getstate()
                            np_state = np.random.get_state()
                            img = pert.apply(img, (bx, by, ow, oh))
                            random.setstate(py_state)
                            np.random.set_state(np_state)
                            neutral = pert.apply(neutral, (bx, by, ow, oh))
                        else:
                            img = pert.apply(img, (bx, by, ow, oh))
                        applied.append(pert.serialize())

                    # Clip perturbations to the plate: restore the clean composite
                    # anywhere outside the plate's opaque pixels.
                    if plate_mask is not None and base is not None:
                        img_arr = np.array(img)
                        img_arr[~plate_mask] = np.array(base)[~plate_mask]
                        img = Image.fromarray(img_arr)
                        if neutral is not None:
                            neutral_arr = np.array(neutral)
                            neutral_arr[~plate_mask] = NEUTRAL_GREY
                            neutral = Image.fromarray(neutral_arr)

                    # Save composite image
                    fname = f"{stem}.png"
                    save_image(img, self.img_dir / fname)

                    # Save all perturbations (patterns + noise) together in one
                    # clear image, with the plate and background removed.
                    if neutral is not None:
                        layer = isolate_neutral_layer(neutral, NEUTRAL_GREY)
                        save_image(layer, self.pert_dir / fname)

                    # Only save metadata if enabled in config
                    if self.save_metadata:
                        metadata: dict[str, Any] = {
                            "background": bg_path.name,
                            "overlay": ov_path.name,
                            "overlay_position": [bx, by],
                            "overlay_size": [ow, oh],
                            "perturbations": applied,
                            "random_seed": self.random_seed,
                            "variant_index": i,
                        }
                        if self.save_perturbation_layer:
                            metadata["perturbation_layer"] = fname
                        save_metadata(metadata, self.label_dir / f"{stem}.json")

                    total_images += 1
                    if self.verbose:
                        print(f"✓ Generated {fname} ({total_images} total)")
                    elif total_images % 100 == 0:
                        print(f"✓ Generated {total_images} images so far...")

        print(f"\n🎉 Dataset generation complete! Generated {total_images} images.")

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plateshapez.pipeline import DatasetGenerator


class TestDatasetGenerator:
    """Test DatasetGenerator pipeline behavior."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bg_dir = temp_path / "backgrounds"
            overlay_dir = temp_path / "overlays"
            output_dir = temp_path / "output"

            bg_dir.mkdir()
            overlay_dir.mkdir()
            output_dir.mkdir()

            # Create test images
            bg_img = Image.new("RGB", (200, 150), color="blue")
            bg_img.save(bg_dir / "test_bg.jpg")

            overlay_img = Image.new("RGBA", (50, 30), color=(255, 0, 0, 128))
            overlay_img.save(overlay_dir / "test_overlay.png")

            yield {
                "bg_dir": bg_dir,
                "overlay_dir": overlay_dir,
                "output_dir": output_dir,
            }

    def test_pipeline_generates_expected_count(self, temp_dirs):
        """Test that pipeline generates expected number of images."""
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=[],
            random_seed=42,
        )

        n_variants = 3
        gen.run(n_variants=n_variants)

        # Check generated files
        img_dir = temp_dirs["output_dir"] / "images"
        label_dir = temp_dirs["output_dir"] / "labels"

        assert img_dir.exists()
        assert label_dir.exists()

        # Should have n_variants images (1 bg × 1 overlay × n_variants)
        images = list(img_dir.glob("*.png"))
        labels = list(label_dir.glob("*.json"))

        assert len(images) == n_variants
        assert len(labels) == n_variants

    def test_metadata_contains_correct_keys(self, temp_dirs):
        """Test that metadata JSON contains all required keys."""
        perturbations: list[DatasetGenerator.PerturbationConf] = [
            {"name": "noise", "params": {"intensity": 5}}
        ]
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=perturbations,
            random_seed=42,
        )

        gen.run(n_variants=1)

        # Check metadata
        label_dir = temp_dirs["output_dir"] / "labels"
        metadata_files = list(label_dir.glob("*.json"))
        assert len(metadata_files) == 1

        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        required_keys = {
            "background",
            "overlay",
            "overlay_position",
            "overlay_size",
            "perturbations",
            "random_seed",
            "variant_index",
        }

        assert set(metadata.keys()) >= required_keys
        assert metadata["random_seed"] == 42
        assert metadata["variant_index"] == 0
        assert len(metadata["perturbations"]) == 1
        assert metadata["perturbations"][0]["type"] == "noise"

    def test_deterministic_behavior_with_seed(self, temp_dirs):
        """Test that same seed produces identical results."""
        perturbations: list[DatasetGenerator.PerturbationConf] = [
            {"name": "noise", "params": {"intensity": 10}}
        ]

        # First run
        gen1 = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"] / "run1",
            perturbations=perturbations,
            random_seed=123,
        )
        gen1.run(n_variants=2)

        # Second run with same seed
        gen2 = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"] / "run2",
            perturbations=perturbations,
            random_seed=123,
        )
        gen2.run(n_variants=2)

        # Compare generated images
        run1_images = sorted((temp_dirs["output_dir"] / "run1" / "images").glob("*.png"))
        run2_images = sorted((temp_dirs["output_dir"] / "run2" / "images").glob("*.png"))

        assert len(run1_images) == len(run2_images) == 2

        # Images should be identical (deterministic)
        # sourcery skip: no-loop-in-tests
        for img1_path, img2_path in zip(run1_images, run2_images):
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)

            # Convert to arrays and compare
            import numpy as np

            arr1 = np.array(img1)
            arr2 = np.array(img2)

            assert np.array_equal(arr1, arr2), (
                f"Images {img1_path.name} and {img2_path.name} differ"
            )

    def _run_with_rounded_plate(self, confine: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one variant with an elliptical (transparent-cornered) plate.

        Returns the perturbed composite and a boolean mask of the plate's
        transparent corners (inside the bbox but outside the ellipse).
        """
        from PIL import ImageDraw

        from plateshapez.utils.overlay import calculate_center_position, ensure_rgb, ensure_rgba

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            bgd, ovd, out = t / "bg", t / "ov", t / "o"
            bgd.mkdir()
            ovd.mkdir()
            Image.new("RGB", (120, 90), (0, 0, 255)).save(bgd / "bg.jpg")
            ov = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
            ImageDraw.Draw(ov).ellipse((2, 2, 57, 37), fill=(255, 255, 255, 255))
            ov.save(ovd / "ov.png")

            DatasetGenerator(
                bg_dir=bgd,
                overlay_dir=ovd,
                out_dir=out,
                perturbations=[
                    {"name": "shapes", "params": {"num_shapes": 200, "max_size": 10}},
                    {"name": "noise", "params": {"intensity": 40}},
                ],
                random_seed=1,
                save_perturbation_layer=False,
                confine_to_plate=confine,
            ).run(n_variants=1)
            composite = np.array(Image.open(next((out / "images").glob("*.png"))).convert("RGB"))

            # Reconstruct the clean composite to compare against.
            bg = ensure_rgb(Image.open(bgd / "bg.jpg"))
            overlay = ensure_rgba(Image.open(ovd / "ov.png"))
            bx, by = calculate_center_position(bg, overlay)
            clean = bg.copy()
            clean.paste(overlay, (bx, by), overlay)

            outside = np.zeros((bg.height, bg.width), dtype=bool)
            outside[by : by + 40, bx : bx + 60] = np.array(overlay.split()[-1]) == 0
            return composite, np.array(clean), outside

    def test_perturbations_confined_to_plate(self):
        """With confinement, nothing changes outside the plate's opaque pixels."""
        composite, clean, outside = self._run_with_rounded_plate(confine=True)
        # Every transparent-corner pixel is left exactly as the clean composite.
        assert np.array_equal(composite[outside], clean[outside])
        # The plate itself still differs (perturbations were applied).
        assert not np.array_equal(composite, clean)

    def test_perturbations_spill_when_unconfined(self):
        """Without confinement, perturbations leak into the transparent corners."""
        composite, clean, outside = self._run_with_rounded_plate(confine=False)
        assert not np.array_equal(composite[outside], clean[outside])

    def test_all_perturbations_in_one_plate_free_image(self, temp_dirs):
        """All perturbations (patterns + noise) land in a single image per variant."""
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=[
                {"name": "shapes", "params": {"num_shapes": 10}},
                {"name": "noise", "params": {"intensity": 20}},
            ],
            random_seed=42,
        )
        gen.run(n_variants=2)

        pert_dir = temp_dirs["output_dir"] / "perturbations"
        assert pert_dir.exists()

        # Exactly one combined perturbation image per composite.
        layers = sorted(pert_dir.glob("*.png"))
        images = sorted((temp_dirs["output_dir"] / "images").glob("*.png"))
        assert len(layers) == len(images) == 2
        assert [p.name for p in layers] == [p.name for p in images]

        layer = Image.open(layers[0])
        assert layer.mode == "RGBA"
        assert layer.size == Image.open(images[0]).size
        assert layer.getpixel((0, 0))[3] == 0  # transparent outside the perturbations

        # Metadata references the single combined perturbation image.
        with open(temp_dirs["output_dir"] / "labels" / f"{layers[0].stem}.json") as f:
            metadata = json.load(f)
        assert metadata["perturbation_layer"] == layers[0].name

    def test_transparent_perturbation_layer(self, temp_dirs):
        """alpha_gain makes the layer's opacity vary (transparent background)."""
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=[{"name": "noise", "params": {"intensity": 20}}],
            random_seed=3,
            perturbation_alpha_gain=4.0,
        )
        gen.run(n_variants=1)
        layer = np.array(
            Image.open(next((temp_dirs["output_dir"] / "perturbations").glob("*.png")))
        )
        alpha = layer[..., 3]
        assert alpha[0, 0] == 0  # transparent outside the plate
        # Opacity varies with noise strength rather than being a flat 0/255 field.
        partial = alpha[(alpha > 0) & (alpha < 255)]
        assert partial.size > 0

    def test_perturbation_layer_can_be_disabled(self, temp_dirs):
        """No perturbations directory is created when the feature is disabled."""
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=[{"name": "shapes", "params": {"num_shapes": 5}}],
            random_seed=42,
            save_perturbation_layer=False,
        )
        gen.run(n_variants=1)

        assert not (temp_dirs["output_dir"] / "perturbations").exists()
        with open(next((temp_dirs["output_dir"] / "labels").glob("*.json"))) as f:
            metadata = json.load(f)
        assert "perturbation_layer" not in metadata

    def test_noise_layer_has_no_plate_ghost(self):
        """The saved perturbation layer must not encode the plate's structure.

        Additive noise clips at 0/255 over the plate's white/black pixels, which
        previously leaked the plate into a naive difference. Rendering onto a
        neutral canvas removes that, so the layer's grey values are uncorrelated
        with the underlying plate brightness.
        """
        import numpy as np

        from plateshapez.utils.overlay import (
            calculate_center_position,
            ensure_rgb,
            ensure_rgba,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            bg_dir, ov_dir, out = tmp / "bg", tmp / "ov", tmp / "out"
            bg_dir.mkdir()
            ov_dir.mkdir()

            Image.new("RGB", (120, 90), "blue").save(bg_dir / "test_bg.jpg")
            # A high-contrast plate: opaque white with a black bar, so the region
            # has real luminance variance to correlate against.
            plate = Image.new("RGBA", (60, 40), (255, 255, 255, 255))
            plate.paste(Image.new("RGBA", (60, 12), (0, 0, 0, 255)), (0, 14))
            plate.save(ov_dir / "test_overlay.png")

            gen = DatasetGenerator(
                bg_dir=bg_dir,
                overlay_dir=ov_dir,
                out_dir=out,
                perturbations=[{"name": "noise", "params": {"intensity": 25}}],
                random_seed=7,
            )
            gen.run(n_variants=1)

            layer = np.array(
                Image.open(next((out / "perturbations").glob("*.png"))).convert("RGBA")
            )

            bg = ensure_rgb(Image.open(bg_dir / "test_bg.jpg"))
            ov = ensure_rgba(Image.open(ov_dir / "test_overlay.png"))
            base = bg.copy()
            base.paste(ov, calculate_center_position(bg, ov), ov)
            base_lum = np.array(base).astype(float).mean(axis=-1)

        opaque = layer[..., 3] > 0
        grey_dev = layer[..., :3].astype(float).mean(axis=-1) - 128
        assert base_lum[opaque].std() > 50  # the plate really does vary in brightness
        # Noise-only layer should be centred on grey and not track the plate.
        corr = np.corrcoef(base_lum[opaque], grey_dev[opaque])[0, 1]
        assert abs(corr) < 0.1
        assert layer[0, 0, 3] == 0  # transparent outside the perturbed region

    def test_error_on_missing_directories(self):
        """Test that missing directories raise appropriate errors."""
        # Test missing background images
        with pytest.raises(ValueError, match="No background images found"):
            gen = DatasetGenerator(
                bg_dir="/nonexistent/bg",
                overlay_dir="/nonexistent/overlay",
                out_dir="/tmp/test_out",
                random_seed=42,
            )
            gen.run(n_variants=1)

        # Test missing overlay images
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as bg_dir:
            with tempfile.TemporaryDirectory() as overlay_dir:
                # Create a dummy background image so bg_dir is not empty
                from PIL import Image

                bg_img_path = os.path.join(bg_dir, "dummy_bg.jpg")
                Image.new("RGB", (10, 10)).save(bg_img_path)

                # Don't create any overlay images, so overlay_dir is empty

                with pytest.raises(ValueError, match="No overlay images found"):
                    gen = DatasetGenerator(
                        bg_dir=bg_dir,
                        overlay_dir=overlay_dir,
                        out_dir="/tmp/test_out",
                        random_seed=42,
                    )
                    gen.run(n_variants=1)

    def test_unknown_perturbation_raises_error(self, temp_dirs):
        """Test that unknown perturbation names raise ValueError."""
        gen = DatasetGenerator(
            bg_dir=temp_dirs["bg_dir"],
            overlay_dir=temp_dirs["overlay_dir"],
            out_dir=temp_dirs["output_dir"],
            perturbations=[{"name": "nonexistent_perturbation"}],
            random_seed=42,
        )

        with pytest.raises(ValueError, match="Unknown perturbation"):
            gen.run(n_variants=1)

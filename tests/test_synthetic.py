import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from plateshapez.synthetic import (
    BACKGROUND_SIZE,
    DEFAULT_BACKGROUNDS,
    DEFAULT_PLATES,
    PLATE_OVERLAY_SIZE,
    PLATE_SIZE,
    PlateStyle,
    create_background_image,
    create_backgrounds,
    create_plate_overlay,
    create_plate_pattern,
    create_plates,
    create_test_images,
)


class TestPlateOverlay:
    """Test the organised, auto-fit license plate overlay."""

    def _text_ink_bbox(
        self, img: Image.Image, fg: tuple[int, int, int] = (180, 180, 180)
    ) -> tuple[int, int, int, int]:
        """Bounding box of light foreground ink, excluding the border ring."""
        arr = np.array(img)
        w, h = img.size
        b = round(h * 0.05) + round(h * 0.03) + 6  # inside the border outline
        inner = arr[b : h - b, b : w - b]
        mask = (
            (inner[..., 0] > fg[0])
            & (inner[..., 1] > fg[1])
            & (inner[..., 2] > fg[2])
            & (inner[..., 3] > 0)
        )
        ys, xs = np.where(mask)
        return xs.min() + b, ys.min() + b, xs.max() + b, ys.max() + b

    def test_default_plate_is_2to1_rgba(self):
        plate = create_plate_overlay("4J9T7W")
        assert plate.mode == "RGBA"
        assert plate.size == PLATE_OVERLAY_SIZE
        assert plate.size[0] == 2 * plate.size[1]

    def test_text_fits_inside_border(self):
        # Even a wide all-W plate must auto-fit within the border.
        for text in ("4J9T7W", "AA9AFX", "WWWWWWW"):
            plate = create_plate_overlay(text)
            w, h = plate.size
            b = round(h * 0.05) + round(h * 0.03)
            x0, y0, x1, y1 = self._text_ink_bbox(plate)
            assert b <= x0 and x1 <= w - b, f"{text} overflows horizontally"
            assert b <= y0 and y1 <= h - b, f"{text} overflows vertically"

    def test_outside_corner_transparent(self):
        plate = create_plate_overlay("4J9T7W")
        assert plate.getpixel((0, 0))[3] == 0

    def test_custom_style_colors(self):
        style = PlateStyle(background=(255, 255, 255), foreground=(0, 0, 0), border=False)
        plate = create_plate_overlay("ABC123", style=style)
        assert plate.size == PLATE_OVERLAY_SIZE


class TestImageBuilders:
    """Test the in-memory image builder functions."""

    def test_create_background_image_defaults(self):
        img = create_background_image((180, 50, 50))
        assert img.mode == "RGB"
        assert img.size == BACKGROUND_SIZE

    def test_create_background_image_custom_size(self):
        img = create_background_image((10, 20, 30), size=(320, 240))
        assert img.size == (320, 240)

    def test_create_plate_pattern_has_transparency(self):
        img = create_plate_pattern("ABC 123")
        assert img.mode == "RGBA"
        assert img.size == PLATE_SIZE

        # Corners stay transparent; the inset plate body is (near-)opaque.
        assert img.getpixel((0, 0))[3] == 0
        center = (PLATE_SIZE[0] // 2, PLATE_SIZE[1] // 2)
        assert img.getpixel(center)[3] >= 240

    def test_create_plate_pattern_custom_size(self):
        img = create_plate_pattern("HI", size=(120, 40), font_size=16)
        assert img.size == (120, 40)


class TestFileWriters:
    """Test the directory-writing helpers."""

    def test_create_backgrounds_writes_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "backgrounds"
            written = create_backgrounds(out)

            assert len(written) == len(DEFAULT_BACKGROUNDS)
            assert all(p.exists() and p.suffix == ".jpg" for p in written)
            # Files are valid images of the expected size.
            assert Image.open(written[0]).size == BACKGROUND_SIZE

    def test_create_plates_writes_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "overlays"
            written = create_plates(out)

            assert len(written) == len(DEFAULT_PLATES)
            assert all(p.exists() and p.suffix == ".png" for p in written)
            assert Image.open(written[0]).mode == "RGBA"

    def test_create_backgrounds_custom_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bg"
            written = create_backgrounds(out, [("only_car", (1, 2, 3))])
            assert [p.name for p in written] == ["only_car.jpg"]

    def test_create_test_images_creates_both_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            backgrounds, plates = create_test_images(tmp, verbose=False)

            assert (Path(tmp) / "backgrounds").is_dir()
            assert (Path(tmp) / "overlays").is_dir()
            assert len(backgrounds) == len(DEFAULT_BACKGROUNDS)
            assert len(plates) == len(DEFAULT_PLATES)

import tempfile
from pathlib import Path

from PIL import Image

from plateshapez.synthetic import (
    BACKGROUND_SIZE,
    DEFAULT_BACKGROUNDS,
    DEFAULT_PLATES,
    PLATE_SIZE,
    create_background_image,
    create_backgrounds,
    create_plate_pattern,
    create_plates,
    create_test_images,
)


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

import json
import tempfile
from pathlib import Path

from PIL import Image

from plateshapez.ocr import (
    OCRResult,
    crop_plate,
    evaluate_dataset,
    ground_truth_from_overlay,
    normalize_plate,
)


def _make_dataset(tmp: Path) -> Path:
    """Create a tiny dataset (images + labels) mimicking DatasetGenerator output."""
    ds = tmp / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    for name in ("a", "b"):
        Image.new("RGB", (120, 90), "white").save(ds / "images" / f"{name}.png")
        meta = {
            "background": "bg.jpg",
            "overlay": "4J9T7W.png",
            "overlay_position": [30, 25],
            "overlay_size": [60, 40],
        }
        (ds / "labels" / f"{name}.json").write_text(json.dumps(meta))
    return ds


def test_normalize_plate():
    assert normalize_plate("4j9 t7w") == "4J9T7W"
    assert normalize_plate("ABC-123!") == "ABC123"


def test_ground_truth_from_overlay():
    assert ground_truth_from_overlay({"overlay": "4J9T7W.png"}) == "4J9T7W"


def test_crop_plate_band_is_narrower_than_full():
    img = Image.new("RGB", (120, 90))
    meta = {"overlay_position": [30, 25], "overlay_size": [60, 40]}
    full = crop_plate(img, meta, band=None)
    band = crop_plate(img, meta, band=(0.28, 0.74))
    assert full.size == (60, 40)
    assert band.height < full.height and band.width == 60


def test_evaluate_dataset_sorts_success_and_failure():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp))
        # Images are read sorted by name: 'a' correct, 'b' misread.
        engine = _SequenceEngine(["4J9T7W", "XXXXXX"])
        summary = evaluate_dataset(ds, engine, success_threshold=100.0)

        assert summary["total"] == 2
        assert summary["success"] == 1
        assert summary["failure"] == 1
        assert summary["read_accuracy"] == 0.5
        assert Path(summary["results_csv"]).exists()
        assert len(list((ds / "ocr_eval" / "success").glob("*.png"))) == 1
        assert len(list((ds / "ocr_eval" / "failure").glob("*.png"))) == 1
        assert all(isinstance(r, OCRResult) for r in summary["results"])


class _SequenceEngine:
    """Returns successive preset reads, one per image (sorted by name)."""

    name = "sequence"

    def __init__(self, reads: list[str]):
        self._reads = list(reads)
        self._i = 0

    def read(self, image: Image.Image) -> str:
        value = self._reads[self._i]
        self._i += 1
        return value


def test_threshold_allows_near_misses():
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp))
        # One exact, one single-character error (score ~83 for 6 chars).
        engine = _SequenceEngine(["4J9T7W", "4J9T7X"])
        summary = evaluate_dataset(ds, engine, success_threshold=80.0, copy_images=False)
        assert summary["success"] == 2  # both pass the looser threshold

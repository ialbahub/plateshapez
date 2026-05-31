"""OCR evaluation harness for generated plate datasets.

Runs an OCR engine over the composites produced by :class:`DatasetGenerator`,
compares each read against the ground-truth plate text, and sorts the images
into ``success``/``failure`` folders so you can see exactly which perturbations
defeat OCR.

The OCR engine is pluggable via the :class:`OCREngine` protocol. A Tesseract
backend is provided; its Python/!system dependencies are optional and only
imported when the backend is constructed.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

from PIL import Image

from plateshapez.utils.io import iter_images


def normalize_plate(text: str) -> str:
    """Upper-case ``text`` and keep only alphanumerics for fair comparison."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def character_score(expected: str, predicted: str) -> float:
    """Character-level similarity in ``[0, 100]`` (100 == identical)."""
    from rapidfuzz import fuzz

    if not expected and not predicted:
        return 100.0
    return float(fuzz.ratio(expected, predicted))


@runtime_checkable
class OCREngine(Protocol):
    """Minimal OCR backend: turn an image of a plate into a text string."""

    name: str

    def read(self, image: Image.Image) -> str: ...


class TesseractEngine:
    """OCR backend using Tesseract via ``pytesseract``.

    Requires the ``tesseract`` system binary and the ``pytesseract`` package
    (install the ``ocr`` extra). Configured for a single line of upper-case
    plate characters.
    """

    name = "tesseract"

    def __init__(
        self,
        psm: int = 7,
        whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ) -> None:
        import pytesseract  # optional dependency, imported on use

        self._pt = pytesseract
        self._config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"

    def read(self, image: Image.Image) -> str:
        return str(self._pt.image_to_string(image, config=self._config)).strip()


class EasyOCREngine:
    """OCR backend using EasyOCR (deep learning, closer to real ALPR).

    Requires ``easyocr`` (pulls PyTorch). Downloads its models on first use.
    """

    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        import easyocr  # optional dependency, imported on use

        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)

    def read(self, image: Image.Image) -> str:
        import numpy as np

        results = self._reader.readtext(np.asarray(image.convert("RGB")), detail=0)
        return " ".join(results)


class PaddleOCREngine:
    """OCR backend using PaddleOCR (modern OCR, strong on text in the wild).

    Requires ``paddleocr`` and ``paddlepaddle``. Downloads models on first use.
    """

    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # optional dependency, imported on use

        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read(self, image: Image.Image) -> str:
        import numpy as np

        result = self._ocr.ocr(np.asarray(image.convert("RGB")), cls=True)
        lines: list[str] = []
        for page in result or []:
            for entry in page or []:
                # entry = [box, (text, confidence)]
                if len(entry) >= 2 and entry[1]:
                    lines.append(str(entry[1][0]))
        return " ".join(lines)


@dataclass
class OCRResult:
    """Outcome of reading a single generated image."""

    image: str
    expected: str
    predicted: str
    score: float
    success: bool


def ground_truth_from_overlay(metadata: dict) -> str:
    """Default ground truth: the overlay file stem (e.g. ``4J9T7W.png``)."""
    return normalize_plate(Path(metadata["overlay"]).stem)


def crop_plate(
    image: Image.Image,
    metadata: dict,
    band: tuple[float, float] | None = (0.28, 0.74),
) -> Image.Image:
    """Crop ``image`` to the plate region, optionally to the character band.

    ``band`` is a ``(top, bottom)`` fraction of the plate height isolating the
    main characters (plates put the state name / slogan above and below). Pass
    ``None`` to keep the whole plate region.
    """
    bx, by = metadata["overlay_position"]
    ow, oh = metadata["overlay_size"]
    if band is not None:
        top = by + int(oh * band[0])
        bottom = by + int(oh * band[1])
    else:
        top, bottom = by, by + oh
    return image.crop((bx, top, bx + ow, bottom))


def evaluate_dataset(
    dataset_dir: str | Path,
    engine: OCREngine,
    *,
    ground_truth: Callable[[dict], str] = ground_truth_from_overlay,
    success_threshold: float = 100.0,
    band: tuple[float, float] | None = (0.28, 0.74),
    out_dir: str | Path | None = None,
    copy_images: bool = True,
) -> dict:
    """Run OCR over a generated dataset and split results by read success.

    For every image in ``<dataset_dir>/images`` with a matching label, the plate
    is cropped, read by ``engine``, normalised and compared to the ground truth.
    A read counts as success when its character score meets ``success_threshold``
    (100 == exact match). Images are copied into ``<out_dir>/success`` or
    ``<out_dir>/failure``, a ``results.csv`` is written, and a summary dict
    (with per-engine accuracy) is returned.

    Args:
        dataset_dir: A dataset produced by :class:`DatasetGenerator`.
        engine: Any :class:`OCREngine`.
        ground_truth: Maps a label's metadata to the expected plate string.
        success_threshold: Minimum character score (0-100) to count as success.
        band: Character-band crop fraction (see :func:`crop_plate`).
        out_dir: Where to write ``success``/``failure`` folders and the CSV.
        copy_images: Copy each image into its outcome folder when ``True``.

    Returns:
        Summary dict: counts, read accuracy, mean score, and the results list.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    out_dir = Path(out_dir) if out_dir else dataset_dir / "ocr_eval"
    success_dir = out_dir / "success"
    failure_dir = out_dir / "failure"
    if copy_images:
        success_dir.mkdir(parents=True, exist_ok=True)
        failure_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[OCRResult] = []
    for img_path in sorted(iter_images(images_dir, [".png"])):
        label_path = labels_dir / f"{img_path.stem}.json"
        if not label_path.exists():
            continue
        metadata = json.loads(label_path.read_text())
        image = Image.open(img_path).convert("RGB")
        crop = crop_plate(image, metadata, band)

        expected = normalize_plate(ground_truth(metadata))
        predicted = normalize_plate(engine.read(crop))
        score = character_score(expected, predicted)
        success = score >= success_threshold
        results.append(OCRResult(img_path.name, expected, predicted, score, success))

        if copy_images:
            shutil.copy(img_path, (success_dir if success else failure_dir) / img_path.name)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "expected", "predicted", "score", "success"]
        )
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    total = len(results)
    successes = sum(r.success for r in results)
    summary = {
        "engine": engine.name,
        "total": total,
        "success": successes,
        "failure": total - successes,
        "read_accuracy": (successes / total) if total else 0.0,
        "mean_score": (sum(r.score for r in results) / total) if total else 0.0,
        "success_threshold": success_threshold,
        "results_csv": str(csv_path),
        "results": results,
    }
    return summary


def sweep_intensity(
    bg_dir: str | Path,
    overlay_dir: str | Path,
    engine: OCREngine,
    *,
    perturbation: str,
    param: str,
    values: Sequence[float],
    work_dir: str | Path,
    base_params: dict | None = None,
    n_variants: int = 10,
    random_seed: int = 2025,
    ground_truth: Callable[[dict], str] = ground_truth_from_overlay,
    band: tuple[float, float] | None = (0.28, 0.74),
    success_threshold: float = 100.0,
) -> list[dict]:
    """Sweep one perturbation parameter and measure OCR accuracy at each level.

    For every value in ``values`` a dataset is generated (with ``perturbation``
    set to that value, all else fixed and the seed held constant so levels are
    comparable), then read by ``engine``. Returns one row per level with the
    parameter value, read accuracy and mean character score — i.e. the
    perturbation's "breaking curve" for that OCR engine.

    Args:
        bg_dir/overlay_dir: Inputs for the generator.
        engine: OCR backend to evaluate with.
        perturbation: Perturbation name to sweep (e.g. ``"noise"``).
        param: The parameter of that perturbation to vary (e.g. ``"intensity"``).
        values: Parameter values to test, in order.
        work_dir: Scratch directory for the per-level datasets.
        base_params: Other fixed params for the perturbation.
        n_variants: Variants per plate per level.
    """
    # Imported here to avoid a circular import at module load time.
    from plateshapez.pipeline import DatasetGenerator

    work_dir = Path(work_dir)
    rows: list[dict] = []
    for value in values:
        params = {**(base_params or {}), param: value}
        level_dir = work_dir / f"{perturbation}_{param}_{value}"
        DatasetGenerator(
            bg_dir=bg_dir,
            overlay_dir=overlay_dir,
            out_dir=level_dir,
            perturbations=[{"name": perturbation, "params": params}],
            random_seed=random_seed,
            save_perturbation_layer=False,
        ).run(n_variants=n_variants)

        summary = evaluate_dataset(
            level_dir,
            engine,
            ground_truth=ground_truth,
            band=band,
            success_threshold=success_threshold,
            copy_images=False,
        )
        rows.append(
            {
                "value": value,
                "read_accuracy": summary["read_accuracy"],
                "mean_score": summary["mean_score"],
                "total": summary["total"],
            }
        )
    return rows

"""Synthetic image generation for demos and testing.

Reusable helpers that build synthetic vehicle background images and license
plate *pattern* overlays. This logic previously lived inside the demo workflow
script (``examples/demo_full_workflow.py``) as a single ``create_test_images``
function; it now lives here so background and plate-pattern creation are split
into separate, independently usable, and unit-testable functions.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Default synthetic content used by the demo workflow.
DEFAULT_BACKGROUNDS: list[tuple[str, tuple[int, int, int]]] = [
    ("red_car", (180, 50, 50)),
    ("blue_car", (50, 100, 180)),
    ("gray_car", (120, 120, 120)),
]

DEFAULT_PLATES: list[tuple[str, str]] = [
    ("plate_abc123", "ABC 123"),
    ("plate_xyz789", "XYZ 789"),
    ("plate_test01", "TEST 01"),
]

# Default canvas sizes for generated assets.
BACKGROUND_SIZE: tuple[int, int] = (800, 600)
PLATE_SIZE: tuple[int, int] = (200, 60)

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a bold TrueType font, falling back to PIL's default."""
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def create_background_image(
    color: tuple[int, int, int],
    size: tuple[int, int] = BACKGROUND_SIZE,
) -> Image.Image:
    """Create a synthetic car background image.

    Args:
        color: RGB fill color for the car body, roof, and windows frame.
        size: Output image size as ``(width, height)``.

    Returns:
        An RGB :class:`PIL.Image.Image` depicting a simple car illustration.
    """
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Car body
    draw.rectangle((200, 300, 600, 450), fill=color, outline=(0, 0, 0), width=3)
    # Car roof
    draw.rectangle((250, 250, 550, 300), fill=color, outline=(0, 0, 0), width=3)
    # Wheels
    draw.ellipse([220, 430, 280, 490], fill=(40, 40, 40), outline=(0, 0, 0), width=2)
    draw.ellipse([520, 430, 580, 490], fill=(40, 40, 40), outline=(0, 0, 0), width=2)
    # Windows
    draw.rectangle((270, 260, 530, 290), fill=(150, 200, 255), outline=(0, 0, 0), width=2)

    return img


def create_plate_pattern(
    text: str,
    size: tuple[int, int] = PLATE_SIZE,
    font_size: int = 24,
) -> Image.Image:
    """Create a synthetic license plate pattern overlay with transparency.

    Args:
        text: Plate text to render, centered on the plate.
        size: Output image size as ``(width, height)``.
        font_size: Font size used to render ``text``.

    Returns:
        An RGBA :class:`PIL.Image.Image` with a transparent background and an
        opaque white plate bearing the given text.
    """
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw plate background (inset by 5px on each edge).
    draw.rectangle(
        (5, 5, width - 5, height - 5),
        fill=(255, 255, 255, 240),
        outline=(0, 0, 0, 255),
        width=2,
    )

    font = _load_font(font_size)

    # Center the text within the plate.
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

    return img


def create_backgrounds(
    out_dir: str | Path,
    backgrounds: list[tuple[str, tuple[int, int, int]]] | None = None,
    *,
    verbose: bool = False,
) -> list[Path]:
    """Render background car images to ``out_dir`` as JPEGs.

    Args:
        out_dir: Directory to write the background images into (created if
            necessary).
        backgrounds: Iterable of ``(name, color)`` pairs. Defaults to
            :data:`DEFAULT_BACKGROUNDS`.
        verbose: If ``True``, print a line for each created file.

    Returns:
        The list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    items = backgrounds if backgrounds is not None else DEFAULT_BACKGROUNDS

    written: list[Path] = []
    for name, color in items:
        img = create_background_image(color)
        dest = out_path / f"{name}.jpg"
        img.save(dest, "JPEG", quality=95)
        written.append(dest)
        if verbose:
            print(f"  ✓ Created background: {dest.name}")

    return written


def create_plates(
    out_dir: str | Path,
    plates: list[tuple[str, str]] | None = None,
    *,
    verbose: bool = False,
) -> list[Path]:
    """Render license plate pattern overlays to ``out_dir`` as PNGs.

    Args:
        out_dir: Directory to write the plate overlays into (created if
            necessary).
        plates: Iterable of ``(name, text)`` pairs. Defaults to
            :data:`DEFAULT_PLATES`.
        verbose: If ``True``, print a line for each created file.

    Returns:
        The list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    items = plates if plates is not None else DEFAULT_PLATES

    written: list[Path] = []
    for name, text in items:
        img = create_plate_pattern(text)
        dest = out_path / f"{name}.png"
        img.save(dest, "PNG")
        written.append(dest)
        if verbose:
            print(f"  ✓ Created overlay: {dest.name}")

    return written


def create_test_images(
    base_dir: str | Path = "dataset/demo",
    *,
    verbose: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Create the full set of synthetic demo assets.

    Writes background images to ``<base_dir>/backgrounds`` and plate pattern
    overlays to ``<base_dir>/overlays``.

    Args:
        base_dir: Base directory under which ``backgrounds`` and ``overlays``
            subdirectories are created.
        verbose: If ``True``, print progress for each created file.

    Returns:
        A ``(background_paths, plate_paths)`` tuple of written file paths.
    """
    base = Path(base_dir)
    if verbose:
        print("🎨 Creating test images...")

    backgrounds = create_backgrounds(base / "backgrounds", verbose=verbose)
    plates = create_plates(base / "overlays", verbose=verbose)
    return backgrounds, plates

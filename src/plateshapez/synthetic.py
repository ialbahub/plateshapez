"""Synthetic image generation for demos and testing.

Reusable helpers that build synthetic vehicle background images and license
plate *pattern* overlays. This logic previously lived inside the demo workflow
script (``examples/demo_full_workflow.py``) as a single ``create_test_images``
function; it now lives here so background and plate-pattern creation are split
into separate, independently usable, and unit-testable functions.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# Real US/Michigan plates are 12 in x 6 in -> a 2:1 aspect ratio.
PLATE_OVERLAY_SIZE: tuple[int, int] = (1200, 600)


@dataclass
class PlateStyle:
    """Colours and labels for a rendered license plate overlay."""

    background: tuple[int, int, int] = (18, 18, 20)
    foreground: tuple[int, int, int] = (235, 235, 235)
    header: str = "MICHIGAN"
    footer: str = "GREAT LAKE STATE"
    sticker: str = "27"
    sticker_sub: str = "AR"
    sticker_color: tuple[int, int, int] = (235, 130, 28)
    border: bool = True


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> tuple[int, int]:
    """Width/height of ``text`` rendered with ``font``."""
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _fit_font(
    draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int, cap: int
) -> ImageFont.FreeTypeFont:
    """Largest bold font for which ``text`` fits inside ``max_w`` x ``max_h``."""
    best = 8
    size = 8
    while size <= cap:
        font = ImageFont.truetype(_FONT_PATH, size)
        w, h = _text_size(draw, text, font)
        if w <= max_w and h <= max_h:
            best = size
            size += 2
        else:
            break
    return ImageFont.truetype(_FONT_PATH, best)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    cx: float,
    band_top: float,
    band_h: float,
    fill: tuple[int, int, int],
) -> None:
    """Draw ``text`` horizontally centred on ``cx`` and vertically within a band."""
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = cx - w / 2 - box[0]
    y = band_top + (band_h - h) / 2 - box[1]
    draw.text((x, y), text, font=font, fill=fill)


def create_plate_overlay(
    text: str,
    *,
    size: tuple[int, int] = PLATE_OVERLAY_SIZE,
    style: PlateStyle | None = None,
) -> Image.Image:
    """Render a license plate overlay with an organised, auto-fit layout.

    The plate content is laid out in three balanced vertical bands inside the
    border — a header (state name), the main characters, and a footer (slogan) —
    with the main characters auto-sized to fill the available width without
    overflowing. A registration sticker is placed in the bottom-right corner.

    Args:
        text: The plate characters (e.g. ``"4J9T7W"``).
        size: Output size as ``(width, height)``; defaults to a 2:1 plate.
        style: Colours and labels; defaults to a black Michigan plate.

    Returns:
        An RGBA :class:`PIL.Image.Image` with a transparent outside corner.
    """
    style = style or PlateStyle()
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded plate body.
    radius = round(height * 0.07)
    margin = round(height * 0.01)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (margin, margin, width - margin, height - margin), radius=radius, fill=255
    )
    img.paste(Image.new("RGB", size, style.background), (0, 0), mask)

    fg = (*style.foreground, 255)

    # Inner content rectangle (inside the embossed border).
    pad = round(height * 0.05)
    left, top = pad, pad
    right, bottom = width - pad, height - pad
    content_w = right - left
    content_h = bottom - top

    if style.border:
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=round(radius * 0.7),
            outline=fg,
            width=max(2, round(height * 0.007)),
        )
        inset = round(height * 0.03)
        left, top, right, bottom = left + inset, top + inset, right - inset, bottom - inset
        content_w = right - left
        content_h = bottom - top

    # Three balanced bands: header / main characters / footer.
    header_h = round(content_h * 0.20)
    footer_h = round(content_h * 0.18)
    gap = round(content_h * 0.04)
    main_h = content_h - header_h - footer_h - 2 * gap
    cx = (left + right) / 2

    header_font = _fit_font(draw, style.header, int(content_w * 0.85), header_h, cap=header_h)
    _draw_centered(draw, style.header, header_font, cx, top, header_h, style.foreground)

    main_top = top + header_h + gap
    main_font = _fit_font(draw, text, int(content_w * 0.92), int(main_h * 0.96), cap=main_h)
    # Subtle embossed shadow then the face.
    shadow_off = max(2, round(height * 0.008))
    box = draw.textbbox((0, 0), text, font=main_font)
    mw, mh = box[2] - box[0], box[3] - box[1]
    mx = cx - mw / 2 - box[0]
    my = main_top + (main_h - mh) / 2 - box[1]
    bgc = style.background
    shadow = (min(255, bgc[0] + 42), min(255, bgc[1] + 42), min(255, bgc[2] + 42), 255)
    draw.text((mx + shadow_off, my + shadow_off), text, font=main_font, fill=shadow)
    draw.text((mx, my), text, font=main_font, fill=fg)

    footer_top = bottom - footer_h
    footer_font = _fit_font(draw, style.footer, int(content_w * 0.70), footer_h, cap=footer_h)
    _draw_centered(draw, style.footer, footer_font, cx, footer_top, footer_h, style.foreground)

    # Registration sticker in the bottom-right corner, inside the border.
    if style.sticker:
        sw = round(content_w * 0.09)
        sh = round(content_h * 0.20)
        sx, sy = right - sw, bottom - sh
        sticker_fill = (*style.sticker_color, 255)
        draw.rounded_rectangle((sx, sy, right, bottom), radius=round(sh * 0.12), fill=sticker_fill)
        ink = (20, 20, 20)
        num_font = _fit_font(draw, style.sticker, round(sw * 0.62), round(sh * 0.62), cap=sh)
        _draw_centered(draw, style.sticker, num_font, sx + sw * 0.38, sy, sh, ink)
        if style.sticker_sub:
            sub_font = _fit_font(draw, "M", round(sw * 0.3), round(sh * 0.3), cap=round(sh * 0.32))
            for i, ch in enumerate(style.sticker_sub[:2]):
                _draw_centered(
                    draw, ch, sub_font, sx + sw * 0.82, sy + sh * (0.15 + 0.38 * i), sh * 0.3, ink
                )

    return img


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


def create_vehicle_background(
    size: tuple[int, int] = (1600, 1100),
    body_color: tuple[int, int, int] = (54, 58, 66),
) -> Image.Image:
    """Render a realistic dark vehicle rear for seating a license plate.

    Unlike :func:`create_background_image` (a light clipart car), this draws a
    car's rear panel — body with a vertical sheen, a bumper seam, a recessed
    plate-mount, a round emblem and tail-light hints — so a plate composited at
    the centre looks mounted on a vehicle.

    Args:
        size: Output image size as ``(width, height)``.
        body_color: Base RGB colour of the vehicle body.

    Returns:
        An RGB :class:`PIL.Image.Image` of the vehicle rear.
    """
    width, height = size
    img = Image.new("RGB", size, body_color)
    draw = ImageDraw.Draw(img)

    r, g, b = body_color
    # Vertical sheen: brighter band across the upper-middle of the panel.
    for y in range(height):
        t = y / height
        sheen = int(38 * max(0.0, 1.0 - abs(0.42 - t) * 3.2))
        draw.line([(0, y), (width, y)], fill=(r + sheen, g + sheen, b + sheen))

    darker = (max(0, r - 24), max(0, g - 24), max(0, b - 24))
    lighter = (min(255, r + 26), min(255, g + 26), min(255, b + 26))

    # Round emblem near the top centre.
    er = round(height * 0.07)
    ecx, ecy = width // 2, round(height * 0.16)
    draw.ellipse((ecx - er, ecy - er, ecx + er, ecy + er), fill=darker, outline=lighter, width=3)
    draw.ellipse(
        (ecx - er // 2, ecy - er // 2, ecx + er // 2, ecy + er // 2), outline=lighter, width=2
    )

    # Bumper seam line below the plate area.
    seam_y = round(height * 0.72)
    draw.line([(0, seam_y), (width, seam_y)], fill=darker, width=max(3, round(height * 0.006)))

    # Recessed plate mount (slightly larger than a 2:1 plate, centred).
    mw, mh = round(width * 0.46), round(height * 0.34)
    mx, my = (width - mw) // 2, round(height * 0.30)
    draw.rounded_rectangle(
        (mx - 12, my - 12, mx + mw + 12, my + mh + 12),
        radius=18,
        fill=darker,
        outline=(0, 0, 0),
        width=3,
    )

    # Tail-light hints at the sides.
    for cx in (round(width * 0.08), round(width * 0.92)):
        draw.rounded_rectangle(
            (cx - 60, seam_y - 40, cx + 60, seam_y + 30),
            radius=16,
            fill=(120, 40, 36),
            outline=darker,
            width=2,
        )

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

from PIL import Image, ImageDraw

from .base import Perturbation, register


@register
class GridPerturbation(Perturbation):
    """Overlay a fine mesh/grid across the plate.

    A semi-transparent grid disrupts the high-frequency structure that ALPR
    plate **detectors** key on (it reliably defeats Fast-ALPR / PP-OCR detection)
    while a human still reads the characters through the mesh. Everything is
    configurable:

    - ``spacing``: pixels between grid lines (smaller = denser).
    - ``width``: line thickness in pixels.
    - ``alpha``: line opacity 0-255 (lower = fainter / more readable).
    - ``color``: RGB line colour (default light grey).
    - ``diagonal``: if true, draw a diagonal mesh instead of axis-aligned.
    """

    name = "grid"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        spacing = max(2, int(self.params.get("spacing", 22)))
        width = int(self.params.get("width", 3))
        alpha = int(self.params.get("alpha", 120))
        color = self.params.get("color", [245, 245, 245])
        diagonal = bool(self.params.get("diagonal", False))
        fill = (int(color[0]), int(color[1]), int(color[2]), alpha)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if diagonal:
            for off in range(0, w + h, spacing):
                draw.line((x + off, y, x, y + off), fill=fill, width=width)
                draw.line((x + w - off, y, x + w, y + off), fill=fill, width=width)
        else:
            for gx in range(x, x + w, spacing):
                draw.line((gx, y, gx, y + h), fill=fill, width=width)
            for gy in range(y, y + h, spacing):
                draw.line((x, gy, x + w, gy), fill=fill, width=width)

        mode = img.mode
        base = img.convert("RGBA")
        base.alpha_composite(overlay)
        return base.convert(mode)

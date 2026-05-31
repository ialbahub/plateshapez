import random

import numpy as np
from PIL import Image, ImageDraw

from .base import Perturbation, register


@register
class ShapesPerturbation(Perturbation):
    """Random geometric shapes perturbation for adversarial occlusion.

    Adds random rectangles, ellipses, and triangles to simulate physical occlusion
    or adversarial patches on license plates.

    Shape parameters such as size, position, and orientation are randomly sampled
    within the specified region for each shape type. The number and properties of
    shapes are determined stochastically to maximize diversity and realism.

    By default the shape colour is chosen automatically to contrast with the
    plate, so occlusions stay visible on dark plates (white shapes) as well as
    light plates (black shapes). Pass ``color`` to force a specific colour
    (``"black"``, ``"white"``, or an RGB/RGBA tuple).
    """

    name = "shapes"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        # Cached once so the same colour is reused when the perturbation is
        # re-applied (e.g. onto the neutral canvas used for the isolated layer).
        self._color: tuple[int, int, int, int] | None = None

    def _resolve_color(
        self, img: Image.Image, region: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Pick the fill colour, auto-contrasting with the plate when unset."""
        if self._color is not None:
            return self._color

        spec = self.params.get("color", "auto")
        if spec == "white":
            color = (255, 255, 255, 255)
        elif isinstance(spec, (list, tuple)):
            vals = tuple(int(v) for v in spec)
            color = vals if len(vals) == 4 else (*vals, 255)  # type: ignore[assignment]
        elif spec == "auto":
            x, y, w, h = region
            crop = img.convert("L").crop((x, y, x + w, y + h))
            mean = float(np.asarray(crop).mean()) if crop.width and crop.height else 255.0
            # Dark plate -> white shapes; light plate -> black shapes.
            color = (255, 255, 255, 255) if mean < 128 else (0, 0, 0, 255)
        else:  # "black" or anything unrecognised
            color = (0, 0, 0, 255)

        self._color = color
        return color

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        fill = self._resolve_color(img, region)
        draw = ImageDraw.Draw(img, "RGBA")
        num_shapes: int = int(self.params.get("num_shapes", 15))
        min_size: int = int(self.params.get("min_size", 2))
        max_size: int = int(self.params.get("max_size", 10))

        for _ in range(num_shapes):
            sx = random.randint(x, x + w)
            sy = random.randint(y, y + h)
            size = random.randint(min_size, max_size)
            shape_type = random.choice(["rect", "ellipse", "triangle"])

            if shape_type == "rect":
                draw.rectangle((sx, sy, sx + size, sy + size), fill=fill)
            elif shape_type == "ellipse":
                draw.ellipse((sx, sy, sx + size, sy + size), fill=fill)
            else:
                draw.polygon(
                    [
                        (sx, sy),
                        (sx + random.randint(-size, size), sy + size),
                        (sx + size, sy + random.randint(-size, size)),
                    ],
                    fill=fill,
                )

        return img

import random

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
    """

    name = "shapes"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        draw = ImageDraw.Draw(img, "RGBA")
        num_shapes: int = int(self.params.get("num_shapes", 15))
        min_size: int = int(self.params.get("min_size", 2))
        max_size: int = int(self.params.get("max_size", 10))

        for _ in range(num_shapes):
            shape_type = random.choice(["rect", "ellipse", "triangle"])
            size = random.randint(min_size, max_size)
            # Cap size so a whole shape can always fit within the region (small plates).
            size = min(size, w // 2, h // 2)

            if shape_type == "triangle":
                # Triangle vertices spread +/- size around the anchor, so inset the
                # anchor by `size` on every edge to keep the full shape in bounds.
                sx = random.randint(x + size, x + w - size)
                sy = random.randint(y + size, y + h - size)
                draw.polygon(
                    [
                        (sx, sy),
                        (sx + random.randint(-size, size), sy + size),
                        (sx + size, sy + random.randint(-size, size)),
                    ],
                    fill=(0, 0, 0, 255),
                )
            else:
                # Rect/ellipse extend by `size` to the right and down only.
                sx = random.randint(x, x + w - size)
                sy = random.randint(y, y + h - size)
                if shape_type == "rect":
                    draw.rectangle((sx, sy, sx + size, sy + size), fill=(0, 0, 0, 255))
                else:
                    draw.ellipse((sx, sy, sx + size, sy + size), fill=(0, 0, 0, 255))

        return img

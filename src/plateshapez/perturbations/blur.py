import numpy as np
from PIL import Image, ImageFilter

from .base import Perturbation, register


@register
class BlurPerturbation(Perturbation):
    """Gaussian or directional motion blur over the plate region.

    Simulates out-of-focus captures (``type: gaussian``) or camera/vehicle
    motion (``type: motion``), both common reasons real ALPR reads degrade.
    """

    name = "blur"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        blur_type: str = self.params.get("type", "gaussian")
        radius: float = float(self.params.get("radius", 2.0))

        crop = img.crop((x, y, x + w, y + h))
        if blur_type == "motion":
            crop = self._motion_blur(crop, radius, float(self.params.get("angle", 0.0)))
        else:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=radius))

        img.paste(crop, (x, y))
        return img

    def _motion_blur(self, crop: Image.Image, radius: float, angle: float) -> Image.Image:
        """Apply a 1-D directional blur by averaging shifted copies."""
        length = max(1, int(radius * 2) + 1)
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        acc = np.zeros_like(arr)
        offsets = range(-(length // 2), length // 2 + 1)
        for t in offsets:
            shifted = np.roll(arr, (int(round(t * dy)), int(round(t * dx))), axis=(0, 1))
            acc += shifted
        acc /= len(list(offsets))
        return Image.fromarray(acc.clip(0, 255).astype(np.uint8))

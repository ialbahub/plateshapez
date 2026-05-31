import numpy as np
from PIL import Image

from .base import Perturbation, register


@register
class GlarePerturbation(Perturbation):
    """Bright specular glare / hot-spot over the plate region.

    Simulates sunlight or headlight reflection on a reflective plate, which
    washes out characters and is a common real-world OCR failure mode. A soft
    elliptical highlight is added with screen-style blending.
    """

    name = "glare"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        intensity: float = float(self.params.get("intensity", 0.6))  # 0..1 peak strength
        cx: float = float(self.params.get("cx", 0.5))  # centre, fraction of region
        cy: float = float(self.params.get("cy", 0.35))
        spread: float = float(self.params.get("spread", 0.35))  # radius, fraction of region

        crop = np.asarray(img.crop((x, y, x + w, y + h)).convert("RGB"), dtype=np.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx = (xx - cx * w) / (spread * w + 1e-6)
        gy = (yy - cy * h) / (spread * h + 1e-6)
        falloff = np.exp(-(gx**2 + gy**2)) * intensity  # 0..intensity gaussian hot-spot

        # Screen blend toward white: out = 255 - (255-img)*(1-glare)
        glare = falloff[..., None]
        out = 255.0 - (255.0 - crop) * (1.0 - glare)

        img.paste(Image.fromarray(out.clip(0, 255).astype(np.uint8)), (x, y))
        return img

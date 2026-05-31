import cv2
import numpy as np
from PIL import Image

from .base import Perturbation, register


@register
class PerspectivePerturbation(Perturbation):
    """Perspective (viewing-angle) warp of the plate region.

    Simulates a plate photographed off-axis (tilted left/right or up/down).
    The plate's rectangle is mapped to a trapezoid via a homography, which is a
    major source of real ALPR difficulty. ``strength`` is the maximum corner
    displacement as a fraction of the region size; ``tilt`` (``"h"``/``"v"``)
    chooses horizontal or vertical skew.
    """

    name = "perspective"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        strength: float = float(self.params.get("strength", 0.18))
        tilt: str = self.params.get("tilt", "h")

        arr = np.array(img)
        crop = arr[y : y + h, x : x + w].copy()

        dx, dy = strength * w, strength * h
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        if tilt == "v":
            # Top edge shrinks inward -> tilted forward/back.
            dst = np.array([[dx, dy], [w - dx, dy], [w, h], [0, h]], dtype=np.float32)
        else:
            # Right edge shrinks inward -> rotated about a vertical axis.
            dst = np.array([[0, 0], [w - dx, dy], [w - dx, h - dy], [0, h]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        # Fill the revealed corners by replicating the edge pixels (the plate
        # border / dark surround) rather than mirroring the whole plate, which
        # would tile a confusing second copy into the frame.
        warped = cv2.warpPerspective(
            crop,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        arr[y : y + h, x : x + w] = warped
        return Image.fromarray(arr)

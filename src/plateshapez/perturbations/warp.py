import cv2
import numpy as np
from PIL import Image

from .base import Perturbation, register


@register
class WarpPerturbation(Perturbation):
    """Mild geometric warping perturbation."""

    name = "warp"

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        """Apply a mild geometric warp."""
        x, y, w, h = region
        intensity: float = float(self.params.get("intensity", 5.0))
        frequency: float = float(self.params.get("frequency", 20.0))
        scope: str = self.params.get("scope", "region")

        arr: np.ndarray = np.array(img)
        img_h, img_w, _ = arr.shape

        if scope == "global":
            # Apply warp to entire image
            grid_x, grid_y = np.meshgrid(np.arange(img_w), np.arange(img_h))
            map_x = grid_x + np.sin(grid_y / frequency) * intensity
            map_y = grid_y + np.cos(map_x / frequency) * intensity
            remap_x = np.clip(map_x, 0, img_w - 1).astype(np.float32)
            remap_y = np.clip(map_y, 0, img_h - 1).astype(np.float32)

            remap: np.ndarray = cv2.remap(arr, remap_x, remap_y, interpolation=cv2.INTER_LINEAR)
            return Image.fromarray(remap)
        else:
            # Apply warp only to region
            region_arr = arr[y : y + h, x : x + w].copy()

            # Create displacement maps for region only
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = grid_x + np.sin(grid_y / frequency) * intensity
            map_y = grid_y + np.cos(map_x / frequency) * intensity
            remap_x = np.clip(map_x, 0, w - 1).astype(np.float32)
            remap_y = np.clip(map_y, 0, h - 1).astype(np.float32)

            warped_region = cv2.remap(region_arr, remap_x, remap_y, interpolation=cv2.INTER_LINEAR)
            arr[y : y + h, x : x + w] = warped_region

            return Image.fromarray(arr)

from __future__ import annotations

"""RGB preprocessing for deterministic parasite detection."""

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, disk
from skimage.util import img_as_float


def _as_float_rgb(img_rgb):
    arr = np.asarray(img_rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("image must be an RGB array with shape (H, W, 3)")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        img = arr.astype(np.float32, copy=False)
        img = img / 255.0 if img.size and np.nanmax(img) > 1.0 else img
        return np.clip(img, 0.0, 1.0)
    return img_as_float(arr).astype(np.float32, copy=False)


def preprocess_rgb(img_rgb: np.ndarray, config: dict | None = None) -> dict:
    config = config or {}
    img = _as_float_rgb(img_rgb)
    gray = rgb2gray(img)
    invalid = gray < float(config.get("invalid_gray_threshold", 0.05))
    invalid_radius = int(config.get("invalid_dilation_radius", 2))
    invalid = binary_dilation(invalid, disk(invalid_radius)) if invalid_radius > 0 else invalid

    if config.get("illumination_correction", True):
        background = gaussian(img, sigma=float(config.get("background_sigma", 20)), channel_axis=-1)
        img_corr = rescale_intensity(img / (background + 1e-6), out_range=(0.0, 1.0)).astype(np.float32, copy=False)
    else:
        img_corr = img.copy()

    return {"img": img, "img_corr": img_corr, "invalid_mask": invalid.astype(bool), "gray": gray}

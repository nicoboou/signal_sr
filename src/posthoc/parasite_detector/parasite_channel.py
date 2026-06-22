"""Purple-stain enhancement channels."""

import numpy as np
from skimage.color import rgb2hsv
from skimage.morphology import disk, white_tophat


def compute_parasite_channel(img_corr: np.ndarray, cell_mask: np.ndarray) -> dict:
    img = np.clip(img_corr, 1e-6, 1.0)
    od = -np.log(img)
    purple = od[..., 1] - 0.5 * (od[..., 0] + od[..., 2])

    vals = purple[cell_mask.astype(bool)]
    if vals.size:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        purple_z = (purple - med) / (1.4826 * mad)
    else:
        purple_z = np.zeros_like(purple)

    hsv = rgb2hsv(img)
    return {"purple": purple, "purple_z": purple_z, "saturation": hsv[..., 1], "darkness": 1.0 - hsv[..., 2]}


def enhance_small_purple_objects(Pz: np.ndarray, D: float, config: dict) -> np.ndarray:
    radius = max(1, int(float(config["tophat_radius_frac"]) * D))
    footprint = disk(radius)
    try:
        return white_tophat(Pz, footprint=footprint)
    except TypeError:
        return white_tophat(Pz, selem=footprint)

from __future__ import annotations

"""Classical red blood cell segmentation."""

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, disk, remove_small_objects


def _otsu_or_midpoint(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.inf
    if float(values.min()) == float(values.max()):
        return float(values.min())
    return float(threshold_otsu(values))


def equivalent_diameter_from_area(area):
    return float(np.sqrt(4.0 * float(area) / np.pi))


def segment_cells(img_corr: np.ndarray, invalid_mask: np.ndarray, config: dict) -> list[np.ndarray]:
    valid = ~invalid_mask.astype(bool)
    if not bool(valid.any()):
        return []

    img = np.clip(img_corr, 1e-6, 1.0)
    hsv = rgb2hsv(img)
    sat = hsv[..., 1]
    od_sum = -np.log(img).sum(axis=-1)

    mask_sat = sat > _otsu_or_midpoint(sat[valid])
    mask_od = od_sum > _otsu_or_midpoint(od_sum[valid])
    cell_mask = (mask_sat | mask_od) & valid
    cell_mask = remove_small_objects(cell_mask, min_size=int(config["min_cell_area_px"]))
    cell_mask = binary_closing(cell_mask, disk(3))
    cell_mask = binary_fill_holes(cell_mask).astype(bool)

    labels = label(cell_mask)
    image_area = cell_mask.size
    area_min = max(int(config["min_cell_area_px"]), float(config["cell_area_min_frac"]) * image_area)
    area_max = float(config["cell_area_max_frac"]) * image_area
    masks = []
    for region in regionprops(labels):
        if area_min <= region.area <= area_max and region.eccentricity <= config["cell_eccentricity_max"] and region.solidity >= config["cell_solidity_min"]:
            masks.append(labels == region.label)

    if masks:
        return sorted(masks, key=lambda mask: int(mask.sum()), reverse=True)

    regions = sorted(regionprops(labels), key=lambda region: region.area, reverse=True)
    return [labels == regions[0].label] if regions and regions[0].area >= config["min_cell_area_px"] else []


def make_inner_cell_mask(cell_mask: np.ndarray, D: float, config: dict) -> dict:
    erosion_radius = max(1, int(float(config["inner_erosion_frac"]) * D))
    inner = binary_erosion(cell_mask, disk(erosion_radius))
    inner = inner if bool(inner.any()) else cell_mask.astype(bool)

    margin_radius = max(0, int(float(config.get("search_margin_frac", 0.0)) * D))
    search = binary_dilation(cell_mask, disk(margin_radius)) if margin_radius > 0 else cell_mask.astype(bool)
    return {"inner_mask": inner.astype(bool), "search_mask": search.astype(bool), "dist_to_border": distance_transform_edt(cell_mask)}

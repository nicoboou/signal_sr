"""Deterministic candidate feature computation."""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk


def compute_candidate_features(
    cand_mask: np.ndarray,
    cell_mask: np.ndarray,
    inner_mask: np.ndarray,
    parasite_channels: dict,
    dist_to_border: np.ndarray,
    D: float,
) -> dict:
    cand_mask = cand_mask.astype(bool)
    cell_mask = cell_mask.astype(bool)
    inner_mask = inner_mask.astype(bool)
    region = regionprops(label(cand_mask))[0]

    area = int(cand_mask.sum())
    cell_area = int(cell_mask.sum())
    inside = cand_mask & cell_mask
    inside_dist = dist_to_border[inside]
    Pz = parasite_channels["purple_z"]
    S = parasite_channels["saturation"]
    darkness = parasite_channels["darkness"]

    ring_radius = max(1, int(0.10 * D))
    local_ring = binary_dilation(cand_mask, disk(ring_radius)) & cell_mask & ~cand_mask
    ring_vals = Pz[local_ring]
    if ring_vals.size:
        local_med = float(np.median(ring_vals))
        local_mad = float(np.median(np.abs(ring_vals - local_med))) + 1e-6
        purple_contrast = (float(np.mean(Pz[cand_mask])) - local_med) / (1.4826 * local_mad)
    else:
        purple_contrast = 0.0

    return {
        "area": area,
        "cell_area": cell_area,
        "area_ratio": float(area / max(cell_area, 1)),
        "equiv_diam": float(region.equivalent_diameter),
        "diam_ratio": float(region.equivalent_diameter / max(D, 1e-6)),
        "centroid_y": float(region.centroid[0]),
        "centroid_x": float(region.centroid[1]),
        "bbox_y0": int(region.bbox[0]),
        "bbox_x0": int(region.bbox[1]),
        "bbox_y1": int(region.bbox[2]),
        "bbox_x1": int(region.bbox[3]),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "inside_fraction": float(inside.sum() / max(area, 1)),
        "inner_fraction": float((cand_mask & inner_mask).sum() / max(area, 1)),
        "mean_purple_z": float(np.mean(Pz[cand_mask])),
        "mean_saturation": float(np.mean(S[cand_mask])),
        "mean_darkness": float(np.mean(darkness[cand_mask])),
        "min_dist_border": float(np.min(inside_dist)) if inside_dist.size else 0.0,
        "dist_border_ratio": float((np.min(inside_dist) if inside_dist.size else 0.0) / max(D, 1e-6)),
        "purple_contrast": float(purple_contrast),
    }

from __future__ import annotations

"""Candidate extraction and hard-rule classification."""

from numbers import Real

import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_opening, disk, remove_small_objects


def extract_candidates(enhanced: np.ndarray, parasite_channels: dict, search_mask: np.ndarray, config: dict) -> list[np.ndarray]:
    candidate_mask = (
        search_mask.astype(bool)
        & (parasite_channels["purple_z"] > float(config["purple_z_min"]))
        & (parasite_channels["saturation"] > float(config["saturation_min"]))
        & (enhanced > float(config["enhanced_min"]))
    )
    candidate_mask = remove_small_objects(candidate_mask, min_size=int(config["candidate_min_area_px"]))
    candidate_mask = binary_opening(candidate_mask, disk(1))
    candidate_mask = binary_dilation(candidate_mask, disk(1))
    labels = label(candidate_mask)
    return [labels == i for i in range(1, int(labels.max()) + 1)]


def classify_candidate(features: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    checks = [
        (features["inside_fraction"] < config["inside_fraction_min"], "not_inside_cell"),
        (features["inner_fraction"] < config["inner_fraction_min"], "not_enough_inner_overlap"),
        (features["area_ratio"] < config["candidate_area_ratio_min"], "too_small"),
        (features["area_ratio"] > config["candidate_area_ratio_max"], "too_large"),
        (features["diam_ratio"] < config["candidate_diam_ratio_min"], "diameter_too_small"),
        (features["diam_ratio"] > config["candidate_diam_ratio_max"], "diameter_too_large"),
        (features["dist_border_ratio"] < config["dist_border_ratio_min"], "too_close_to_cell_border"),
        (features["mean_purple_z"] < config["mean_purple_z_min"], "not_purple_enough"),
        (features["purple_contrast"] < config["purple_contrast_min"], "low_local_contrast"),
        (features["mean_saturation"] < config["mean_saturation_min"], "low_saturation"),
        (features["eccentricity"] > config["eccentricity_max"], "too_elongated"),
        (features["solidity"] < config["solidity_min"], "low_solidity"),
    ]
    reasons.extend(reason for failed, reason in checks if failed)
    if not all(np.isfinite(value) for value in features.values() if isinstance(value, Real)):
        reasons.append("non_finite_features")
    return len(reasons) == 0, reasons

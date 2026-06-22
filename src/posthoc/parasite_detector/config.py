"""Default thresholds for the deterministic parasite detector."""

DEFAULT_CONFIG = {
    # preprocessing
    "illumination_correction": True,
    "invalid_gray_threshold": 0.05,
    "invalid_dilation_radius": 2,
    "background_sigma": 20,
    # cell segmentation
    "min_cell_area_px": 100,
    "cell_area_min_frac": 0.05,
    "cell_area_max_frac": 0.95,
    "cell_eccentricity_max": 0.98,
    "cell_solidity_min": 0.50,
    # intracellular/search mask
    "inner_erosion_frac": 0.03,
    "search_margin_frac": 0.02,
    # parasite channel
    "purple_z_min": 2.5,
    "saturation_min": 0.12,
    # top-hat enhancement
    "tophat_radius_frac": 0.06,
    "enhanced_min": 0.5,
    # candidate cleanup
    "candidate_min_area_px": 2,
    # candidate filters
    "candidate_area_ratio_min": 0.0003,
    "candidate_area_ratio_max": 0.04,
    "candidate_diam_ratio_min": 0.015,
    "candidate_diam_ratio_max": 0.25,
    "inside_fraction_min": 0.90,
    "inner_fraction_min": 0.50,
    "dist_border_ratio_min": 0.01,
    "mean_purple_z_min": 2.5,
    "purple_contrast_min": 2.5,
    "mean_saturation_min": 0.12,
    "eccentricity_max": 0.98,
    "solidity_min": 0.20,
}


def merged_config(config=None):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(config or {})
    return cfg

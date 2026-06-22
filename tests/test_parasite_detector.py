import numpy as np
from skimage.draw import disk

from posthoc.parasite_detector.candidates import classify_candidate
from posthoc.parasite_detector.config import merged_config
from posthoc.parasite_detector.detector import detect_parasites


def synthetic_cell(parasite_inside=True, parasite_outside=False):
    img = np.ones((96, 96, 3), dtype=np.float32)
    rr, cc = disk((48, 48), 25, shape=img.shape[:2])
    img[rr, cc] = (0.85, 0.45, 0.55)
    if parasite_inside:
        rr, cc = disk((42, 53), 3, shape=img.shape[:2])
        img[rr, cc] = (0.20, 0.05, 0.35)
    if parasite_outside:
        rr, cc = disk((12, 12), 3, shape=img.shape[:2])
        img[rr, cc] = (0.20, 0.05, 0.35)
    return (255 * img).astype(np.uint8)


def detector_config():
    return {"illumination_correction": False, "search_margin_frac": 0.0, "tophat_radius_frac": 0.12}


def test_detects_synthetic_intracellular_purple_spot():
    result = detect_parasites(synthetic_cell(parasite_inside=True), config=detector_config(), true_label="parasitized", return_debug=False)
    assert result["true_label"] == "parasitized"
    assert result["inferred_label"] == "parasitized"
    assert result["segmentation_status"] == "ok"
    assert len(result["accepted_candidates"]) >= 1
    assert result["parasite_mask"].sum() > 0
    assert not result["features"].empty


def test_rejects_purple_spot_outside_cell_search_region():
    result = detect_parasites(
        synthetic_cell(parasite_inside=False, parasite_outside=True), config=detector_config(), true_label="uninfected", return_debug=False
    )
    assert result["inferred_label"] == "uninfected"
    assert len(result["accepted_candidates"]) == 0
    assert result["parasite_mask"].sum() == 0


def test_black_image_does_not_attempt_parasite_detection():
    result = detect_parasites(np.zeros((64, 64, 3), dtype=np.uint8), config=detector_config(), return_debug=False)
    assert result["segmentation_status"] == "undetected_cell"
    assert result["inferred_label"] == "uninfected"
    assert len(result["cell_masks"]) == 0
    assert len(result["accepted_candidates"]) == 0


def test_inner_fraction_rule_is_enforced():
    config = merged_config()
    features = {
        "inside_fraction": 1.0,
        "inner_fraction": 0.0,
        "area_ratio": 0.01,
        "diam_ratio": 0.05,
        "dist_border_ratio": 0.05,
        "mean_purple_z": 4.0,
        "purple_contrast": 4.0,
        "mean_saturation": 0.5,
        "eccentricity": 0.0,
        "solidity": 1.0,
    }
    accepted, reasons = classify_candidate(features, config)
    assert not accepted
    assert "not_enough_inner_overlap" in reasons

from __future__ import annotations

"""Public deterministic parasite detection pipeline."""

import numpy as np
import pandas as pd

from posthoc.parasite_detector.candidates import classify_candidate, extract_candidates
from posthoc.parasite_detector.cells import equivalent_diameter_from_area, make_inner_cell_mask, segment_cells
from posthoc.parasite_detector.config import merged_config
from posthoc.parasite_detector.features import compute_candidate_features
from posthoc.parasite_detector.parasite_channel import compute_parasite_channel, enhance_small_purple_objects
from posthoc.parasite_detector.preprocessing import preprocess_rgb
from posthoc.parasite_detector.visualize import make_debug_images


def _candidate_record(mask, features, cell_id, accepted, reasons):
    return {
        "bbox": [features["bbox_y0"], features["bbox_x0"], features["bbox_y1"], features["bbox_x1"]],
        "centroid": [features["centroid_y"], features["centroid_x"]],
        "area": int(features["area"]),
        "cell_id": int(cell_id),
        "status": "accepted" if accepted else "rejected",
        "confidence": 1.0 if accepted else 0.0,
        "reasons": reasons,
        "features": features,
        "mask": mask.astype(bool),
    }


def _features_row(candidate):
    row = {"cell_id": candidate["cell_id"], "status": candidate["status"], "reasons": ";".join(candidate["reasons"])}
    row.update(candidate["features"])
    return row


def detect_parasites(image_rgb: np.ndarray, config: dict | None = None, true_label: str | None = None, return_debug: bool = True) -> dict:
    cfg = merged_config(config)
    prep = preprocess_rgb(image_rgb, cfg)
    cell_masks = segment_cells(prep["img_corr"], prep["invalid_mask"], cfg)
    accepted_candidates, rejected_candidates, inner_masks = [], [], []
    parasite_mask = np.zeros(prep["img"].shape[:2], dtype=bool)
    debug_purple_z = np.zeros_like(parasite_mask, dtype=float)
    debug_enhanced = np.zeros_like(parasite_mask, dtype=float)

    for cell_id, cell_mask in enumerate(cell_masks):
        D = equivalent_diameter_from_area(cell_mask.sum())
        masks = make_inner_cell_mask(cell_mask, D, cfg)
        inner_masks.append(masks["inner_mask"])
        channels = compute_parasite_channel(prep["img_corr"], cell_mask)
        enhanced = enhance_small_purple_objects(channels["purple_z"], D, cfg)
        debug_purple_z = np.maximum(debug_purple_z, np.where(cell_mask, channels["purple_z"], 0.0))
        debug_enhanced = np.maximum(debug_enhanced, np.where(masks["search_mask"], enhanced, 0.0))

        for cand_mask in extract_candidates(enhanced, channels, masks["search_mask"] & ~prep["invalid_mask"], cfg):
            features = compute_candidate_features(cand_mask, cell_mask, masks["inner_mask"], channels, masks["dist_to_border"], D)
            accepted, reasons = classify_candidate(features, cfg)
            candidate = _candidate_record(cand_mask, features, cell_id, accepted, reasons)
            if accepted:
                accepted_candidates.append(candidate)
                parasite_mask |= cand_mask
            else:
                rejected_candidates.append(candidate)

    feature_rows = [_features_row(candidate) for candidate in accepted_candidates + rejected_candidates]
    inferred_label = "parasitized" if accepted_candidates else "uninfected"
    result = {
        "true_label": true_label,
        "inferred_label": inferred_label,
        "infered_label": inferred_label,
        "segmentation_status": "ok" if cell_masks else "undetected_cell",
        "cell_masks": cell_masks,
        "inner_masks": inner_masks,
        "parasites_masks": [candidate["mask"] for candidate in accepted_candidates],
        "parasite_mask": parasite_mask,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "features": pd.DataFrame(feature_rows),
        "debug_images": {},
    }
    if return_debug:
        debug_channels = {"purple_z": debug_purple_z}
        result["debug_images"] = make_debug_images(
            prep["img"],
            cell_masks,
            inner_masks,
            debug_channels,
            debug_enhanced,
            accepted_candidates,
            rejected_candidates,
        )
    return result

"""Debug visualizations for parasite detection decisions."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries


def to_uint8_rgb(img):
    arr = np.asarray(img)
    arr = arr[..., :3] if arr.ndim == 3 else np.repeat(arr[..., None], 3, axis=-1)
    arr = arr / 255.0 if np.issubdtype(arr.dtype, np.floating) and arr.size and np.nanmax(arr) > 1.0 else arr
    return np.clip(arr * 255.0 if np.issubdtype(arr.dtype, np.floating) else arr, 0, 255).astype(np.uint8)


def normalize_gray(x):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        out = np.zeros_like(x, dtype=np.uint8)
    else:
        out = np.zeros_like(x, dtype=float)
        valid = np.isfinite(x)
        out[valid] = np.clip((x[valid] - finite.min()) / (finite.max() - finite.min()), 0.0, 1.0)
        out = (255.0 * out).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=-1)


def overlay_mask(img, mask, color, alpha=0.35, boundaries=False):
    out = to_uint8_rgb(img).astype(float)
    draw_mask = find_boundaries(mask, mode="outer") if boundaries else mask.astype(bool)
    out[draw_mask] = (1.0 - alpha) * out[draw_mask] + alpha * np.asarray(color, dtype=float)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_debug_overlay(img, cell_masks, inner_masks, accepted_candidates, rejected_candidates):
    overlay = to_uint8_rgb(img)
    for mask in cell_masks:
        overlay = overlay_mask(overlay, mask, (0, 255, 0), alpha=0.85, boundaries=True)
    for mask in inner_masks:
        overlay = overlay_mask(overlay, mask, (0, 160, 255), alpha=0.20, boundaries=False)
    for candidate in rejected_candidates:
        overlay = overlay_mask(overlay, candidate["mask"], (255, 210, 0), alpha=0.45, boundaries=False)
    for candidate in accepted_candidates:
        overlay = overlay_mask(overlay, candidate["mask"], (255, 0, 0), alpha=0.65, boundaries=False)

    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for candidate in accepted_candidates + rejected_candidates:
        y0, x0, y1, x1 = candidate["bbox"]
        color = (255, 0, 0) if candidate["status"] == "accepted" else (255, 210, 0)
        text = "ok" if candidate["status"] == "accepted" else ",".join(candidate["reasons"][:2])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=1)
        draw.text((x0, max(0, y0 - 10)), text, fill=color)
    return np.asarray(image)


def make_debug_images(img, cell_masks, inner_masks, parasite_channels, enhanced, accepted_candidates, rejected_candidates):
    original = to_uint8_rgb(img)
    cell_overlay = original.copy()
    inner_overlay = original.copy()
    for mask in cell_masks:
        cell_overlay = overlay_mask(cell_overlay, mask, (0, 255, 0), alpha=0.85, boundaries=True)
    for mask in inner_masks:
        inner_overlay = overlay_mask(inner_overlay, mask, (0, 160, 255), alpha=0.25, boundaries=False)
    return {
        "original": original,
        "cell_mask_overlay": cell_overlay,
        "inner_cell_overlay": inner_overlay,
        "purple_channel": normalize_gray(parasite_channels.get("purple_z", np.zeros(original.shape[:2]))),
        "enhanced": normalize_gray(enhanced),
        "overlay": make_debug_overlay(original, cell_masks, inner_masks, accepted_candidates, rejected_candidates),
    }


def save_debug_outputs(result: dict, output_dir, image_id="image"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in result.get("debug_images", {}).items():
        Image.fromarray(to_uint8_rgb(image)).save(output_dir / f"{image_id}_{name}.png")
    features = result.get("features")
    if features is not None:
        features.to_csv(output_dir / f"{image_id}_features.csv", index=False)

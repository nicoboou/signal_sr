"""Synthetic microscopy dataset generation utilities."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology
from skimage.draw import line, polygon


@dataclass(frozen=True)
class NormalSpec:
    mean: float
    std: float


def N(mean, std):
    """Gaussian sampling specification used in notebook configs."""
    return NormalSpec(float(mean), float(std))


@dataclass
class SyntheticDataset:
    images: np.ndarray
    labels: pd.DataFrame
    cell_masks: np.ndarray
    parasite_masks: np.ndarray
    filament_masks: np.ndarray
    metadata: pd.DataFrame


def sample_value(spec, rng, min_value=1.0):
    value = rng.normal(spec.mean, spec.std) if isinstance(spec, NormalSpec) else float(spec)
    return float(max(min_value, value))


def normalise_binary_values(values, name):
    if isinstance(values, (int, np.integer, bool, str)):
        values = [values]
    values = list(values)
    if not values:
        raise ValueError(f"{name} must contain at least one binary value")
    out = []
    for value in values:
        try:
            ivalue = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only 0/1 values") from exc
        if ivalue not in (0, 1):
            raise ValueError(f"{name} must contain only 0/1 values")
        if ivalue not in out:
            out.append(ivalue)
    return tuple(out)


def equivalent_diameter(area):
    return float(2.0 * np.sqrt(float(area) / np.pi)) if area > 0 else 0.0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return max(0, int(x1) - int(x0)) * max(0, int(y1) - int(y0))


def mask_min_distance(mask_a, mask_b):
    """Euclidean minimum distance in pixels between two binary masks.

    Returns np.inf if either mask is empty. Returns 0.0 when masks overlap.
    """
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if not mask_a.any() or not mask_b.any():
        return float(np.inf)
    return float(ndi.distance_transform_edt(~mask_b)[mask_a].min())


def make_cell_mask(size, diameter, rng):
    center = (size - 1) / 2
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    noise = ndi.gaussian_filter1d(rng.normal(0, 1, len(angles)), sigma=4, mode="wrap")
    noise = noise / (noise.std() + 1e-8)
    radii = 0.5 * diameter * (1 + 0.05 * noise)
    ys = center + radii * np.sin(angles)
    xs = center + radii * np.cos(angles)
    rr, cc = polygon(ys, xs, shape=(size, size))
    mask = np.zeros((size, size), dtype=bool)
    mask[rr, cc] = True
    return ndi.binary_fill_holes(mask)


def make_cell_texture(size, cell_mask, period, amplitude, rng):
    yy, xx = np.mgrid[:size, :size]
    angle = rng.uniform(0, np.pi)
    phase = rng.uniform(0, 2 * np.pi)
    coordinate = xx * np.cos(angle) + yy * np.sin(angle)
    wave = np.sin(2 * np.pi * coordinate / period + phase)
    noise = ndi.gaussian_filter(rng.normal(0, 1, (size, size)), sigma=max(1.0, period / 3))
    noise = noise / (noise[cell_mask].std() + 1e-8)
    texture = 0.75 * wave + 0.25 * noise
    image = np.full((size, size), 0.08, dtype=np.float32)
    image[cell_mask] = np.clip(0.70 + amplitude * texture[cell_mask], 0, 1)
    return image


def random_point_inside(cell_mask, margin, rng):
    distance = ndi.distance_transform_edt(cell_mask)
    coords = np.argwhere(distance > margin)
    if len(coords) == 0:
        coords = np.argwhere(cell_mask)
    y, x = coords[rng.integers(0, len(coords))]
    return float(y), float(x)


def ellipse_mask(size, cy, cx, diameter):
    yy, xx = np.mgrid[:size, :size]
    radius = max(0.5, diameter / 2)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def make_parasite_mask(
    size,
    cell_mask,
    diameter,
    rng,
    avoid_mask=None,
    min_distance=0.0,
):
    """Create a compact parasite mask inside the cell.

    If ``avoid_mask`` is provided, the parasite center is sampled so that the
    final parasite disk is at least ``min_distance`` pixels away from that mask.
    If the requested constraint is infeasible for a particular cell, the
    distance is relaxed gradually instead of returning an empty parasite.
    """
    radius = max(0.5, float(diameter) / 2.0)
    cell_distance = ndi.distance_transform_edt(cell_mask)
    base_valid = cell_distance > (radius + 6.0)

    if avoid_mask is not None and np.asarray(avoid_mask, dtype=bool).any() and min_distance > 0:
        avoid_distance = ndi.distance_transform_edt(~np.asarray(avoid_mask, dtype=bool))
        for relaxation in (1.0, 0.75, 0.5, 0.25, 0.0):
            required_center_distance = radius + float(min_distance) * relaxation
            valid = base_valid & (avoid_distance > required_center_distance)
            coords = np.argwhere(valid)
            if len(coords) > 0:
                cy, cx = coords[rng.integers(0, len(coords))]
                return ellipse_mask(size, float(cy), float(cx), diameter) & cell_mask

    coords = np.argwhere(base_valid)
    if len(coords) == 0:
        coords = np.argwhere(cell_mask)
    cy, cx = coords[rng.integers(0, len(coords))]
    return ellipse_mask(size, float(cy), float(cx), diameter) & cell_mask


def _draw_filament_candidate(size, cell_mask, cy, cx, width, length, rng):
    angle = rng.uniform(0, np.pi)
    bend = rng.uniform(4, 10)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(-0.5, 0.5, 80)
    along_y, along_x = np.sin(angle), np.cos(angle)
    perp_y, perp_x = np.cos(angle), -np.sin(angle)
    ys = cy + length * t * along_y + bend * np.sin(2 * np.pi * t + phase) * perp_y
    xs = cx + length * t * along_x + bend * np.sin(2 * np.pi * t + phase) * perp_x
    centerline = np.zeros((size, size), dtype=bool)
    points = np.c_[np.clip(np.round(ys), 0, size - 1), np.clip(np.round(xs), 0, size - 1)].astype(int)
    for p0, p1 in zip(points[:-1], points[1:]):
        rr, cc = line(p0[0], p0[1], p1[0], p1[1])
        centerline[rr, cc] = True
    radius = max(1, int(round(width / 2)))
    return morphology.dilation(centerline, morphology.disk(radius)) & cell_mask


def make_filament_mask(
    size,
    cell_mask,
    width,
    length,
    rng,
    avoid_mask=None,
    min_distance=0.0,
    max_tries=256,
):
    """Create a filament mask inside the cell.

    If ``avoid_mask`` is provided, candidates that overlap or come closer than
    ``min_distance`` pixels to it are rejected. After ``max_tries`` attempts,
    the farthest valid candidate found is returned as a robust fallback.
    """
    distance = ndi.distance_transform_edt(cell_mask)
    coords = np.argwhere(distance > 22)
    if len(coords) == 0:
        coords = np.argwhere(cell_mask)

    avoid_mask = None if avoid_mask is None else np.asarray(avoid_mask, dtype=bool)
    use_distance_constraint = avoid_mask is not None and avoid_mask.any() and min_distance > 0
    best_mask = None
    best_distance = -np.inf

    for _ in range(int(max_tries)):
        cy, cx = coords[rng.integers(0, len(coords))]
        candidate = _draw_filament_candidate(size, cell_mask, float(cy), float(cx), width, length, rng)
        if not candidate.any():
            continue

        candidate_distance = mask_min_distance(candidate, avoid_mask) if use_distance_constraint else np.inf
        if candidate_distance > best_distance:
            best_distance = candidate_distance
            best_mask = candidate
        if not use_distance_constraint or candidate_distance >= float(min_distance):
            return candidate

    if best_mask is not None:
        return best_mask
    return np.zeros_like(cell_mask, dtype=bool)


def object_row(prefix, mask):
    bbox = bbox_from_mask(mask)
    area = int(mask.sum())
    return {
        f"{prefix}_count": int(area > 0),
        f"{prefix}_area_px2": area,
        f"{prefix}_bbox_x0": bbox[0],
        f"{prefix}_bbox_y0": bbox[1],
        f"{prefix}_bbox_x1": bbox[2],
        f"{prefix}_bbox_y1": bbox[3],
        f"{prefix}_mask_equivalent_diameter_px": equivalent_diameter(area),
        f"{prefix}_bbox_equivalent_diameter_px": equivalent_diameter(bbox_area(bbox)),
    }


def generate_synthetic_dataset(
    n_per_combination=150,
    image_size=128,
    seed=7,
    cell_diam=N(92, 4),
    parasite_diam=N(10, 2),
    filament_width=N(4, 1),
    filament_length=N(44, 6),
    cell_texture_period=N(12, 2),
    poor_texture_amplitude=0.035,
    strong_texture_amplitude=0.14,
    parasite_filament_min_distance_px=10,
    parasite_values=(0, 1),
    filament_values=(0, 1),
    strong_texture_values=(0, 1),
):
    """Generate a configurable factorial synthetic microscopy dataset.

    When both parasite and filament are present, the filament is sampled first
    and the parasite is placed at least ``parasite_filament_min_distance_px``
    pixels away from the filament whenever the cell geometry makes it feasible.
    This prevents the parasite label from being confounded by filament pixels.
    """
    rng = np.random.default_rng(seed)
    images, cell_masks, parasite_masks, filament_masks, rows = [], [], [], [], []
    parasite_values = normalise_binary_values(parasite_values, "parasite_values")
    filament_values = normalise_binary_values(filament_values, "filament_values")
    strong_texture_values = normalise_binary_values(strong_texture_values, "strong_texture_values")
    combinations = list(product(parasite_values, filament_values, strong_texture_values))

    for parasite, filament, strong_texture in combinations:
        for _ in range(int(n_per_combination)):
            cell_d = sample_value(cell_diam, rng, min_value=20)
            parasite_d = sample_value(parasite_diam, rng, min_value=2)
            filament_w = sample_value(filament_width, rng, min_value=1)
            filament_l = sample_value(filament_length, rng, min_value=4)
            texture_p = sample_value(cell_texture_period, rng, min_value=3)
            amplitude = strong_texture_amplitude if strong_texture else poor_texture_amplitude

            cell_mask = make_cell_mask(image_size, cell_d, rng)
            image = make_cell_texture(image_size, cell_mask, texture_p, amplitude, rng)

            filament_mask = make_filament_mask(image_size, cell_mask, filament_w, filament_l, rng) if filament else np.zeros_like(cell_mask)
            parasite_mask = (
                make_parasite_mask(
                    image_size,
                    cell_mask,
                    parasite_d,
                    rng,
                    avoid_mask=filament_mask if filament else None,
                    min_distance=parasite_filament_min_distance_px if filament else 0.0,
                )
                if parasite
                else np.zeros_like(cell_mask)
            )

            parasite_filament_distance = mask_min_distance(parasite_mask, filament_mask) if parasite and filament else np.nan

            image[filament_mask] = 0.20
            image[parasite_mask] = 0.16

            row = {
                "parasite": parasite,
                "filament": filament,
                "strong_texture": strong_texture,
                "cell_diameter_px": cell_d,
                "parasite_diameter_px": parasite_d if parasite else 0.0,
                "filament_width_px": filament_w if filament else 0.0,
                "filament_length_px": filament_l if filament else 0.0,
                "cell_texture_period_px": texture_p,
                "cell_mask_equivalent_diameter_px": equivalent_diameter(int(cell_mask.sum())),
                "parasite_filament_requested_min_distance_px": float(parasite_filament_min_distance_px) if parasite and filament else 0.0,
                "parasite_filament_distance_px": parasite_filament_distance,
            }
            row.update(object_row("parasite", parasite_mask))
            row.update(object_row("filament", filament_mask))

            images.append(image.astype(np.float32))
            cell_masks.append(cell_mask)
            parasite_masks.append(parasite_mask)
            filament_masks.append(filament_mask)
            rows.append(row)

    order = rng.permutation(len(images))
    metadata = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    metadata.insert(0, "sample_id", np.arange(len(metadata)))
    labels = metadata[["parasite", "filament", "strong_texture"]].astype(int)

    return SyntheticDataset(
        images=np.stack(images)[order],
        labels=labels,
        cell_masks=np.stack(cell_masks)[order],
        parasite_masks=np.stack(parasite_masks)[order],
        filament_masks=np.stack(filament_masks)[order],
        metadata=metadata,
    )

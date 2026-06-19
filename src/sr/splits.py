from __future__ import annotations

from pathlib import Path

import numpy as np

from data.synthetic.synthetic import generate_synthetic_dataset, normalise_binary_values


LABEL_COLUMNS = ["parasite", "filament", "strong_texture"]


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _factor_values_from_kwargs(generator_kwargs):
    generator_kwargs = dict(generator_kwargs or {})
    return {
        "parasite": list(normalise_binary_values(generator_kwargs.get("parasite_values", (0, 1)), "parasite_values")),
        "filament": list(normalise_binary_values(generator_kwargs.get("filament_values", (0, 1)), "filament_values")),
        "strong_texture": list(
            normalise_binary_values(generator_kwargs.get("strong_texture_values", (0, 1)), "strong_texture_values")
        ),
    }


def _synthetic_split_metadata(
    strategy,
    split_ratios,
    split_seed,
    n_per_combination,
    image_size,
    generation_seed,
    generator_kwargs,
):
    factor_values = _factor_values_from_kwargs(generator_kwargs)
    n_factor_combinations = int(np.prod([len(values) for values in factor_values.values()]))
    return {
        "strategy": strategy,
        "split_ratios": _plain(tuple(float(x) for x in split_ratios)),
        "split_seed": int(split_seed),
        "generation_seed": int(generation_seed),
        "n_per_combination": int(n_per_combination),
        "image_size": int(image_size),
        "generator_kwargs": _plain(dict(generator_kwargs or {})),
        "factor_values": factor_values,
        "n_factor_combinations": n_factor_combinations,
        "expected_num_samples": n_factor_combinations * int(n_per_combination),
        "label_columns": list(LABEL_COLUMNS),
    }


def _load_split_metadata(split_dir):
    path = Path(split_dir) / "split_metadata.npy"
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    return payload.item() if getattr(payload, "shape", None) == () else payload


def _metadata_matches(current, requested):
    if not isinstance(current, dict):
        return False
    current = _plain(current)
    requested = _plain(requested)
    return all(current.get(key) == value for key, value in requested.items())


def _domains_match_strategy(split_dir, strategy):
    split_dir = Path(split_dir)
    train_path = split_dir / "train_domains.npy"
    val_path = split_dir / "val_domains.npy"
    test_path = split_dir / "test_domains.npy"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        return False
    train_domains = np.load(train_path)
    val_domains = np.load(val_path)
    test_domains = np.load(test_path)
    if strategy == "hr_lr_50_50_then_tvt":
        frac_hr = float(train_domains.mean()) if len(train_domains) else 0.0
        return 0.45 <= frac_hr <= 0.55 and val_domains.sum() == 0 and test_domains.sum() == 0
    if strategy == "train_hr_val_test_lr":
        return train_domains.sum() == len(train_domains) and val_domains.sum() == 0 and test_domains.sum() == 0
    return False


def _normalise_ratios(split_ratios):
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,):
        raise ValueError("split_ratios must contain train/val/test ratios")
    ratios = ratios / ratios.sum()
    return ratios


def _stratified_split_indices(labels: np.ndarray, split_ratios, seed: int):
    ratios = _normalise_ratios(split_ratios)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    keys = [tuple(row.tolist()) for row in labels]
    for key in sorted(set(keys)):
        indices = np.asarray([i for i, item in enumerate(keys) if item == key], dtype=np.int64)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(np.floor(ratios[0] * n))
        n_val = int(np.floor(ratios[1] * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train.extend(indices[:n_train].tolist())
        val.extend(indices[n_train : n_train + n_val].tolist())
        test.extend(indices[n_train + n_val :].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _assign_train_domains(train_idx: np.ndarray, labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    domains = np.zeros(len(train_idx), dtype=np.int64)
    leftovers = []
    keys = [tuple(labels[i].tolist()) for i in train_idx]
    for key in sorted(set(keys)):
        positions = np.asarray([i for i, item in enumerate(keys) if item == key], dtype=np.int64)
        rng.shuffle(positions)
        half = len(positions) // 2
        domains[positions[:half]] = 1
        domains[positions[half:]] = 0
        if len(positions) % 2 == 1:
            leftovers.append(int(positions[half]))
    target_hr = len(train_idx) // 2
    needed = max(0, target_hr - int(domains.sum()))
    if needed > 0 and leftovers:
        leftovers = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(leftovers)
        domains[leftovers[:needed]] = 1
    return domains


def _save_split_arrays(out_dir, name, indices, domains, dataset):
    out_dir = Path(out_dir)
    np.save(out_dir / f"{name}_images.npy", dataset.images[indices].astype(np.float32))
    np.save(out_dir / f"{name}_domains.npy", domains.astype(np.int64))
    np.save(out_dir / f"{name}_labels.npy", dataset.labels.iloc[indices][LABEL_COLUMNS].to_numpy(dtype=np.int64))
    sample_ids = dataset.metadata.iloc[indices]["sample_id"].to_numpy(dtype=np.int64)
    np.save(out_dir / f"{name}_sample_ids.npy", sample_ids)
    np.save(out_dir / f"{name}_cell_masks.npy", dataset.cell_masks[indices].astype(bool))
    np.save(out_dir / f"{name}_parasite_masks.npy", dataset.parasite_masks[indices].astype(bool))
    np.save(out_dir / f"{name}_filament_masks.npy", dataset.filament_masks[indices].astype(bool))
    metadata_rows = dataset.metadata.iloc[indices].to_dict(orient="records")
    np.save(out_dir / f"{name}_metadata.npy", np.asarray(metadata_rows, dtype=object), allow_pickle=True)


def create_synthetic_split_npy(
    out_dir,
    strategy="hr_lr_50_50_then_tvt",
    split_ratios=(0.80, 0.05, 0.15),
    split_seed=0,
    n_per_combination=150,
    image_size=128,
    generation_seed=7,
    generator_kwargs=None,
):
    """Generate synthetic data once and save split arrays with annotations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator_kwargs = dict(generator_kwargs or {})
    metadata = _synthetic_split_metadata(
        strategy=strategy,
        split_ratios=split_ratios,
        split_seed=split_seed,
        n_per_combination=n_per_combination,
        image_size=image_size,
        generation_seed=generation_seed,
        generator_kwargs=generator_kwargs,
    )
    dataset = generate_synthetic_dataset(
        n_per_combination=n_per_combination,
        image_size=image_size,
        seed=generation_seed,
        **generator_kwargs,
    )
    labels = dataset.labels[LABEL_COLUMNS].to_numpy(dtype=np.int64)
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, split_ratios, split_seed)

    if strategy == "hr_lr_50_50_then_tvt":
        train_domains = _assign_train_domains(train_idx, labels, split_seed + 1)
        val_domains = np.zeros(len(val_idx), dtype=np.int64)
        test_domains = np.zeros(len(test_idx), dtype=np.int64)
    elif strategy == "train_hr_val_test_lr":
        train_domains = np.ones(len(train_idx), dtype=np.int64)
        val_domains = np.zeros(len(val_idx), dtype=np.int64)
        test_domains = np.zeros(len(test_idx), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    _save_split_arrays(out_dir, "train", train_idx, train_domains, dataset)
    _save_split_arrays(out_dir, "val", val_idx, val_domains, dataset)
    _save_split_arrays(out_dir, "test", test_idx, test_domains, dataset)

    np.save(out_dir / "split_metadata.npy", metadata, allow_pickle=True)

    total = len(train_idx) + len(val_idx) + len(test_idx)
    expected = int(metadata["expected_num_samples"])
    if total != expected:
        raise AssertionError(f"Expected {expected} samples, found {total}")
    if strategy == "hr_lr_50_50_then_tvt" and len(train_domains) > 0:
        frac_hr = float(train_domains.mean())
        if not (0.45 <= frac_hr <= 0.55):
            raise AssertionError(f"Train HR fraction should be near 0.5, got {frac_hr:.3f}")
        if val_domains.sum() != 0 or test_domains.sum() != 0:
            raise AssertionError("Strategy A requires LR-domain val/test splits")
    return out_dir


def ensure_synthetic_split(cfg) -> None:
    split_dir = Path(cfg.data.split_dir)
    required_suffixes = (
        "images",
        "domains",
        "labels",
        "sample_ids",
        "cell_masks",
        "parasite_masks",
        "filament_masks",
        "metadata",
    )
    required = [
        split_dir / f"{name}_{suffix}.npy"
        for name in ("train", "val", "test")
        for suffix in required_suffixes
    ]
    gen = cfg.get("synthetic_generation", {})
    strategy = cfg.data.get("split_strategy", "hr_lr_50_50_then_tvt")
    split_ratios = cfg.data.get("split_ratios", (0.80, 0.05, 0.15))
    split_seed = cfg.data.get("split_seed", 0)
    n_per_combination = gen.get("n_per_combination", 150)
    image_size = gen.get("image_size", cfg.data.image_size)
    generation_seed = gen.get("seed", 7)
    generator_kwargs = gen.get("generator_kwargs", {})
    requested_metadata = _synthetic_split_metadata(
        strategy=strategy,
        split_ratios=split_ratios,
        split_seed=split_seed,
        n_per_combination=n_per_combination,
        image_size=image_size,
        generation_seed=generation_seed,
        generator_kwargs=generator_kwargs,
    )
    if (
        all(path.exists() for path in required)
        and _metadata_matches(_load_split_metadata(split_dir), requested_metadata)
        and _domains_match_strategy(split_dir, strategy)
    ):
        return

    create_synthetic_split_npy(
        out_dir=split_dir,
        strategy=strategy,
        split_ratios=split_ratios,
        split_seed=split_seed,
        n_per_combination=n_per_combination,
        image_size=image_size,
        generation_seed=generation_seed,
        generator_kwargs=generator_kwargs,
    )

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from prehoc.datasets import NLMDataset
from prehoc.models.classifier import ClassifierImage
from prehoc.utils.seed import seed_everything
from prehoc.utils.transforms import make_transform


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/pre_hoc/nlm_degradation_sweep"
PLOT_METRICS = ("accuracy", "precision", "recall", "f1")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_levels(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_int_values(values):
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    if isinstance(values, str):
        return [int(value.strip()) for value in values.split(",") if value.strip()]
    return [int(value) for value in values]


def apply_overrides(config, args):
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.run_id:
        config["run_id"] = args.run_id
    if args.device:
        config["device"] = args.device
    if args.seeds:
        config["seeds"] = parse_int_values(args.seeds)
    if args.levels:
        config.setdefault("degradation", {})["levels"] = parse_levels(args.levels)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.max_samples_per_split is not None:
        config.setdefault("data", {})["max_samples_per_split"] = int(args.max_samples_per_split)
    if args.split_dir:
        config.setdefault("data", {})["split_dir"] = args.split_dir
    if args.kfold:
        config.setdefault("data", {}).setdefault("kfold", {})["enabled"] = True
    if args.num_folds is not None:
        config.setdefault("data", {}).setdefault("kfold", {})["num_folds"] = int(args.num_folds)
    if args.folds:
        config.setdefault("data", {}).setdefault("kfold", {})["folds"] = parse_int_values(args.folds)
    for name in ("train_csv", "val_csv", "test_csv"):
        value = getattr(args, name)
        if value:
            config.setdefault("data", {})[name] = value
    return config


def choose_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device {device}, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return str(device)


def make_run_output_dir(base_output_dir, run_id=None):
    base_output_dir = resolve_path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_id) if run_id else datetime.now().strftime("%Y%m%d_%H%M")

    for suffix in [""] + [f"_{i:02d}" for i in range(1, 100)]:
        candidate = base_output_dir / f"{run_id}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, candidate.name
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {base_output_dir} for run_id={run_id}")


def subset_dataset(dataset, max_samples, seed):
    if max_samples is None or len(dataset) <= int(max_samples):
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(len(dataset), size=int(max_samples), replace=False).tolist())
    return Subset(dataset, indices)


def training_seeds(config):
    seeds = parse_int_values(config.get("seeds"))
    return seeds if seeds else [int(config.get("seed", 0))]


def cv_folds(config):
    kfold_cfg = config.get("data", {}).get("kfold", {}) or {}
    if not bool(kfold_cfg.get("enabled", False)):
        return [None]
    folds = parse_int_values(kfold_cfg.get("folds"))
    if folds:
        return folds
    num_folds = int(kfold_cfg.get("num_folds", 0))
    if num_folds < 2:
        raise ValueError("data.kfold.num_folds must be >= 2 when kfold is enabled")
    return list(range(1, num_folds + 1))


def resolve_split_csv(config, split_name, fold=None):
    data_cfg = config["data"]
    if fold is not None:
        template = data_cfg.get(f"{split_name}_csv_template")
        if template:
            return resolve_path(str(template).format(fold=int(fold)))
        split_dir = data_cfg.get("split_dir")
        if not split_dir:
            raise ValueError("data.split_dir is required when data.kfold.enabled is true")
        return resolve_path(split_dir) / f"{split_name}_fold_{int(fold)}.csv"

    key = f"{split_name}_csv"
    if not data_cfg.get(key):
        return None
    return resolve_path(data_cfg[key])


def fold_seed_offset(fold):
    return 0 if fold is None else 10_000 * int(fold)


def make_loader(config, split_name, split_csv, transform, shuffle, loader_seed=None, subset_seed=None):
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    crop_root = resolve_path(data_cfg["root"])
    split_csv = resolve_path(split_csv)
    default_seed = int(config.get("seed", 0)) + SPLIT_SEED_OFFSETS.get(split_name, 0)
    loader_seed = default_seed if loader_seed is None else int(loader_seed)
    subset_seed = default_seed if subset_seed is None else int(subset_seed)

    dataset = NLMDataset(root=crop_root, split_csv=split_csv, transform=transform)
    dataset = subset_dataset(dataset, data_cfg.get("max_samples_per_split"), seed=subset_seed)

    num_workers = int(data_cfg.get("num_workers", 0))
    generator = torch.Generator().manual_seed(loader_seed)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", False)) and str(config["device"]).startswith("cuda"),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)) and num_workers > 0,
        generator=generator if shuffle else None,
    )


def binary_metrics(y_true, positive_probability, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(positive_probability) >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_and_eval_level(level, config, classifier_seed=None, fold=None):
    classifier_seed = int(config.get("seed", 0) if classifier_seed is None else classifier_seed)
    seed_everything(classifier_seed)

    transform = make_transform(config, level)
    data_seed = int(config.get("seed", 0)) + fold_seed_offset(fold)
    split_csvs = {
        "train": resolve_split_csv(config, "train", fold=fold),
        "val": resolve_split_csv(config, "val", fold=fold),
    }
    test_csv = None if fold is not None else resolve_split_csv(config, "test")

    train_loader = make_loader(
        config,
        "train",
        split_csvs["train"],
        transform,
        shuffle=True,
        loader_seed=classifier_seed + SPLIT_SEED_OFFSETS["train"],
        subset_seed=data_seed + SPLIT_SEED_OFFSETS["train"],
    )
    val_loader = make_loader(
        config,
        "val",
        split_csvs["val"],
        transform,
        shuffle=False,
        loader_seed=classifier_seed + SPLIT_SEED_OFFSETS["val"],
        subset_seed=data_seed + SPLIT_SEED_OFFSETS["val"],
    )
    test_loader = None
    if test_csv is not None:
        test_loader = make_loader(
            config,
            "test",
            test_csv,
            transform,
            shuffle=False,
            loader_seed=classifier_seed + SPLIT_SEED_OFFSETS["test"],
            subset_seed=data_seed + SPLIT_SEED_OFFSETS["test"],
        )

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    classifier = ClassifierImage(
        device=config["device"],
        epochs=train_cfg.get("epochs", 5),
        batch_size=train_cfg.get("batch_size", 64),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        width=model_cfg.get("width", 24),
        backbone=model_cfg.get("backbone", "small_cnn"),
        pretrained=model_cfg.get("resnet_pretrained", False),
        in_channels=model_cfg.get("in_channels", 3),
    )
    classifier.fit_loader(train_loader, val_loader, seed=classifier_seed)

    threshold = float(config.get("eval", {}).get("threshold", 0.5))
    val_prob, val_labels = classifier.probabilities_loader(val_loader)
    val_metrics = binary_metrics(val_labels, val_prob, threshold=threshold)
    out = {f"val_{key}": value for key, value in val_metrics.items()}
    if test_loader is None:
        out.update(val_metrics)
        out.update({"n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset)})
    else:
        test_prob, test_labels = classifier.probabilities_loader(test_loader)
        test_metrics = binary_metrics(test_labels, test_prob, threshold=threshold)
        out.update(test_metrics)
        out.update({"n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset), "n_test": len(test_loader.dataset)})
    return out


def summarize_metrics(results):
    metrics = [metric for metric in PLOT_METRICS if metric in results.columns]
    group_cols = ["degradation_type", "level"]
    rows = []
    for keys, group in results.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_repetitions"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else 1
        row["n_folds"] = int(group["fold"].nunique()) if "fold" in group else 1
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_sem"] = float(row[f"{metric}_std"] / np.sqrt(len(values))) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("level")


def plot_metrics(results, output_path, summary=None):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.usetex": False,
            "pdf.fonttype": 42,
        }
    )
    summary = summarize_metrics(results) if summary is None else summary.sort_values("level")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {"accuracy": "#1f77b4", "precision": "#ff7f0e", "recall": "#2ca02c", "f1": "#9467bd"}
    x = summary["level"].to_numpy(dtype=float)
    for metric in PLOT_METRICS:
        mean_col = f"{metric}_mean"
        sem_col = f"{metric}_sem"
        if mean_col not in summary:
            continue
        y = summary[mean_col].to_numpy(dtype=float)
        sem = summary[sem_col].to_numpy(dtype=float) if sem_col in summary else np.zeros_like(y)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.5, label=metric, color=colors[metric])
        if np.nanmax(sem) > 0:
            ax.fill_between(x, np.clip(y - sem, 0, 1), np.clip(y + sem, 0, 1), color=colors[metric], alpha=0.14, linewidth=0)

    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    if len(x) > 2 and np.all(x > 0):
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _safe_wandb_artifact_name(value):
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return (safe.strip("_") or "prehoc-results")[:120]


def log_wandb_outputs(config, output_dir, results, summary):
    wandb_cfg = config.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        return
    try:
        import wandb
    except ImportError:
        print("wandb.enabled=true but wandb is not installed; skipping W&B logging")
        return

    run_id = wandb_cfg.get("id") or config.get("run_id")
    init_kwargs = {
        "project": wandb_cfg.get("project", "signal_sr"),
        "config": config,
    }
    if run_id:
        run_id = str(run_id)
        init_kwargs.update(
            {
                "id": run_id,
                "name": wandb_cfg.get("name", run_id),
                "resume": wandb_cfg.get("resume", "allow"),
            }
        )
    for key in ("entity", "group", "tags", "notes", "job_type"):
        if key in wandb_cfg:
            init_kwargs[key] = wandb_cfg[key]

    run = wandb.init(**init_kwargs)
    try:
        run.log({"metrics": wandb.Table(dataframe=results), "metrics_summary": wandb.Table(dataframe=summary)})
        artifact = wandb.Artifact(_safe_wandb_artifact_name(f"{config.get('run_id', run_id)}_prehoc_results"), type="results")
        for filename in ("metrics.csv", "metrics_summary.csv", "metrics.pdf", "config.yaml", "argv.json"):
            path = Path(output_dir) / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)
        run.log_artifact(artifact)
    finally:
        run.finish()


def run(config, config_path=None):
    seed_everything(int(config.get("seed", 0)))
    config["device"] = choose_device(config.get("device", "cpu"))
    base_output_dir = resolve_path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir, run_id = make_run_output_dir(base_output_dir, config.get("run_id"))
    config["run_id"] = run_id
    config["base_output_dir"] = str(base_output_dir)
    config["output_dir"] = str(output_dir)
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_dir}")

    with (output_dir / "argv.json").open("w", encoding="utf-8") as file:
        json.dump({"argv": sys.argv}, file, indent=2)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    if config_path is not None:
        shutil.copy2(config_path, output_dir / "source_config.yaml")

    rows = []
    folds = cv_folds(config)
    seeds = training_seeds(config)
    for fold in folds:
        for classifier_seed in seeds:
            for level in config["degradation"]["levels"]:
                fold_text = f" fold={fold}" if fold is not None else ""
                print(f"Training/evaluating {config['degradation']['type']} level={level} seed={classifier_seed}{fold_text}")
                metrics = train_and_eval_level(level, config, classifier_seed=classifier_seed, fold=fold)
                row = {"degradation_type": config["degradation"]["type"], "level": float(level), "seed": int(classifier_seed), **metrics}
                if fold is not None:
                    row["fold"] = int(fold)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)

    sort_cols = [column for column in ("fold", "seed", "level") if column in rows[0]]
    results = pd.DataFrame(rows).sort_values(sort_cols)
    results.to_csv(output_dir / "metrics.csv", index=False)
    summary = summarize_metrics(results)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    plot_metrics(results, output_dir / "metrics.pdf", summary=summary)
    log_wandb_outputs(config, output_dir, results, summary)
    print(f"Wrote {output_dir / 'metrics.csv'}")
    print(f"Wrote {output_dir / 'metrics_summary.csv'}")
    print(f"Wrote {output_dir / 'metrics.pdf'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a classifier sweep over one degradation type.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated classifier/training seeds overriding the YAML config.")
    parser.add_argument("--levels", default=None, help="Comma-separated degradation levels overriding the YAML config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--kfold", action="store_true", help="Enable train_fold_i.csv/val_fold_i.csv cross-validation mode.")
    parser.add_argument("--num-folds", type=int, default=None)
    parser.add_argument("--folds", default=None, help="Comma-separated 1-indexed folds to run, overriding data.kfold.folds.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = apply_overrides(load_config(config_path), args)
    run(config, config_path=config_path)


if __name__ == "__main__":
    main()

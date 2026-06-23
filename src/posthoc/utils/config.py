from pathlib import Path
import yaml
import torch
from datetime import datetime


def resolve_path(path, repo_root=Path(__file__).parent.parent.parent):
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_levels(text):
    return [float(value.strip()) for value in str(text).split(",") if value.strip()]


def apply_overrides(config, args):
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.run_id:
        config["run_id"] = args.run_id
    if args.device:
        config["device"] = args.device
    if args.levels:
        config.setdefault("degradation", {})["levels"] = parse_levels(args.levels)
    if args.split:
        config.setdefault("sr", {})["split"] = args.split
    if args.sr_checkpoint:
        config.setdefault("sr", {})["checkpoint"] = args.sr_checkpoint
    if args.classifier_checkpoint:
        config.setdefault("classifier", {})["checkpoint"] = args.classifier_checkpoint
    if args.max_samples is not None:
        config.setdefault("eval", {})["max_samples"] = int(args.max_samples)
    return config


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

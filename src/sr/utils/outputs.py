from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src/sr/outputs"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def default_train_output_base(config_path):
    return DEFAULT_OUTPUT_ROOT / Path(config_path).stem


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

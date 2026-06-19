from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_EXCLUDE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".archive",
    "wandb",
    "runs",
    "outputs",
    "checkpoints",
}


@dataclass
class JobRunner:
    code_dir: str
    command: list[str]
    env: dict[str, str]

    def __call__(self) -> None:
        code_dir = Path(self.code_dir)
        env = os.environ.copy()
        env.update(self.env)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(code_dir / "src"), str(code_dir), env.get("PYTHONPATH", "")]
        )
        for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME"):
            if env.get(key):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        subprocess.run(self.command, cwd=code_dir, env=env, check=True)


def safe_name(value: str) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "job"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def expand_vars(value):
    if isinstance(value, dict):
        return {key: expand_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def clean_training_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def default_scratch_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit("Set $SCRATCH or pass --scratch-root.")
    return Path(scratch).expanduser().resolve() / "signal_sr"


def default_cpus_per_task(args: argparse.Namespace) -> int:
    constraint = str(args.constraint or "").lower()
    partition = str(args.partition or "").lower()
    gpus = int(args.gpus or 1)
    if constraint == "a100":
        return 8 * gpus
    if constraint == "h100":
        return 24 * gpus
    if partition.startswith("gpu_p2"):
        return 3 * gpus
    return 10 * gpus


def apply_jz_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.jz_config:
        return args
    jz_path = repo_path(args.jz_config).resolve()
    if not jz_path.is_file():
        raise SystemExit(f"Missing Jean Zay config: {jz_path}")
    jz = expand_vars(load_yaml(jz_path))
    pipeline = jz.get("pipeline")
    if pipeline and pipeline != args.pipeline:
        raise SystemExit(f"Jean Zay config pipeline={pipeline!r} does not match subcommand {args.pipeline!r}")

    job_cfg = jz.get("job", {}) or {}
    paths_cfg = jz.get("paths", {}) or {}
    runtime_cfg = jz.get("runtime", {}) or {}
    slurm_cfg = jz.get("slurm", {}) or {}
    data_cfg = jz.get("data", {}) or {}
    wandb_cfg = jz.get("wandb", {}) or {}
    train_cfg = jz.get("training", {}) or {}

    args.config = args.config or jz.get("base_config")
    args.job_name = args.job_name or job_cfg.get("name")
    args.scratch_root = args.scratch_root or paths_cfg.get("scratch_root")
    args.output_root = args.output_root or paths_cfg.get("output_root")
    args.python = args.python or runtime_cfg.get("python")
    args.data_root = args.data_root or data_cfg.get("root")
    args.split_dir = args.split_dir or data_cfg.get("split_dir")
    args.account = args.account or slurm_cfg.get("account")
    args.partition = args.partition if args.partition is not None else slurm_cfg.get("partition")
    args.qos = args.qos or slurm_cfg.get("qos")
    args.constraint = args.constraint if args.constraint is not None else slurm_cfg.get("constraint")
    args.time_min = args.time_min or slurm_cfg.get("time_min")
    args.nodes = args.nodes or slurm_cfg.get("nodes")
    args.tasks_per_node = args.tasks_per_node or slurm_cfg.get("tasks_per_node")
    args.gpus = args.gpus or slurm_cfg.get("gpus")
    args.gres = args.gres or slurm_cfg.get("gres")
    args.cpus_per_task = args.cpus_per_task or slurm_cfg.get("cpus_per_task")
    args.hint = args.hint if args.hint is not None else slurm_cfg.get("hint")

    if wandb_cfg.get("enabled") is False:
        args.disable_wandb = True
    elif wandb_cfg.get("enabled") is True and not args.wandb_project:
        args.wandb_project = wandb_cfg.get("project") or "signal_sr"
    args.wandb_entity = args.wandb_entity or wandb_cfg.get("entity")

    if args.pipeline == "prehoc":
        args.disable_pretrained = bool(args.disable_pretrained or train_cfg.get("disable_pretrained", False))
    else:
        args.num_workers = args.num_workers or train_cfg.get("num_workers")
        args.mixed_precision = args.mixed_precision or train_cfg.get("mixed_precision")

    config_training_args = train_cfg.get("args") or []
    args.training_args = [*config_training_args, *clean_training_args(args.training_args)]
    return args


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args = apply_jz_config(args)
    if not args.config:
        raise SystemExit("Pass --config or provide base_config in --jz-config.")
    args.python = args.python or sys.executable
    args.time_min = int(args.time_min or 120)
    args.nodes = int(args.nodes or 1)
    args.tasks_per_node = int(args.tasks_per_node or 1)
    args.gpus = int(args.gpus or 1)
    args.gres = args.gres or f"gpu:{args.gpus}"
    args.cpus_per_task = int(args.cpus_per_task or default_cpus_per_task(args))
    args.hint = "nomultithread" if args.hint is None else args.hint
    if args.pipeline == "sr":
        args.num_workers = int(args.num_workers or 8)
    return args


def make_job_id(args: argparse.Namespace, config_path: Path) -> str:
    prefix = args.job_name or f"{args.pipeline}_{config_path.stem}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return safe_name(f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}")


def copy_code_snapshot(destination: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SNAPSHOT_EXCLUDE_NAMES}

    shutil.copytree(REPO_ROOT, destination, ignore=ignore)


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return "git not found\n"
    text = result.stdout
    if result.stderr:
        text += result.stderr
    return text


def dump_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def patch_common_config(cfg: dict, args: argparse.Namespace, job_id: str, output_base: Path) -> dict:
    if args.data_root:
        cfg.setdefault("data", {})["root"] = args.data_root
    if args.split_dir:
        cfg.setdefault("data", {})["split_dir"] = args.split_dir
    if args.disable_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    if args.wandb_project:
        wandb_cfg = cfg.setdefault("wandb", {})
        wandb_cfg["enabled"] = True
        wandb_cfg["project"] = args.wandb_project
    if args.wandb_entity:
        cfg.setdefault("wandb", {})["entity"] = args.wandb_entity
    if cfg.get("wandb", {}).get("enabled", False):
        wandb_cfg = cfg.setdefault("wandb", {})
        wandb_cfg.setdefault("id", job_id)
        wandb_cfg.setdefault("name", job_id)
        wandb_cfg.setdefault("resume", "allow")
    if args.pipeline == "prehoc":
        cfg["run_id"] = job_id
        cfg["output_dir"] = str(output_base)
        cfg["device"] = "cuda:0"
        if args.disable_pretrained:
            cfg.setdefault("model", {})["resnet_pretrained"] = False
    else:
        train_cfg = cfg.setdefault("train", {})
        train_cfg["run_id"] = job_id
        train_cfg["output_dir"] = str(output_base)
    return cfg


def download_risk_warnings(cfg: dict, pipeline: str) -> list[str]:
    warnings = []
    if pipeline == "prehoc" and cfg.get("model", {}).get("resnet_pretrained", False):
        warnings.append("model.resnet_pretrained=true requires cached TorchVision weights on compute nodes.")
    for metric in ("mind", "fid"):
        metric_cfg = cfg.get("evaluation", {}).get(metric, {}) or {}
        if metric_cfg.get("enabled", False) and str(metric_cfg.get("weights", "DEFAULT")).upper() == "DEFAULT":
            warnings.append(f"evaluation.{metric}.weights=DEFAULT requires cached TorchVision Inception weights.")
    autoencoder_cfg = cfg.get("autoencoder", {}) or {}
    if autoencoder_cfg.get("target") == "diffusers.AutoencoderKL":
        warnings.append("diffusers.AutoencoderKL.from_pretrained requires cached Hugging Face weights.")
    return warnings


def build_env(scratch_root: Path) -> dict[str, str]:
    work_root = Path(os.environ.get("WORK", str(scratch_root))).expanduser().resolve()
    cache_root = work_root / ".cache"
    return {
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(scratch_root / "wandb"),
        "WANDB_CACHE_DIR": str(scratch_root / "wandb_cache"),
        "WANDB_SILENT": "true",
        "TORCH_HOME": str(cache_root / "torch"),
        "HF_HOME": str(cache_root / "huggingface"),
        "XDG_CACHE_HOME": str(cache_root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    }


def build_command(
    args: argparse.Namespace,
    cfg: dict,
    config_path: Path,
    output_base: Path,
    job_id: str,
) -> list[str]:
    extra = clean_training_args(args.training_args)
    if args.pipeline == "prehoc":
        return [
            args.python,
            "-m",
            "prehoc.run",
            "--config",
            str(config_path),
            "--run-id",
            job_id,
            "--device",
            "cuda:0",
            "--output-dir",
            str(output_base),
            *extra,
        ]

    mixed_precision = args.mixed_precision or str(cfg.get("train", {}).get("mixed_precision", "no"))
    return [
        args.python,
        "-m",
        "accelerate.commands.launch",
        "--module",
        "--num_processes",
        str(args.gpus),
        "--mixed_precision",
        mixed_precision,
        "sr.train_pixel",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_base),
        "--run-id",
        job_id,
        "--num-workers",
        str(args.num_workers),
        *extra,
    ]


def create_bundle(args: argparse.Namespace) -> tuple[Path, list[str], dict[str, str], str]:
    scratch_root = default_scratch_root(args.scratch_root)
    config_source = repo_path(args.config).resolve()
    if not config_source.is_file():
        raise SystemExit(f"Missing config: {config_source}")

    job_id = make_job_id(args, config_source)
    bundle_dir = scratch_root / "jobs" / job_id
    code_dir = bundle_dir / "code"
    output_base = Path(args.output_root).expanduser().resolve() if args.output_root else scratch_root / "runs" / args.pipeline / config_source.stem
    bundle_dir.mkdir(parents=True, exist_ok=False)
    output_base.mkdir(parents=True, exist_ok=True)
    copy_code_snapshot(code_dir)

    cfg = patch_common_config(load_yaml(config_source), args, job_id, output_base)
    bundled_config = bundle_dir / "config.yaml"
    write_yaml(bundled_config, cfg)

    env = build_env(scratch_root)
    command = build_command(args, cfg, bundled_config, output_base, job_id)
    warnings = download_risk_warnings(cfg, args.pipeline)

    dump_text(bundle_dir / "command.txt", shlex.join(command) + "\n")
    dump_text(bundle_dir / "git_head.txt", git_output("rev-parse", "HEAD"))
    dump_text(bundle_dir / "git_status.txt", git_output("status", "--short"))
    dump_text(bundle_dir / "git_diff.patch", git_output("diff"))
    dump_text(bundle_dir / "git_diff_staged.patch", git_output("diff", "--staged"))
    dump_text(bundle_dir / "warnings.txt", "\n".join(warnings) + ("\n" if warnings else ""))
    dump_text(bundle_dir / "wandb_sync.txt", f"wandb sync {env['WANDB_DIR']}/offline-run-*\n")
    with (bundle_dir / "submit_args.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable_args(args), handle, indent=2)
    with (bundle_dir / "env.json").open("w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)

    return bundle_dir, command, env, job_id


def submit_job(args: argparse.Namespace, bundle_dir: Path, command: list[str], env: dict[str, str], job_id: str) -> None:
    import submitit

    log_dir = default_scratch_root(args.scratch_root) / "slurm_logs" / job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    params = {
        "name": job_id,
        "timeout_min": int(args.time_min),
        "nodes": int(args.nodes),
        "tasks_per_node": int(args.tasks_per_node),
        "cpus_per_task": int(args.cpus_per_task),
    }
    slurm_additional_parameters = {}
    if args.gres:
        slurm_additional_parameters["gres"] = args.gres
    if args.hint:
        slurm_additional_parameters["hint"] = args.hint
    if slurm_additional_parameters:
        params["slurm_additional_parameters"] = slurm_additional_parameters
    if args.partition:
        params["slurm_partition"] = args.partition
    if args.account:
        params["slurm_account"] = args.account
    if args.qos:
        params["slurm_qos"] = args.qos
    if args.constraint:
        params["slurm_constraint"] = args.constraint
    executor.update_parameters(**params)
    runner = JobRunner(str(bundle_dir / "code"), command, env)
    job = executor.submit(runner)
    print(f"Submitted {job_id} as SLURM job {job.job_id}")
    print(f"Bundle: {bundle_dir}")
    print(f"Command: {shlex.join(command)}")
    print(f"Sync later from frontal node: wandb sync {env['WANDB_DIR']}/offline-run-*")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    parser.add_argument("--jz-config", default=None, help="Jean Zay launch YAML. CLI values override YAML values.")
    parser.add_argument("--scratch-root", default=None, help="Default: $SCRATCH/signal_sr")
    parser.add_argument("--output-root", default=None, help="Base output directory. Default: <scratch-root>/runs/<pipeline>/<config-stem>")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--account", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time-min", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--tasks-per-node", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem-gb", type=int, default=None, help="Ignored on Jean Zay; memory is controlled by --cpus-per-task.")
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--hint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Create the immutable bundle and print the command without submitting.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit immutable Signal SR jobs on Jean Zay with offline W&B.")
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    prehoc = subparsers.add_parser("prehoc")
    add_common_args(prehoc)
    prehoc.add_argument("--disable-pretrained", action="store_true")
    prehoc.add_argument("training_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to prehoc.run.")

    sr = subparsers.add_parser("sr")
    add_common_args(sr)
    sr.add_argument("--num-workers", type=int, default=None)
    sr.add_argument("--mixed-precision", default=None)
    sr.add_argument("training_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to sr.train_pixel.")
    return parser.parse_args()


def main() -> None:
    args = normalize_args(parse_args())
    bundle_dir, command, env, job_id = create_bundle(args)
    print(f"Bundle: {bundle_dir}")
    print(f"Command: {shlex.join(command)}")
    warnings = (bundle_dir / "warnings.txt").read_text(encoding="utf-8").strip()
    if warnings:
        print(f"Warnings:\n{warnings}")
    if args.dry_run:
        print("Dry run: no SLURM job submitted.")
        print(f"Sync later from frontal node: wandb sync {env['WANDB_DIR']}/offline-run-*")
        return
    submit_job(args, bundle_dir, command, env, job_id)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dict with attribute access for YAML configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = _to_config(data)
    if "schedule" in cfg or "scheduler" in cfg:
        raise ValueError("Use the canonical top-level key `noise_scheduler`, not `schedule` or `scheduler`.")
    if "noise_scheduler" not in cfg:
        raise ValueError("Config must define `noise_scheduler`.")
    return cfg


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    return value


def merge_dict(base: Config, override: dict | None) -> Config:
    out = Config({k: _to_config(v) for k, v in base.items()})
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = _to_config(value)
    return out

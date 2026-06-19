from __future__ import annotations

import importlib
from typing import Any


def import_string(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Expected a full import path, got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def build_from_config(cfg):
    cls = import_string(cfg.target)
    params = cfg.get("params", {})
    return cls(**params)

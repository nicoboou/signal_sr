from __future__ import annotations

from ..utils.import_utils import build_from_config
from .conditioning import build_conditioner as _build_conditioner
from .denoiser_adapter import DenoiserAdapter, validate_denoiser


def build_denoiser(cfg, data_channels=None):
    model = build_from_config(cfg)
    validate_denoiser(model, data_channels=data_channels)
    return DenoiserAdapter(model)


def build_conditioner():
    return _build_conditioner()

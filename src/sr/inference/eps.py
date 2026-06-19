from __future__ import annotations

import torch
import torch.nn.functional as F


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _expand_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while x.ndim < target.ndim:
        x = x[..., None]
    return x


def eps_config(cfg):
    sampling = _cfg_get(cfg, "sampling", {})
    return _cfg_get(sampling, "eps", {})


def eps_sigma_y(cfg) -> float:
    eps_cfg = eps_config(cfg)
    measurement_cfg = _cfg_get(cfg, "measurement", {})
    sigma_y = float(_cfg_get(eps_cfg, "sigma_y", _cfg_get(measurement_cfg, "sigma_y", 0.05)))
    if sigma_y <= 0.0:
        raise ValueError("EPS requires sampling.eps.sigma_y > 0")
    return sigma_y


def validate_eps_config(cfg, objective=None, denoiser_mode=None):
    if _cfg_get(cfg, "space", "pixel") != "pixel":
        raise ValueError("EPS currently supports only space=pixel")
    if objective is not None:
        if objective.name != "diffusion" or objective.prediction_type != "sample":
            raise ValueError("EPS requires objective.name=diffusion and prediction_type=sample")
    if denoiser_mode is not None and denoiser_mode != "concat":
        raise ValueError("EPS requires channel-concat conditioning with [mu_star, y_up]")

    eps_cfg = eps_config(cfg)
    operator = _cfg_get(eps_cfg, "operator", "super_resolution")
    if operator != "super_resolution":
        raise ValueError("EPS currently supports only operator=super_resolution")

    measurement_cfg = _cfg_get(cfg, "measurement", {})
    downsample_mode = _cfg_get(eps_cfg, "downsample_mode", _cfg_get(measurement_cfg, "downsample_mode", "nearest"))
    if downsample_mode != "nearest":
        raise ValueError("EPS currently supports only nearest downsampling")


def nearest_downsample(x: torch.Tensor, lr_size: int) -> torch.Tensor:
    return F.interpolate(x, size=(int(lr_size), int(lr_size)), mode="nearest")


def nearest_upsample(x_lr: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(x_lr, size=(int(image_size), int(image_size)), mode="nearest")


def nearest_adjoint(x_lr: torch.Tensor, image_size: int) -> torch.Tensor:
    if x_lr.ndim != 4:
        raise ValueError(f"Expected LR tensor [B,C,H,W], got {tuple(x_lr.shape)}")
    image_size = int(image_size)
    lr_h, lr_w = x_lr.shape[-2:]
    if image_size % lr_h != 0 or image_size % lr_w != 0:
        raise ValueError(f"image_size={image_size} must be divisible by LR shape {(lr_h, lr_w)}")
    step_h = image_size // lr_h
    step_w = image_size // lr_w
    out = x_lr.new_zeros((*x_lr.shape[:2], image_size, image_size))
    out[..., ::step_h, ::step_w] = x_lr
    return out


def nearest_observation_mask(x_lr: torch.Tensor, image_size: int) -> torch.Tensor:
    return nearest_adjoint(torch.ones_like(x_lr[:, :1]), image_size)


def eps_observation_from_hr(x0: torch.Tensor, cfg, noisy: bool = False) -> torch.Tensor:
    lr_size = int(_cfg_get(cfg, "lr_size", int(_cfg_get(cfg, "image_size")) // int(_cfg_get(cfg, "scale"))))
    y_lr = nearest_downsample(x0, lr_size)
    if noisy:
        y_lr = y_lr + eps_sigma_y(cfg) * torch.randn_like(y_lr)
    return y_lr


def eps_pivot(x_t: torch.Tensor, y_lr: torch.Tensor, timesteps, noise_scheduler, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    validate_eps_config(cfg)
    image_size = int(x_t.shape[-1])
    if x_t.shape[-2] != image_size:
        raise ValueError("EPS nearest SR pivot expects square HR tensors")

    y_lr = y_lr.to(device=x_t.device, dtype=x_t.dtype)
    if y_lr.shape[:2] != x_t.shape[:2]:
        raise ValueError(f"Measurement shape {tuple(y_lr.shape)} is incompatible with state {tuple(x_t.shape)}")

    x_float = x_t.float()
    y_float = y_lr.float()
    alpha_bar = noise_scheduler.alpha_bar(timesteps).to(device=x_t.device, dtype=torch.float32).clamp(1e-8, 1.0 - 1e-8)
    beta2 = (1.0 - alpha_bar).clamp_min(1e-8)
    alpha = alpha_bar.sqrt()

    current_precision = _expand_to(alpha_bar / beta2, x_float)
    current_weight = _expand_to(alpha / beta2, x_float)
    measurement_precision = 1.0 / (eps_sigma_y(cfg) ** 2)

    aty = nearest_adjoint(y_float, image_size)
    mask = nearest_observation_mask(y_float, image_size)
    mu_star = (current_weight * x_float + measurement_precision * aty) / (current_precision + measurement_precision * mask).clamp_min(1e-8)
    y_up = nearest_upsample(y_lr, image_size)
    return mu_star.to(dtype=x_t.dtype), y_up

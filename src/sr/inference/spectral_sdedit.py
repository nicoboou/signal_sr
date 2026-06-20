from __future__ import annotations

import torch
import torch.nn.functional as F

from ..schedules.spectral import (
    dct2,
    dct_rapsd_from_coefficients,
    find_t_star_logfreq_budget,
    fit_power_law,
    idct2,
    make_frequency_radius_grid,
    make_reliability_mask,
)


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _spectral_cfg(cfg):
    if cfg is None:
        return {}
    section = _cfg_get(cfg, "spectral_sdedit", None)
    if section is not None:
        return section
    sampling = _cfg_get(cfg, "sampling", None)
    if sampling is not None:
        return _cfg_get(sampling, "spectral_sdedit", {})
    return {}


def _interpolate_to_hr(x_lr: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode in {"nearest", "area"}:
        return F.interpolate(x_lr, size=size, mode=mode)
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x_lr, size=size, mode=mode, align_corners=False)
    raise ValueError(f"Unsupported upsample_mode: {mode}")


def _masked_frequency_mean(values: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
    if not bool(mask_2d.any()):
        return torch.full((values.shape[0],), float("nan"), device=values.device, dtype=values.dtype)
    return values[:, :, mask_2d].reshape(values.shape[0], -1).mean(dim=1)


@torch.no_grad()
def spectral_sdedit_init(
    x_lr: torch.Tensor,
    hr_scheduler,
    scale_r: int | float,
    image_size: int | tuple[int, int],
    rho: float = 0.75,
    m_min: float = 0.0,
    m_max: float = 0.85,
    num_log_bins: int = 64,
    transform: str = "dct",
    upsample_mode: str = "nearest",
    init_formula: str = "scheduler_compatible",
):
    """Build a scheduler-compatible spectral SDEdit initialization.

    The spectral mask defines a clean proxy image x_proxy = IDCT(m * DCT(x_lr_up)).
    We then forward-noise x_proxy using the standard scalar HR scheduler at t_star.
    This keeps z_init compatible with the distribution expected by the HR denoiser.
    """
    if transform != "dct":
        raise ValueError("spectral_sdedit_init currently supports transform='dct' only")
    if x_lr.ndim != 4:
        raise ValueError(f"Expected x_lr [B,C,H,W], got {tuple(x_lr.shape)}")
    if scale_r <= 0:
        raise ValueError("scale_r must be positive")
    if hr_scheduler.timesteps is None:
        raise ValueError("Call hr_scheduler.set_timesteps(...) before spectral_sdedit_init")

    size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(int(v) for v in image_size)
    if len(size) != 2:
        raise ValueError("image_size must be an int or (H,W)")
    x_lr_up = _interpolate_to_hr(x_lr, size=size, mode=upsample_mode)
    work_dtype = torch.float64 if x_lr_up.dtype == torch.float64 else torch.float32
    x_lr_up = x_lr_up.to(dtype=work_dtype)
    b, _, h, w = x_lr_up.shape

    x_freq = dct2(x_lr_up)
    k, psd = dct_rapsd_from_coefficients(x_freq)
    slope, beta = fit_power_law(k, psd)
    k_2d = make_frequency_radius_grid(h, w, device=x_lr_up.device, dtype=work_dtype, transform=transform)
    t_star, t_stats = find_t_star_logfreq_budget(
        scheduler=hr_scheduler,
        timesteps=hr_scheduler.timesteps,
        slope=slope.to(work_dtype),
        beta=beta.to(work_dtype),
        k_2d=k_2d,
        scale_r=float(scale_r),
        rho=float(rho),
        num_log_bins=int(num_log_bins),
    )
    mask, mask_stats = make_reliability_mask(
        scheduler=hr_scheduler,
        t_star=t_star,
        slope=slope.to(work_dtype),
        beta=beta.to(work_dtype),
        k_2d=k_2d,
        m_min=float(m_min),
        m_max=float(m_max),
    )

    noise_freq = torch.randn_like(x_freq)
    alpha_star = t_stats["alpha_star"].to(device=x_lr_up.device, dtype=work_dtype)
    alpha = alpha_star[:, None, None, None]
    signal_freq = alpha.sqrt() * mask * x_freq
    sigma = (1.0 - alpha).clamp_min(0.0).sqrt()
    legacy_sigma = (1.0 - alpha * mask.square()).clamp_min(0.0).sqrt()
    if init_formula == "scheduler_compatible":
        z_freq = signal_freq + sigma * noise_freq
    elif init_formula == "frequency_dependent_legacy":
        z_freq = signal_freq + legacy_sigma * noise_freq
    else:
        raise ValueError(f"Unsupported spectral_sdedit init_formula: {init_formula}")
    z_init = idct2(z_freq).to(dtype=x_lr.dtype)

    k_hr = min(h, w) / 2.0
    k_lr = k_hr / float(scale_r)
    valid_freq = (k_2d > 0) & (k_2d <= k_hr)
    low_freq = valid_freq & (k_2d <= k_lr)
    high_freq = valid_freq & (k_2d > k_lr)
    stats = {
        "t_star": t_star.detach(),
        "lambda_t_star": t_stats["lambda_t_star"].detach(),
        "alpha_star": t_stats["alpha_star"].detach(),
        "sigma_star": sigma.reshape(b, -1)[:, 0].detach(),
        "init_formula": init_formula,
        "slope": slope.detach(),
        "beta": beta.detach(),
        "scale_r": torch.full((b,), float(scale_r), device=x_lr_up.device, dtype=work_dtype),
        "rho": torch.full((b,), float(rho), device=x_lr_up.device, dtype=work_dtype),
        "K_HR": t_stats["K_HR"].detach(),
        "K_LR": t_stats["K_LR"].detach(),
        "F_target": t_stats["F_target"].detach(),
        "F_t_star": t_stats["F_t_star"].detach(),
        "F_min": t_stats["F_min"].detach(),
        "F_max": t_stats["F_max"].detach(),
        "target_out_of_range": t_stats["target_out_of_range"].detach(),
        "target_below_range": t_stats["target_below_range"].detach(),
        "target_above_range": t_stats["target_above_range"].detach(),
        "num_active_log_bins": t_stats["num_active_log_bins"].detach(),
        "num_lr_log_bins": t_stats["num_lr_log_bins"].detach(),
        "mask_mean": mask_stats["mask_mean"].detach(),
        "mask_min": mask_stats["mask_min"].detach(),
        "mask_max": mask_stats["mask_max"].detach(),
        "mask_low_freq_mean": _masked_frequency_mean(mask, low_freq).detach(),
        "mask_high_freq_mean": _masked_frequency_mean(mask, high_freq).detach(),
        "z_init_finite": torch.isfinite(z_init).reshape(b, -1).all(dim=1).to(dtype=work_dtype),
    }
    if init_formula == "frequency_dependent_legacy":
        stats["legacy_noise_scale_low_mean"] = _masked_frequency_mean(legacy_sigma, low_freq).detach()
        stats["legacy_noise_scale_high_mean"] = _masked_frequency_mean(legacy_sigma, high_freq).detach()
    return z_init, t_star, stats


@torch.no_grad()
def spectral_sdedit_sr(
    x_lr: torch.Tensor,
    batch,
    model,
    sampler,
    hr_scheduler,
    scale_r: int | float,
    cfg,
):
    """Run blind unpaired spectral SDEdit SR with an HR-domain reverse DDIM path."""
    del model
    global_cfg = getattr(sampler, "global_cfg", None)
    sdedit_cfg = _spectral_cfg(cfg) or _spectral_cfg(global_cfg)
    image_size = int(_cfg_get(cfg, "image_size", _cfg_get(global_cfg, "image_size", x_lr.shape[-1])))

    if getattr(sampler, "method", None) != "ddim":
        raise ValueError("spectral_sdedit_sr requires sampling.method='ddim' to avoid DPS/EPS LR conditioning")
    conditioning_mode = getattr(getattr(sampler, "denoiser", None), "conditioning_mode", None)
    if conditioning_mode == "concat":
        raise ValueError("spectral_sdedit_sr does not support channel-concat LR conditioning")
    allow_batch = bool(_cfg_get(sdedit_cfg, "allow_batch_size_gt_1", False))
    if x_lr.shape[0] > 1 and not allow_batch:
        raise ValueError("spectral_sdedit_sr currently supports batch size 1 unless allow_batch_size_gt_1=true")

    z_init, t_star, stats = spectral_sdedit_init(
        x_lr=x_lr,
        hr_scheduler=hr_scheduler,
        scale_r=scale_r,
        image_size=image_size,
        rho=float(_cfg_get(sdedit_cfg, "rho", 0.75)),
        m_min=float(_cfg_get(sdedit_cfg, "m_min", 0.0)),
        m_max=float(_cfg_get(sdedit_cfg, "m_max", 0.85)),
        num_log_bins=int(_cfg_get(sdedit_cfg, "num_log_bins", 64)),
        transform=_cfg_get(sdedit_cfg, "transform", "dct"),
        upsample_mode=_cfg_get(sdedit_cfg, "upsample_mode", "nearest"),
        init_formula=_cfg_get(sdedit_cfg, "init_formula", "scheduler_compatible"),
    )
    sampling_cfg = _cfg_get(cfg, "sampling", _cfg_get(global_cfg, "sampling", {}))
    clip_denoised = _cfg_get(sdedit_cfg, "clip_denoised", _cfg_get(sampling_cfg, "clip_denoised", None))
    x_sr = sampler.ddim_loop_from(
        z_init,
        batch,
        condition_domain="HR",
        conditioning_image=None,
        start_timestep=t_star,
        clip_denoised=clip_denoised,
    )
    return x_sr, z_init, stats

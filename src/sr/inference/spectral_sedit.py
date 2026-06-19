from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..schedules.spectral import find_t_star_from_lambda, fit_power_law, make_frequency_radius_grid, rapsd, soft_snr_mask, spectral_logsnr_2d


def _dct_basis(n: int, device, dtype) -> torch.Tensor:
    n = int(n)
    i = torch.arange(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype)[:, None]
    basis = torch.cos((math.pi / float(n)) * (i[None] + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / float(n))
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / float(n))
    return basis


def dct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    basis_h = _dct_basis(h, x.device, x.dtype)
    basis_w = _dct_basis(w, x.device, x.dtype)
    x = torch.einsum("kh,bchw->bckw", basis_h, x)
    return torch.einsum("lw,bckw->bckl", basis_w, x)


def idct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    basis_h = _dct_basis(h, x.device, x.dtype)
    basis_w = _dct_basis(w, x.device, x.dtype)
    x = torch.einsum("ki,bckw->bciw", basis_h, x)
    return torch.einsum("lj,bcil->bcij", basis_w, x)


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _spectral_cfg(cfg):
    if cfg is None:
        return {}
    section = _cfg_get(cfg, "spectral_sedit", None)
    if section is not None:
        return section
    sampling = _cfg_get(cfg, "sampling", None)
    if sampling is not None:
        return _cfg_get(sampling, "spectral_sedit", {})
    return {}


@torch.no_grad()
def spectral_sedit_init(
    x_lr: torch.Tensor,
    hr_scheduler,
    scale_r: int | float,
    image_size: int | tuple[int, int],
    eta: float = 1.0,
    delta: float = 1.0,
    temperature: float = 0.5,
    m_min: float = 0.0,
    m_max: float = 0.85,
    transform: str = "dct",
):
    """Build a noisy HR-compatible spectral SEdit initialization from an LR image."""
    if transform != "dct":
        raise ValueError("spectral_sedit_init currently supports transform='dct' only")
    if x_lr.ndim != 4:
        raise ValueError(f"Expected x_lr [B,C,H,W], got {tuple(x_lr.shape)}")
    if scale_r <= 0:
        raise ValueError("scale_r must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if hr_scheduler.timesteps is None:
        raise ValueError("Call hr_scheduler.set_timesteps(...) before spectral_sedit_init")

    size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(int(v) for v in image_size)
    if len(size) != 2:
        raise ValueError("image_size must be an int or (H,W)")
    x_lr_up = F.interpolate(x_lr, size=size, mode="nearest")
    work_dtype = torch.float64 if x_lr_up.dtype == torch.float64 else torch.float32
    x_lr_up = x_lr_up.to(dtype=work_dtype)
    b, _, h, w = x_lr_up.shape

    k, psd = rapsd(x_lr_up)
    slope, beta = fit_power_law(k, psd)
    k_hr = min(h, w) / 2.0
    k_anchor = float(eta) * k_hr / float(scale_r)
    k_anchor_t = torch.as_tensor(k_anchor, device=x_lr_up.device, dtype=work_dtype).clamp_min(1e-12)
    lambda_target = math.log(float(delta)) - torch.log(beta.to(dtype=work_dtype)) - slope.to(dtype=work_dtype) * torch.log(k_anchor_t)
    t_star = find_t_star_from_lambda(hr_scheduler, lambda_target, hr_scheduler.timesteps)
    lambda_star = hr_scheduler.logsnr(t_star).to(device=x_lr_up.device, dtype=work_dtype)
    alpha_star = hr_scheduler.alpha_bar(t_star).to(device=x_lr_up.device, dtype=work_dtype)

    k_2d = make_frequency_radius_grid(h, w, device=x_lr_up.device, dtype=work_dtype, transform=transform)
    logsnr_2d = spectral_logsnr_2d(lambda_star, slope.to(work_dtype), beta.to(work_dtype), k_2d)
    mask = soft_snr_mask(logsnr_2d, delta=delta, temperature=temperature, m_min=m_min, m_max=m_max)

    x_freq = dct2(x_lr_up)
    noise_freq = torch.randn_like(x_freq)
    alpha = alpha_star[:, None, None, None]
    z_freq = alpha.sqrt() * mask * x_freq + (1.0 - alpha * mask.square()).clamp_min(0.0).sqrt() * noise_freq
    z_init = idct2(z_freq).to(dtype=x_lr.dtype)

    mask_flat = mask.reshape(b, -1)
    stats = {
        "t_star": t_star.detach(),
        "lambda_star": lambda_star.detach(),
        "lambda_target": lambda_target.detach(),
        "alpha_star": alpha_star.detach(),
        "slope": slope.detach(),
        "beta": beta.detach(),
        "k_anchor": torch.full((b,), k_anchor, device=x_lr_up.device, dtype=work_dtype),
        "mask_mean": mask_flat.mean(dim=1).detach(),
        "mask_min": mask_flat.min(dim=1).values.detach(),
        "mask_max": mask_flat.max(dim=1).values.detach(),
    }
    return z_init, t_star, stats


@torch.no_grad()
def spectral_sedit_sr(
    x_lr: torch.Tensor,
    batch,
    model,
    sampler,
    hr_scheduler,
    scale_r: int | float,
    cfg,
):
    """Run unpaired spectral SEdit SR with an HR-domain unconditional reverse DDIM path."""
    del model
    global_cfg = getattr(sampler, "global_cfg", None)
    sedit_cfg = _spectral_cfg(cfg) or _spectral_cfg(global_cfg)
    image_size = int(_cfg_get(cfg, "image_size", _cfg_get(global_cfg, "image_size", x_lr.shape[-1])))
    z_init, t_star, stats = spectral_sedit_init(
        x_lr=x_lr,
        hr_scheduler=hr_scheduler,
        scale_r=scale_r,
        image_size=image_size,
        eta=float(_cfg_get(sedit_cfg, "eta", 1.0)),
        delta=float(_cfg_get(sedit_cfg, "delta", 1.0)),
        temperature=float(_cfg_get(sedit_cfg, "temperature", 0.5)),
        m_min=float(_cfg_get(sedit_cfg, "m_min", 0.0)),
        m_max=float(_cfg_get(sedit_cfg, "m_max", 0.85)),
        transform=_cfg_get(sedit_cfg, "transform", "dct"),
    )
    conditioning_image = None
    if getattr(getattr(sampler, "denoiser", None), "conditioning_mode", None) == "concat":
        conditioning_image = F.interpolate(x_lr, size=z_init.shape[-2:], mode="nearest")
    x_sr = sampler.ddim_loop_from(
        z_init,
        batch,
        condition_domain="HR",
        conditioning_image=conditioning_image,
        start_timestep=t_star,
    )
    return x_sr, z_init, stats

from __future__ import annotations

import math
import torch


def rapsd(x: torch.Tensor):
    """Radially averaged power spectral density.

    Args:
        x: Tensor with shape [B,C,H,W].

    Returns:
        k: Frequency-radius bins [K], excluding DC.
        psd: Radially averaged PSD [B,K], averaged over channels.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    b, _, h, w = x.shape
    fft = torch.fft.fft2(x.float(), norm="ortho")
    power = (fft.real.square() + fft.imag.square()).mean(dim=1)

    fy = torch.fft.fftfreq(h, device=x.device) * h
    fx = torch.fft.fftfreq(w, device=x.device) * w
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    bins = torch.round(radius).long()
    max_bin = int(bins.max().item())

    flat_bins = bins.reshape(-1)
    flat_power = power.reshape(b, -1)
    sums = torch.zeros(b, max_bin + 1, device=x.device, dtype=flat_power.dtype)
    counts = torch.zeros(max_bin + 1, device=x.device, dtype=flat_power.dtype)
    sums.scatter_add_(1, flat_bins[None].expand(b, -1), flat_power)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=flat_power.dtype))

    valid = (counts > 0) & (torch.arange(max_bin + 1, device=x.device) > 0)
    k = torch.arange(max_bin + 1, device=x.device, dtype=flat_power.dtype)[valid]
    psd = sums[:, valid] / counts[valid].clamp_min(1.0)
    return k, psd


def fit_power_law(k: torch.Tensor, psd: torch.Tensor, eps: float = 1e-12):
    """Fit log PSD(k) = slope * log(k) + log_beta for each batch item."""
    if psd.ndim != 2:
        raise ValueError(f"Expected psd [B,K], got {tuple(psd.shape)}")
    x = torch.log(k.float().clamp_min(eps))
    y = torch.log(psd.float().clamp_min(eps))
    x_centered = x - x.mean()
    y_centered = y - y.mean(dim=1, keepdim=True)
    denom = x_centered.square().sum().clamp_min(eps)
    slope = (y_centered * x_centered[None]).sum(dim=1) / denom
    intercept = y.mean(dim=1) - slope * x.mean()
    beta = torch.exp(intercept)
    slope = slope.clamp(max=-1e-4)
    beta = beta.clamp(min=eps)
    return slope, beta


def _as_batch_vector(x, *, device, dtype, name: str) -> torch.Tensor:
    x = torch.as_tensor(x, device=device, dtype=dtype)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim != 1:
        raise ValueError(f"Expected {name} to be scalar or [B], got {tuple(x.shape)}")
    return x


def _broadcast_batch_vectors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch_size = max(value.shape[0] for value in values)
    out = []
    for value in values:
        if value.shape[0] == batch_size:
            out.append(value)
        elif value.shape[0] == 1:
            out.append(value.expand(batch_size))
        else:
            raise ValueError("Batch dimensions are not broadcastable")
    return tuple(out)


def spectral_logsnr_2d(lambda_t, slope, beta, k_2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute lambda(t) + log(beta) + slope * log(k) on a 2D frequency grid.

    Returns a tensor with shape [B,1,H,W], where scalar inputs are broadcast to B.
    """
    if k_2d.ndim != 2:
        raise ValueError(f"Expected k_2d [H,W], got {tuple(k_2d.shape)}")
    lambda_t = _as_batch_vector(lambda_t, device=k_2d.device, dtype=k_2d.dtype, name="lambda_t")
    slope = _as_batch_vector(slope, device=k_2d.device, dtype=k_2d.dtype, name="slope")
    beta = _as_batch_vector(beta, device=k_2d.device, dtype=k_2d.dtype, name="beta")
    lambda_t, slope, beta = _broadcast_batch_vectors(lambda_t, slope, beta)
    return (
        lambda_t[:, None, None, None]
        + torch.log(beta.clamp_min(eps))[:, None, None, None]
        + slope[:, None, None, None] * torch.log(k_2d.clamp_min(eps))[None, None]
    )


def make_frequency_radius_grid(h: int, w: int, device=None, dtype=None, transform: str = "dct") -> torch.Tensor:
    """Build a [H,W] frequency-radius grid in FFT-like pixel-frequency units."""
    h = int(h)
    w = int(w)
    dtype = dtype or torch.float32
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    if transform == "dct":
        fy = torch.arange(h, device=device, dtype=dtype) / 2.0
        fx = torch.arange(w, device=device, dtype=dtype) / 2.0
    elif transform == "fft":
        fy = torch.fft.fftfreq(h, device=device).to(dtype=dtype) * h
        fx = torch.fft.fftfreq(w, device=device).to(dtype=dtype) * w
    else:
        raise ValueError(f"Unsupported frequency transform: {transform}")
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(xx.square() + yy.square())


def soft_snr_mask(
    logsnr_2d: torch.Tensor,
    delta: float = 1.0,
    temperature: float = 0.5,
    m_min: float = 0.0,
    m_max: float = 0.85,
) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (0.0 <= m_min <= m_max < 1.0):
        raise ValueError("Expected 0 <= m_min <= m_max < 1")
    threshold = math.log(float(delta))
    mask = torch.sigmoid((logsnr_2d - threshold) / float(temperature))
    return mask.clamp(float(m_min), float(m_max))


def find_t_star_from_lambda(scheduler, lambda_star, timesteps=None) -> torch.Tensor:
    """Return the scheduler timestep whose logSNR is closest to lambda_star."""
    if timesteps is None:
        timesteps = scheduler.timesteps
    if timesteps is None:
        raise ValueError("Call scheduler.set_timesteps(...) or pass timesteps before finding t_star")
    lambda_star = torch.as_tensor(lambda_star, device=timesteps.device, dtype=timesteps.dtype)
    if lambda_star.ndim == 0:
        lambda_star = lambda_star.reshape(1)
    elif lambda_star.ndim != 1:
        raise ValueError(f"Expected lambda_star scalar or [B], got {tuple(lambda_star.shape)}")
    lambda_grid = scheduler.logsnr(timesteps.to(device=lambda_star.device, dtype=lambda_star.dtype))
    if lambda_grid.ndim != 1:
        lambda_grid = lambda_grid.reshape(-1)
    closest = (lambda_grid[None] - lambda_star[:, None]).abs().argmin(dim=1)
    return timesteps.to(device=lambda_star.device)[closest]


def spectral_logsnr(
    tau: torch.Tensor,
    slope: torch.Tensor,
    beta: torch.Tensor,
    n_freq: int,
    kind: str = "mixed",
    kappa_min: float = 0.2,
    kappa_max: float = 200.0,
    eps: float = 1e-12,
):
    tau = tau.to(device=slope.device, dtype=slope.dtype).clamp(0.0, 1.0)
    if tau.ndim == 0:
        tau = tau.expand_as(slope)
    while slope.ndim < tau.ndim:
        slope = slope.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

    n_freq_t = torch.as_tensor(float(n_freq), device=tau.device, dtype=tau.dtype)
    log_kappa = (1.0 - tau) * math.log(kappa_min) + tau * math.log(kappa_max)
    log_beta = torch.log(beta.clamp_min(eps))

    mu_f = n_freq_t + (1.0 - n_freq_t) * tau
    lambda_f = -log_kappa - log_beta - slope * torch.log(mu_f.clamp_min(eps))

    exponent = slope + 1.0
    safe_exponent = torch.where(exponent.abs() < 1e-3, torch.ones_like(exponent), exponent)
    regular = torch.pow(
        (1.0 + (1.0 - tau) * (torch.pow(n_freq_t, safe_exponent) - 1.0)).clamp_min(eps),
        1.0 / safe_exponent,
    )
    fallback = torch.pow(n_freq_t, 1.0 - tau)
    mu_p = torch.where(exponent.abs() < 1e-3, fallback, regular)
    lambda_p = -log_kappa - log_beta - slope * torch.log(mu_p.clamp_min(eps))

    if kind == "frequency":
        return lambda_f
    if kind == "power":
        return lambda_p
    if kind == "mixed":
        return 0.5 * (lambda_f + lambda_p)
    raise ValueError(f"Unsupported spectral kind: {kind}")

from __future__ import annotations

import math
import torch


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
    """Orthonormal 2D DCT-II for tensors with shape [B,C,H,W]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    basis_h = _dct_basis(h, x.device, x.dtype)
    basis_w = _dct_basis(w, x.device, x.dtype)
    x = torch.einsum("kh,bchw->bckw", basis_h, x)
    return torch.einsum("lw,bckw->bckl", basis_w, x)


def idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2` for orthonormal DCT coefficients."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    basis_h = _dct_basis(h, x.device, x.dtype)
    basis_w = _dct_basis(w, x.device, x.dtype)
    x = torch.einsum("ki,bckw->bciw", basis_h, x)
    return torch.einsum("lj,bcil->bcij", basis_w, x)


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


def dct_rapsd_from_coefficients(coeffs: torch.Tensor, eps: float = 1e-12):
    """DCT-domain radially averaged PSD, excluding DC and radii above HR Nyquist.

    Args:
        coeffs: DCT coefficients with shape [B,C,H,W].

    Returns:
        k: Mean DCT radius per radial bin [K], excluding DC.
        psd: Channel-averaged radial PSD [B,K].
    """
    if coeffs.ndim != 4:
        raise ValueError(f"Expected DCT coefficients [B,C,H,W], got {tuple(coeffs.shape)}")
    b, _, h, w = coeffs.shape
    power = coeffs.float().square().mean(dim=1)
    k_2d = make_frequency_radius_grid(h, w, device=coeffs.device, dtype=power.dtype, transform="dct", eps=eps)
    k_hr = min(h, w) / 2.0
    valid = (k_2d > 0) & (k_2d <= k_hr)
    if not bool(valid.any()):
        raise ValueError("No non-DC DCT frequencies are available for RAPSD")

    # DCT radii live on a half-integer grid under the repo convention k_i=i/2.
    bins = torch.round(k_2d * 2.0).long()
    max_bin = int(torch.ceil(torch.as_tensor(k_hr * 2.0, device=coeffs.device)).item())
    flat_bins = bins[valid].reshape(-1)
    flat_k = k_2d[valid].reshape(-1)
    flat_power = power.reshape(b, -1)[:, valid.reshape(-1)]

    sums = torch.zeros(b, max_bin + 1, device=coeffs.device, dtype=flat_power.dtype)
    counts = torch.zeros(max_bin + 1, device=coeffs.device, dtype=flat_power.dtype)
    k_sums = torch.zeros(max_bin + 1, device=coeffs.device, dtype=flat_power.dtype)
    sums.scatter_add_(1, flat_bins[None].expand(b, -1), flat_power)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=flat_power.dtype))
    k_sums.scatter_add_(0, flat_bins, flat_k.to(dtype=flat_power.dtype))

    valid_bins = counts > 0
    k = k_sums[valid_bins] / counts[valid_bins].clamp_min(1.0)
    psd = sums[:, valid_bins] / counts[valid_bins].clamp_min(1.0)
    return k.clamp_min(eps), psd


def dct_rapsd(x: torch.Tensor, eps: float = 1e-12):
    """DCT-domain RAPSD for images with shape [B,C,H,W]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    return dct_rapsd_from_coefficients(dct2(x.float()), eps=eps)


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


def make_frequency_radius_grid(h: int, w: int, device=None, dtype=None, transform: str = "dct", eps: float = 1e-12) -> torch.Tensor:
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


def spectral_power_law_2d(slope: torch.Tensor, beta: torch.Tensor, k_2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute P(k)=beta*k**slope on a 2D frequency grid.

    Returns a tensor with shape [B,1,H,W], where scalar inputs are broadcast to B.
    """
    if k_2d.ndim != 2:
        raise ValueError(f"Expected k_2d [H,W], got {tuple(k_2d.shape)}")
    slope = _as_batch_vector(slope, device=k_2d.device, dtype=k_2d.dtype, name="slope")
    beta = _as_batch_vector(beta, device=k_2d.device, dtype=k_2d.dtype, name="beta")
    slope, beta = _broadcast_batch_vectors(slope, beta)
    log_power = torch.log(beta.clamp_min(eps))[:, None, None, None] + slope[:, None, None, None] * torch.log(k_2d.clamp_min(eps))[None, None]
    return torch.exp(log_power.clamp(-80.0, 80.0))


def spectral_snr_2d(lambda_t: torch.Tensor, slope: torch.Tensor, beta: torch.Tensor, k_2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute SNR(t,k)=exp(lambda(t))*beta*k**slope on a 2D grid."""
    logsnr = spectral_logsnr_2d(lambda_t, slope, beta, k_2d, eps=eps)
    return torch.exp(logsnr.clamp(-80.0, 80.0))


def spectral_reliability(snr: torch.Tensor) -> torch.Tensor:
    """Compute w=snr/(1+snr), clamped to [0,1] for numerical safety."""
    snr = snr.clamp_min(0.0)
    return (1.0 - torch.reciprocal(1.0 + snr)).clamp(0.0, 1.0)


def _valid_log_frequency_mask(k_2d: torch.Tensor) -> torch.Tensor:
    if k_2d.ndim != 2:
        raise ValueError(f"Expected k_2d [H,W], got {tuple(k_2d.shape)}")
    h, w = k_2d.shape
    k_hr = min(h, w) / 2.0
    return (k_2d > 0) & (k_2d <= k_hr)


def log_frequency_bin_average(values_2d: torch.Tensor, k_2d: torch.Tensor, num_bins: int = 64, eps: float = 1e-12):
    """Average values over logarithmic radial-frequency bins using log(1+k).

    Args:
        values_2d: Tensor with shape [B,H,W] or [B,1,H,W].
        k_2d: Frequency-radius grid [H,W]. DC and k>min(H,W)/2 are excluded.

    Returns:
        F: [B], mean of non-empty per-bin means.
        stats: dict with bin centers, counts, active mask, and per-bin means.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if values_2d.ndim == 4:
        if values_2d.shape[1] == 1:
            values = values_2d[:, 0]
        else:
            values = values_2d.mean(dim=1)
    elif values_2d.ndim == 3:
        values = values_2d
    else:
        raise ValueError(f"Expected values_2d [B,H,W] or [B,1,H,W], got {tuple(values_2d.shape)}")
    if values.shape[-2:] != k_2d.shape:
        raise ValueError(f"Values spatial shape {tuple(values.shape[-2:])} does not match k_2d {tuple(k_2d.shape)}")

    b = values.shape[0]
    valid = _valid_log_frequency_mask(k_2d)
    if not bool(valid.any()):
        raise ValueError("No non-DC frequencies are available for log-frequency binning")

    k_valid = k_2d[valid].to(dtype=values.dtype)
    k_min = k_valid.min().clamp_min(eps)
    k_hr = torch.as_tensor(min(k_2d.shape) / 2.0, device=k_2d.device, dtype=values.dtype)
    edges_log = torch.linspace(torch.log1p(k_min), torch.log1p(k_hr), int(num_bins) + 1, device=k_2d.device, dtype=values.dtype)
    centers = torch.expm1(0.5 * (edges_log[:-1] + edges_log[1:]))
    flat_bins = torch.bucketize(torch.log1p(k_valid), edges_log, right=False) - 1
    flat_bins = flat_bins.clamp(0, int(num_bins) - 1).long()

    flat_values = values.reshape(b, -1)[:, valid.reshape(-1)]
    sums = torch.zeros(b, int(num_bins), device=values.device, dtype=values.dtype)
    counts = torch.zeros(int(num_bins), device=values.device, dtype=values.dtype)
    sums.scatter_add_(1, flat_bins[None].expand(b, -1), flat_values)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=values.dtype))

    active = counts > 0
    per_bin_means = sums / counts.clamp_min(1.0)[None]
    per_bin_means = torch.where(active[None], per_bin_means, torch.full_like(per_bin_means, float("nan")))
    if not bool(active.any()):
        raise ValueError("All log-frequency bins are empty")
    F = per_bin_means[:, active].mean(dim=1)
    stats = {
        "bin_edges": torch.expm1(edges_log).detach(),
        "bin_centers": centers.detach(),
        "bin_counts": counts.detach(),
        "active_bins": active.detach(),
        "per_bin_means": per_bin_means.detach(),
        "K_HR": k_hr.detach(),
    }
    return F, stats


def find_t_star_logfreq_budget(
    scheduler,
    timesteps: torch.Tensor,
    slope: torch.Tensor,
    beta: torch.Tensor,
    k_2d: torch.Tensor,
    scale_r: float,
    rho: float = 0.75,
    num_log_bins: int = 64,
):
    """Choose t_star by matching a log-frequency retained-reliability budget."""
    if timesteps is None:
        raise ValueError("Call scheduler.set_timesteps(...) before finding t_star")
    if scale_r <= 0:
        raise ValueError("scale_r must be positive")
    if not (0.0 <= float(rho) <= 1.0):
        raise ValueError("rho must be in [0,1]")

    dtype = k_2d.dtype
    device = k_2d.device
    slope = _as_batch_vector(slope, device=device, dtype=dtype, name="slope")
    beta = _as_batch_vector(beta, device=device, dtype=dtype, name="beta")
    slope, beta = _broadcast_batch_vectors(slope, beta)
    b = slope.shape[0]
    timesteps = timesteps.to(device=device)

    _, bin_stats = log_frequency_bin_average(torch.ones(b, 1, *k_2d.shape, device=device, dtype=dtype), k_2d, num_bins=num_log_bins)
    active_bins = bin_stats["active_bins"]
    active_centers = bin_stats["bin_centers"][active_bins]
    n_active = active_centers.numel()
    k_hr = torch.as_tensor(min(k_2d.shape) / 2.0, device=device, dtype=dtype)
    k_lr = k_hr / float(scale_r)
    n_lr_bins = (active_centers <= k_lr).sum()
    F_target_scalar = torch.as_tensor(float(rho), device=device, dtype=dtype) * n_lr_bins.to(dtype=dtype) / max(int(n_active), 1)
    F_target = F_target_scalar.expand(b)

    per_t_F = []
    for timestep in timesteps:
        lambda_t = scheduler.logsnr(timestep.to(device=device)).to(device=device, dtype=dtype)
        logsnr_2d = spectral_logsnr_2d(lambda_t, slope, beta, k_2d)
        reliability = torch.sigmoid(logsnr_2d)
        F_t, _ = log_frequency_bin_average(reliability, k_2d, num_bins=num_log_bins)
        per_t_F.append(F_t)
    per_t_F = torch.stack(per_t_F, dim=1)
    closest = (per_t_F - F_target[:, None]).abs().argmin(dim=1)
    t_star = timesteps[closest]
    F_t_star = per_t_F[torch.arange(b, device=device), closest]
    lambda_t_star = scheduler.logsnr(t_star).to(device=device, dtype=dtype)
    alpha_star = scheduler.alpha_bar(t_star).to(device=device, dtype=dtype)

    F_min = per_t_F.min(dim=1).values
    F_max = per_t_F.max(dim=1).values
    target_below = F_target < F_min
    target_above = F_target > F_max
    target_out_of_range = target_below | target_above
    stats = {
        "F_target": F_target.detach(),
        "F_t_star": F_t_star.detach(),
        "lambda_t_star": lambda_t_star.detach(),
        "alpha_star": alpha_star.detach(),
        "K_HR": k_hr.expand(b).detach(),
        "K_LR": k_lr.expand(b).detach(),
        "per_t_F": per_t_F.detach(),
        "F_min": F_min.detach(),
        "F_max": F_max.detach(),
        "target_out_of_range": target_out_of_range.to(dtype=dtype).detach(),
        "target_below_range": target_below.to(dtype=dtype).detach(),
        "target_above_range": target_above.to(dtype=dtype).detach(),
        "num_active_log_bins": torch.full((b,), int(n_active), device=device, dtype=dtype),
        "num_lr_log_bins": n_lr_bins.to(device=device, dtype=dtype).expand(b).detach(),
        "bin_centers": bin_stats["bin_centers"].detach(),
        "active_bins": active_bins.detach(),
    }
    return t_star, stats


def make_reliability_mask(
    scheduler,
    t_star: torch.Tensor,
    slope: torch.Tensor,
    beta: torch.Tensor,
    k_2d: torch.Tensor,
    m_min: float = 0.0,
    m_max: float = 0.85,
):
    """Build m(k)=clamp(sqrt(w(t_star,k)), m_min, m_max)."""
    if not (0.0 <= float(m_min) <= float(m_max) <= 1.0):
        raise ValueError("Expected 0 <= m_min <= m_max <= 1")
    dtype = k_2d.dtype
    device = k_2d.device
    t_star = torch.as_tensor(t_star, device=device)
    t_star_dtype = dtype if t_star.is_floating_point() else torch.float32
    t_star = _as_batch_vector(t_star, device=device, dtype=t_star_dtype, name="t_star")
    slope = _as_batch_vector(slope, device=device, dtype=dtype, name="slope")
    beta = _as_batch_vector(beta, device=device, dtype=dtype, name="beta")
    t_star, slope, beta = _broadcast_batch_vectors(t_star.to(dtype=dtype), slope, beta)
    lambda_t = scheduler.logsnr(t_star).to(device=device, dtype=dtype)
    logsnr_2d = spectral_logsnr_2d(lambda_t, slope, beta, k_2d)
    reliability = torch.sigmoid(logsnr_2d)
    mask = reliability.sqrt().clamp(float(m_min), float(m_max))
    flat = mask.reshape(mask.shape[0], -1)
    stats = {
        "mask_mean": flat.mean(dim=1).detach(),
        "mask_min": flat.min(dim=1).values.detach(),
        "mask_max": flat.max(dim=1).values.detach(),
    }
    return mask, stats


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

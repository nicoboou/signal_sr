from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _dct_basis(n: int, device, dtype) -> torch.Tensor:
    n = int(n)
    i = torch.arange(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype)[:, None]
    basis = torch.cos((math.pi / float(n)) * (i[None] + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / float(n))
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / float(n))
    return basis


def dct2(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Orthonormal 2D DCT-II for tensors with shape [B,C,H,W]."""
    if norm != "ortho":
        raise ValueError("dct2 currently supports norm='ortho' only")
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    basis_h = _dct_basis(h, x.device, x.dtype)
    basis_w = _dct_basis(w, x.device, x.dtype)
    x = torch.einsum("kh,bchw->bckw", basis_h, x)
    return torch.einsum("lw,bckw->bckl", basis_w, x)


def idct2(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Inverse of :func:`dct2` for orthonormal DCT coefficients."""
    if norm != "ortho":
        raise ValueError("idct2 currently supports norm='ortho' only")
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


def _target_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError("size must be an int or (H,W)")
    return int(size[0]), int(size[1])


def make_dct_frequency_grid(H: int, W: int, device=None, dtype=None) -> torch.Tensor:
    """Build a DCT radial-frequency grid with k[p,q]=sqrt((p/2)^2+(q/2)^2).

    The DC entry is set to the smallest positive DCT radius so callers can take
    logs safely. Fitting and quality metrics should still explicitly exclude DC.
    """
    H = int(H)
    W = int(W)
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    dtype = dtype or torch.float32
    p = torch.arange(H, device=device, dtype=dtype) / 2.0
    q = torch.arange(W, device=device, dtype=dtype) / 2.0
    pp, qq = torch.meshgrid(p, q, indexing="ij")
    k = torch.sqrt(pp.square() + qq.square())
    positive = k > 0
    if bool(positive.any()):
        k = k.clone()
        k[0, 0] = k[positive].min()
    return k


def _dct_non_dc_mask(H: int, W: int, device=None) -> torch.Tensor:
    p = torch.arange(int(H), device=device)[:, None]
    q = torch.arange(int(W), device=device)[None]
    return (p != 0) | (q != 0)


def _dct_log_bin_setup(
    H: int,
    W: int,
    num_log_bins: int,
    device=None,
    dtype=None,
    eps: float = 1e-8,
    bin_edges: torch.Tensor | None = None,
):
    if num_log_bins <= 0:
        raise ValueError("num_log_bins must be positive")
    H = int(H)
    W = int(W)
    dtype = dtype or torch.float32
    k_2d = make_dct_frequency_grid(H, W, device=device, dtype=dtype)
    non_dc = _dct_non_dc_mask(H, W, device=device)
    k_hr = torch.as_tensor(min(H, W) / 2.0, device=device, dtype=dtype)
    valid = non_dc & (k_2d <= k_hr)
    if not bool(valid.any()):
        raise ValueError("No non-DC DCT frequencies are available")

    if bin_edges is None:
        k_min = k_2d[valid].min().clamp_min(eps)
        edges = torch.exp(torch.linspace(torch.log(k_min), torch.log(k_hr.clamp_min(k_min)), int(num_log_bins) + 1, device=device, dtype=dtype))
        edges[0] = k_min
        edges[-1] = k_hr.clamp_min(k_min)
    else:
        edges = torch.as_tensor(bin_edges, device=device, dtype=dtype)
        if edges.numel() != int(num_log_bins) + 1:
            raise ValueError(f"Expected {int(num_log_bins) + 1} bin edges, got {edges.numel()}")
    centers = torch.exp(0.5 * (torch.log(edges[:-1].clamp_min(eps)) + torch.log(edges[1:].clamp_min(eps))))
    flat_bins = torch.bucketize(k_2d[valid].contiguous(), edges.contiguous(), right=False) - 1
    flat_bins = flat_bins.clamp(0, int(num_log_bins) - 1).long()
    counts = torch.zeros(int(num_log_bins), device=device, dtype=dtype)
    counts.scatter_add_(0, flat_bins, torch.ones_like(flat_bins, dtype=dtype))
    return k_2d, valid, edges, centers, flat_bins, counts, k_hr


def hard_embed_native_lr_dct(y_lr: torch.Tensor, target_hr_size: int | tuple[int, int], norm: str = "ortho") -> tuple[torch.Tensor, torch.Tensor]:
    """DCT native LR coefficients and hard-embed them into the HR DCT grid.

    Returns:
        X_emb: [B,C,H,W] HR DCT coefficients with the LR block copied exactly.
        Y: [B,C,h,w] native LR DCT coefficients.
    """
    if y_lr.ndim != 4:
        raise ValueError(f"Expected y_lr [B,C,h,w], got {tuple(y_lr.shape)}")
    if not torch.is_floating_point(y_lr):
        raise ValueError("y_lr must be a floating-point tensor")
    H, W = _target_hw(target_hr_size)
    b, c, h, w = y_lr.shape
    if h > H or w > W:
        raise ValueError(f"LR size {(h, w)} must fit inside target HR size {(H, W)}")

    Y = dct2(y_lr, norm=norm)
    X_emb = torch.zeros((b, c, H, W), device=y_lr.device, dtype=y_lr.dtype)
    gamma = math.sqrt((float(H) * float(W)) / (float(h) * float(w)))
    X_emb[:, :, :h, :w] = gamma * Y
    return X_emb, Y


def native_lr_dct_proxy(y_lr: torch.Tensor, target_hr_size: int | tuple[int, int], norm: str = "ortho") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the hard-embedded HR proxy image, embedded DCT coefficients, and LR DCT."""
    X_emb, Y = hard_embed_native_lr_dct(y_lr, target_hr_size, norm=norm)
    return idct2(X_emb, norm=norm), X_emb, Y


def _extract_hr_images(batch) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        for key in ("hr", "image"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (list, tuple)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    raise ValueError("HR dataloader must yield a tensor batch or a dict containing 'hr' or 'image'")


@torch.no_grad()
def estimate_hr_spectral_stats(
    hr_dataloader,
    hr_size: int | tuple[int, int],
    num_log_bins: int = 64,
    eps: float = 1e-8,
    save_path=None,
):
    """Compute dataset-level average DCT power spectrum and power-law fit."""
    H, W = _target_hw(hr_size)
    total_sums = None
    total_counts = None
    total_images = 0
    cached_edges = None
    cached_centers = None
    cached_counts = None

    for batch in hr_dataloader:
        x_hr = _extract_hr_images(batch)
        if x_hr.ndim != 4:
            raise ValueError(f"Expected HR images [B,C,H,W], got {tuple(x_hr.shape)}")
        if x_hr.shape[-2:] != (H, W):
            x_hr = F.interpolate(x_hr.float(), size=(H, W), mode="area")
        else:
            x_hr = x_hr.float()
        b = x_hr.shape[0]
        if b == 0:
            continue

        _, valid, edges, centers, flat_bins, counts, k_hr = _dct_log_bin_setup(
            H,
            W,
            int(num_log_bins),
            device=x_hr.device,
            dtype=x_hr.dtype,
            eps=eps,
            bin_edges=cached_edges,
        )
        cached_edges = edges.detach()
        cached_centers = centers.detach()
        cached_counts = counts.detach()

        coeffs = dct2(x_hr, norm="ortho")
        power = coeffs.abs().square().mean(dim=1)
        flat_power = power.reshape(b, -1)[:, valid.reshape(-1)]
        batch_sums = torch.zeros(b, int(num_log_bins), device=x_hr.device, dtype=x_hr.dtype)
        batch_sums.scatter_add_(1, flat_bins[None].expand(b, -1), flat_power)

        if total_sums is None:
            total_sums = torch.zeros(int(num_log_bins), device=x_hr.device, dtype=x_hr.dtype)
            total_counts = torch.zeros(int(num_log_bins), device=x_hr.device, dtype=x_hr.dtype)
        total_sums = total_sums + batch_sums.sum(dim=0)
        total_counts = total_counts + counts * b
        total_images += int(b)

    if total_images == 0 or total_sums is None or total_counts is None:
        raise ValueError("Expected at least one HR image to estimate spectral stats")

    P_hr_bins = torch.where(total_counts > 0, total_sums / total_counts.clamp_min(1.0), torch.zeros_like(total_sums))
    active = (total_counts > 0) & torch.isfinite(P_hr_bins) & (P_hr_bins > 0)
    if int(active.sum().item()) >= 2:
        x = torch.log(cached_centers[active].clamp_min(eps))
        y = torch.log(P_hr_bins[active].clamp_min(eps))
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = x_centered.square().sum().clamp_min(eps)
        slope_hr_t = (x_centered * y_centered).sum() / denom
        intercept = y.mean() - slope_hr_t * x.mean()
        beta_hr_t = torch.exp(intercept).clamp_min(eps)
        slope_hr_t = slope_hr_t.clamp(max=-1e-4)
    else:
        beta_hr_t = P_hr_bins[active].mean().clamp_min(eps) if bool(active.any()) else torch.as_tensor(eps, device=P_hr_bins.device)
        slope_hr_t = torch.as_tensor(-2.0, device=P_hr_bins.device, dtype=P_hr_bins.dtype)

    stats = {
        "bin_edges": cached_edges.detach().cpu(),
        "bin_centers": cached_centers.detach().cpu(),
        "P_hr_bins": P_hr_bins.detach().cpu(),
        "beta_hr": float(beta_hr_t.detach().cpu()),
        "slope_hr": float(slope_hr_t.detach().cpu()),
        "K_hr": float(k_hr.detach().cpu()),
        "H": int(H),
        "W": int(W),
        "num_log_bins": int(num_log_bins),
        "eps": float(eps),
        "bin_counts": cached_counts.detach().cpu(),
        "num_images": int(total_images),
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, path)
    return stats


def load_hr_spectral_stats(path, map_location="cpu"):
    try:
        stats = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        stats = torch.load(path, map_location=map_location)
    required = {"bin_edges", "bin_centers", "P_hr_bins", "beta_hr", "slope_hr", "K_hr", "H", "W", "num_log_bins"}
    missing = required.difference(stats)
    if missing:
        raise ValueError(f"HR spectral stats file is missing keys: {sorted(missing)}")
    return stats


def estimate_effective_bandwidth(
    X_emb: torch.Tensor,
    hr_stats,
    h: int,
    w: int,
    H: int,
    W: int,
    eps: float = 1e-8,
):
    """Estimate per-image effective radial bandwidth from hard-embedded LR DCT energy."""
    if X_emb.ndim != 4:
        raise ValueError(f"Expected X_emb [B,C,H,W], got {tuple(X_emb.shape)}")
    if X_emb.shape[-2:] != (int(H), int(W)):
        raise ValueError(f"X_emb shape {tuple(X_emb.shape[-2:])} does not match H,W={(H, W)}")
    h = int(h)
    w = int(w)
    H = int(H)
    W = int(W)
    if h <= 0 or w <= 0 or h > H or w > W:
        raise ValueError(f"Invalid LR support {(h, w)} for HR grid {(H, W)}")

    dtype = X_emb.dtype if X_emb.is_floating_point() else torch.float32
    num_bins = int(hr_stats.get("num_log_bins", len(hr_stats["P_hr_bins"])))
    k_2d, valid, edges, centers, flat_bins, counts_all, k_hr = _dct_log_bin_setup(
        H,
        W,
        num_bins,
        device=X_emb.device,
        dtype=dtype,
        eps=eps,
        bin_edges=hr_stats["bin_edges"],
    )
    p = torch.arange(H, device=X_emb.device)[:, None]
    q = torch.arange(W, device=X_emb.device)[None]
    observed = valid & (p < h) & (q < w)

    q_bins = torch.zeros((X_emb.shape[0], num_bins), device=X_emb.device, dtype=dtype)
    E_bins = torch.zeros_like(q_bins)
    observed_counts = torch.zeros(num_bins, device=X_emb.device, dtype=dtype)
    if bool(observed.any()):
        observed_bins = torch.bucketize(k_2d[observed].contiguous(), edges.contiguous(), right=False) - 1
        observed_bins = observed_bins.clamp(0, num_bins - 1).long()
        power = X_emb.to(dtype=dtype).abs().square().mean(dim=1)
        flat_power = power.reshape(power.shape[0], -1)[:, observed.reshape(-1)]
        sums = torch.zeros_like(q_bins)
        sums.scatter_add_(1, observed_bins[None].expand(power.shape[0], -1), flat_power)
        observed_counts.scatter_add_(0, observed_bins, torch.ones_like(observed_bins, dtype=dtype))
        E_bins = torch.where(observed_counts[None] > 0, sums / observed_counts.clamp_min(1.0)[None], E_bins)
        P_hr_bins = torch.as_tensor(hr_stats["P_hr_bins"], device=X_emb.device, dtype=dtype)
        q_bins = (E_bins / (P_hr_bins.clamp_min(0.0)[None] + float(eps))).clamp(0.0, 1.0)
        q_bins = torch.where(observed_counts[None] > 0, q_bins, torch.zeros_like(q_bins))

    R_eff = q_bins.mean(dim=1)
    K_hr = torch.as_tensor(float(hr_stats.get("K_hr", float(k_hr.detach().cpu()))), device=X_emb.device, dtype=dtype)
    positive_centers = torch.as_tensor(hr_stats["bin_centers"], device=X_emb.device, dtype=dtype).clamp_min(eps)
    k_min = positive_centers[positive_centers > 0].min().clamp_min(eps)
    K_eff = torch.expm1(R_eff * torch.log1p(K_hr)).clamp(k_min, K_hr)
    debug = {
        "q_bins": q_bins.detach(),
        "R_eff": R_eff.detach(),
        "K_eff": K_eff.detach(),
        "E_bins": E_bins.detach(),
        "observed_bin_counts": observed_counts.detach(),
        "bin_counts": counts_all.detach(),
        "bin_centers": centers.detach(),
        "K_hr": K_hr.expand_as(K_eff).detach(),
    }
    return K_eff, debug


def select_t_star_by_spectral_activation(
    scheduler,
    hr_stats,
    K_eff: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    activation_snr: float = 1.0,
    eps: float = 1e-8,
):
    """Map lambda_star=-log(P_HR(K_eff)) to the closest inference timestep."""
    if timesteps is None:
        timesteps = scheduler.timesteps
    if timesteps is None:
        raise ValueError("Call scheduler.set_timesteps(...) before selecting t_star")
    if activation_snr <= 0:
        raise ValueError("activation_snr must be positive")

    device = K_eff.device
    dtype = K_eff.dtype if K_eff.is_floating_point() else torch.float32
    timesteps = timesteps.to(device=device)
    beta_hr = torch.as_tensor(float(hr_stats["beta_hr"]), device=device, dtype=dtype).clamp_min(eps)
    slope_hr = torch.as_tensor(float(hr_stats["slope_hr"]), device=device, dtype=dtype).clamp(max=-1e-4)
    K_eff = K_eff.to(device=device, dtype=dtype).clamp_min(eps)

    log_p_hr = torch.log(beta_hr) + slope_hr * torch.log(K_eff)
    lambda_star_raw = math.log(float(activation_snr)) - log_p_hr
    lambda_grid = scheduler.logsnr(timesteps).to(device=device, dtype=dtype)
    lambda_min = lambda_grid.min()
    lambda_max = lambda_grid.max()
    lambda_star = lambda_star_raw.clamp(lambda_min, lambda_max)
    clamped = (lambda_star != lambda_star_raw)
    closest = (lambda_grid[None] - lambda_star[:, None]).abs().argmin(dim=1)
    t_star = timesteps[closest]
    alpha_star = scheduler.alpha_bar(t_star).to(device=device, dtype=dtype)
    sigma_star = (1.0 - alpha_star).clamp_min(0.0).sqrt()
    debug = {
        "lambda_star": lambda_star.detach(),
        "lambda_star_raw": lambda_star_raw.detach(),
        "t_star": t_star.detach(),
        "alpha_star": alpha_star.detach(),
        "sigma_star": sigma_star.detach(),
        "clamped_lambda": clamped.detach(),
        "lambda_t_grid": lambda_grid.detach(),
        "timesteps": timesteps.detach(),
    }
    return t_star, debug


@torch.no_grad()
def native_lr_spectral_sdedit_init(
    y_lr: torch.Tensor,
    scheduler,
    hr_stats,
    target_hr_size: int | tuple[int, int],
    generator=None,
    activation_snr: float = 1.0,
    eps: float = 1e-8,
    allow_batch_size_gt_1: bool = False,
):
    """Build native-LR spectral SDEdit initialization with scalar scheduler noise."""
    if scheduler.timesteps is None:
        raise ValueError("Call scheduler.set_timesteps(...) before native_lr_spectral_sdedit_init")
    if y_lr.ndim != 4:
        raise ValueError(f"Expected y_lr [B,C,h,w], got {tuple(y_lr.shape)}")
    if y_lr.shape[0] > 1 and not allow_batch_size_gt_1:
        raise ValueError("native LR spectral SDEdit V1 supports batch size 1 unless allow_batch_size_gt_1=true")

    H, W = _target_hw(target_hr_size)
    h, w = y_lr.shape[-2:]
    x_proxy, X_emb, Y_lr = native_lr_dct_proxy(y_lr, (H, W), norm="ortho")
    K_eff, bandwidth_debug = estimate_effective_bandwidth(X_emb, hr_stats, h=h, w=w, H=H, W=W, eps=eps)
    t_star, activation_debug = select_t_star_by_spectral_activation(
        scheduler=scheduler,
        hr_stats=hr_stats,
        K_eff=K_eff,
        timesteps=scheduler.timesteps,
        activation_snr=activation_snr,
        eps=eps,
    )
    alpha_star = activation_debug["alpha_star"].to(device=y_lr.device, dtype=x_proxy.dtype)
    alpha = alpha_star[:, None, None, None]
    if generator is None:
        noise = torch.randn_like(x_proxy)
    else:
        noise = torch.randn(x_proxy.shape, device=x_proxy.device, dtype=x_proxy.dtype, generator=generator)
    sigma = (1.0 - alpha).clamp_min(0.0).sqrt()
    z_init = alpha.sqrt() * x_proxy + sigma * noise

    clamped = activation_debug["clamped_lambda"]
    debug = {
        "x_proxy": x_proxy.detach(),
        "z_init": z_init.detach(),
        "X_emb": X_emb.detach(),
        "Y_lr": Y_lr.detach(),
        "q_bins": bandwidth_debug["q_bins"].detach(),
        "R_eff": bandwidth_debug["R_eff"].detach(),
        "K_eff": K_eff.detach(),
        "lambda_star": activation_debug["lambda_star"].detach(),
        "lambda_star_raw": activation_debug["lambda_star_raw"].detach(),
        "t_star": t_star.detach(),
        "alpha_star": alpha_star.detach(),
        "sigma_star": activation_debug["sigma_star"].to(device=y_lr.device, dtype=x_proxy.dtype).detach(),
        "clamped_lambda": bool(clamped.reshape(-1).any().detach().cpu().item()) if clamped.numel() == 1 else clamped.detach(),
        "E_bins": bandwidth_debug["E_bins"].detach(),
        "observed_bin_counts": bandwidth_debug["observed_bin_counts"].detach(),
        "bin_centers": bandwidth_debug["bin_centers"].detach(),
        "K_hr": bandwidth_debug["K_hr"].detach(),
    }
    return z_init, t_star, debug


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

import numpy as np
import torch
import torch.nn.functional as F
from prehoc.utils.degradations import mc_psf_degrade, nyquist_lowpass


def interpolate(x, size, mode):
    return F.interpolate(x, size=size, mode=mode) if mode in {"nearest", "area"} else F.interpolate(x, size=size, mode=mode, align_corners=False)


def to_unit(x):
    return ((x.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)


def from_unit(x):
    return (x.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)


def degrade_batch(hr, level, cfg, sr_cfg, sample_ids):
    cfg = cfg or {}
    degradation_type = str(cfg.get("type", "bilinear"))
    h = int(hr.shape[-1])
    x = to_unit(hr)
    downsample_mode = cfg.get("downsample_mode", "area")
    upsample_mode = cfg.get("upsample_mode", "nearest")

    if degradation_type in {"none", "identity"}:
        lr_size = int(cfg.get("lr_size", sr_cfg.get("lr_size", h)))
        lr = interpolate(x, (lr_size, lr_size), downsample_mode)
        return from_unit(lr), from_unit(interpolate(lr, (h, h), upsample_mode))

    if degradation_type == "bilinear":
        lr_size = max(1, int(round(h / float(level))))
        lr = interpolate(x, (lr_size, lr_size), "bilinear")
        return from_unit(lr), from_unit(interpolate(lr, (h, h), "bilinear"))

    arr = x.detach().cpu().numpy().astype(np.float32)
    out = np.zeros_like(arr)
    if degradation_type == "nyquist":
        for channel in range(arr.shape[1]):
            out[:, channel] = nyquist_lowpass(
                arr[:, channel], float(level), batch_size=int(cfg.get("batch_size", 128)), device=cfg.get("device", "cpu")
            )
    elif degradation_type == "mc_psf":
        seed = int(cfg.get("seed", 0))
        for i, sample_id in enumerate(sample_ids):
            for channel in range(arr.shape[1]):
                out[i, channel] = mc_psf_degrade(
                    arr[i : i + 1, channel],
                    resolution_um_per_px=float(level),
                    native_pixel_size_um=cfg.get("native_pixel_size_um", 0.1),
                    continuous_upsampling_factor=cfg.get("continuous_upsampling_factor", 4),
                    sigma0=cfg.get("mc_psf_sigma_hr_px", 1.0),
                    n_samples=cfg.get("mc_n_samples", 8),
                    seed=seed + int(sample_id),
                )[0]
    else:
        raise ValueError(f"Unsupported degradation type: {degradation_type}")

    degraded = torch.from_numpy(out).to(device=hr.device, dtype=hr.dtype)
    lr_size = int(cfg.get("lr_size", sr_cfg.get("lr_size", h)))
    lr = interpolate(degraded, (lr_size, lr_size), downsample_mode)
    return from_unit(lr), from_unit(interpolate(lr, (h, h), upsample_mode))

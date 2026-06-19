from __future__ import annotations

import torch


def make_logs(loss, noise_scheduler, timesteps, image):
    with torch.no_grad():
        lambda_t = noise_scheduler.logsnr(timesteps, image=image)
        tau = noise_scheduler.timesteps_to_tau(timesteps)
        logs = {
            "train/loss": float(loss.detach().mean().cpu()),
            "train/lambda_mean": float(lambda_t.mean().detach().cpu()),
            "train/lambda_min": float(lambda_t.min().detach().cpu()),
            "train/lambda_max": float(lambda_t.max().detach().cpu()),
            "train/tau_mean": float(tau.float().mean().detach().cpu()),
        }
        stats = getattr(noise_scheduler, "last_stats", None)
        if stats:
            if "slope" in stats:
                logs["train/slope_mean"] = float(stats["slope"].mean().detach().cpu())
            if "beta" in stats:
                logs["train/log_beta_mean"] = float(stats["beta"].log().mean().detach().cpu())
        return logs

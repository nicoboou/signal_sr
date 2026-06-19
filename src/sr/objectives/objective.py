from __future__ import annotations

import torch
import torch.nn.functional as F


class Objective:
    def __init__(self, cfg, noise_scheduler):
        self.name = cfg.name
        self.prediction_type = cfg.prediction_type
        self.noise_scheduler = noise_scheduler

    def training_target(self, x0, noise, timesteps, image=None):
        if self.name == "diffusion":
            if self.prediction_type == "epsilon":
                return noise
            if self.prediction_type == "sample":
                return x0
            if self.prediction_type == "v_prediction":
                return self.noise_scheduler.get_velocity(x0, noise, timesteps, image=image)
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        raise ValueError(f"Unsupported objective: {self.name}")

    def prepare_flow_training_input(self, x0, timesteps):
        x1 = torch.randn_like(x0)
        tau = timesteps.float() / float(self.noise_scheduler.num_train_timesteps - 1)
        while tau.ndim < x0.ndim:
            tau = tau[..., None]
        x_t = (1.0 - tau) * x0 + tau * x1
        target = x1 - x0
        return x_t, target

    def loss(self, model_output, target):
        return F.mse_loss(model_output, target)

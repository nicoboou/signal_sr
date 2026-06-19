from __future__ import annotations

import torch
import torch.nn.functional as F

from .eps import eps_pivot, validate_eps_config


DOMAIN_TO_ID = {"LR": 0, "HR": 1, 0: 0, 1: 1}


def _batch_with_domain(batch, domain, device, batch_size):
    out = dict(batch)
    domain_id = DOMAIN_TO_ID[domain]
    out["domain"] = torch.full((batch_size,), domain_id, device=device, dtype=torch.long)
    return out


def _slice_batch(batch, indices, batch_size):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            out[key] = value[indices]
        else:
            out[key] = value
    return out


def _slice_conditioning_image(conditioning_image, indices, batch_size):
    if torch.is_tensor(conditioning_image) and conditioning_image.shape[:1] == (batch_size,):
        return conditioning_image[indices]
    return conditioning_image


def downsample_to_lr(x, lr_size, mode="nearest"):
    if mode == "nearest":
        return F.interpolate(x, size=(lr_size, lr_size), mode="nearest")
    if mode == "area":
        return F.interpolate(x, size=(lr_size, lr_size), mode="area")
    raise ValueError(mode)


def upsample_to_hr(x_lr, image_size):
    return F.interpolate(x_lr, size=(image_size, image_size), mode="nearest")


def measurement_loss(x0_hat, batch, cfg, loss_type="l2_norm"):
    measurement = cfg.get("measurement", {})
    space = measurement.get("space", "lr_up")
    downsample_mode = measurement.get("downsample_mode", "nearest")
    lr_size = int(cfg.lr_size)
    image_size = int(cfg.image_size)
    if space == "lr":
        pred_y = downsample_to_lr(x0_hat, lr_size, mode=downsample_mode)
        target_y = batch["lr"].to(x0_hat.device)
    elif space == "lr_up":
        pred_lr = downsample_to_lr(x0_hat, lr_size, mode=downsample_mode)
        pred_y = upsample_to_hr(pred_lr, image_size)
        target_y = batch["lr_up"].to(x0_hat.device)
    else:
        raise ValueError(space)
    residual = pred_y - target_y
    if loss_type == "l2_norm":
        return torch.linalg.norm(residual.flatten(1), dim=1).mean()
    if loss_type == "mse":
        return F.mse_loss(pred_y, target_y)
    raise ValueError(f"Unsupported DPS measurement loss: {loss_type}")


class Sampler:
    def __init__(self, cfg, objective, noise_scheduler, denoiser, conditioner, global_cfg=None):
        self.cfg = cfg
        self.objective = objective
        self.noise_scheduler = noise_scheduler
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.global_cfg = global_cfg
        self.method = cfg.method
        if self.method == "eps":
            if global_cfg is None:
                raise ValueError("EPS requires global config with image_size/lr_size/sampling.eps")
            validate_eps_config(global_cfg, objective=objective)

    def predict(self, x, timestep, batch, condition_domain, conditioning_image):
        conditioned_batch = _batch_with_domain(batch, condition_domain, x.device, x.shape[0])
        if self.method == "eps":
            if self.global_cfg is None:
                raise ValueError("EPS requires global config with image_size/lr_size/sampling.eps")
            if "lr" not in batch:
                raise ValueError("EPS sampling requires batches with an lr measurement")
            mu_star, y_up = eps_pivot(x, batch["lr"], timestep, self.noise_scheduler, self.global_cfg)
            model_input = self.noise_scheduler.scale_model_input(mu_star, timestep)
            cond = self.conditioner(conditioned_batch, conditioning_image=y_up)
            return self.denoiser(model_input, timestep, cond)
        model_input = self.noise_scheduler.scale_model_input(x, timestep)
        cond = self.conditioner(conditioned_batch, conditioning_image=conditioning_image)
        return self.denoiser(model_input, timestep, cond)

    @torch.no_grad()
    def ddim_loop(self, x, batch, condition_domain, conditioning_image, clip_denoised=None):
        return self.ddim_loop_from(x, batch, condition_domain, conditioning_image, clip_denoised=clip_denoised)

    def _start_indices(self, start_timestep, batch_size, device):
        if self.noise_scheduler.timesteps is None:
            raise ValueError("Call noise_scheduler.set_timesteps(...) before sampling")
        if start_timestep is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        timesteps = self.noise_scheduler.timesteps.to(device=device)
        start_timestep = torch.as_tensor(start_timestep, device=device, dtype=timesteps.dtype)
        if start_timestep.ndim == 0:
            start_timestep = start_timestep.expand(batch_size)
        elif start_timestep.shape != (batch_size,):
            raise ValueError(f"Expected start_timestep scalar or [{batch_size}], got {tuple(start_timestep.shape)}")
        return (timesteps[None] - start_timestep[:, None]).abs().argmin(dim=1)

    @torch.no_grad()
    def ddim_loop_from(self, x, batch, condition_domain, conditioning_image=None, start_timestep=None, clip_denoised=None):
        start_indices = self._start_indices(start_timestep, x.shape[0], x.device)
        batch_size = x.shape[0]
        for step_idx, timestep in enumerate(self.noise_scheduler.timesteps[:-1]):
            active = start_indices <= step_idx
            if not bool(active.any()):
                continue
            if bool(active.all()):
                model_output = self.predict(x, timestep, batch, condition_domain, conditioning_image)
                x = self.noise_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=x,
                    image=conditioning_image,
                    clip_denoised=clip_denoised,
                ).prev_sample
                continue

            indices = active.nonzero(as_tuple=False).flatten()
            x_active = x[indices]
            batch_active = _slice_batch(batch, indices, batch_size)
            conditioning_active = _slice_conditioning_image(conditioning_image, indices, batch_size)
            model_output = self.predict(x_active, timestep, batch_active, condition_domain, conditioning_active)
            x_next = self.noise_scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=x_active,
                image=conditioning_active,
                clip_denoised=clip_denoised,
            ).prev_sample
            x = x.clone()
            x[indices] = x_next
        return x

    def dps_loop(self, x, batch, condition_domain, conditioning_image):
        if self.global_cfg is None:
            raise ValueError("DPS requires global config with image_size/lr_size/measurement")
        dps_cfg = self.cfg.get("dps", {})
        start_step = int(dps_cfg.get("start_step", 0))
        end_step = dps_cfg.get("end_step", None)
        end_step = len(self.noise_scheduler.timesteps) - 1 if end_step is None else int(end_step)
        guidance_scale = float(dps_cfg.get("guidance_scale", 0.1))
        base_sampler = dps_cfg.get("base_sampler", "ddpm")
        clip_denoised = bool(dps_cfg.get("clip_denoised", True))
        loss_type = dps_cfg.get("loss", "l2_norm")

        for step_idx, timestep in enumerate(self.noise_scheduler.timesteps[:-1]):
            if start_step <= step_idx <= end_step and guidance_scale != 0.0:
                x = self.dps_step(
                    x,
                    timestep,
                    batch,
                    condition_domain,
                    conditioning_image,
                    guidance_scale,
                    base_sampler=base_sampler,
                    clip_denoised=clip_denoised,
                    loss_type=loss_type,
                )
            else:
                with torch.no_grad():
                    model_output = self.predict(x, timestep, batch, condition_domain, conditioning_image)
                    x = self._base_step(
                        model_output,
                        timestep,
                        x,
                        conditioning_image,
                        base_sampler=base_sampler,
                        clip_denoised=clip_denoised,
                    ).prev_sample
        return x

    def _base_step(self, model_output, timestep, x, conditioning_image, base_sampler="ddpm", clip_denoised=None):
        if base_sampler == "ddim":
            return self.noise_scheduler.step(
                model_output,
                timestep,
                x,
                image=conditioning_image,
                clip_denoised=clip_denoised,
            )
        if base_sampler == "ddpm":
            return self.noise_scheduler.ddpm_step(
                model_output,
                timestep,
                x,
                image=conditioning_image,
                clip_denoised=clip_denoised,
            )
        raise ValueError(f"Unsupported DPS base_sampler: {base_sampler}")

    def dps_step(
        self,
        x,
        timestep,
        batch,
        condition_domain,
        conditioning_image,
        guidance_scale,
        base_sampler="ddpm",
        clip_denoised=True,
        loss_type="l2_norm",
    ):
        x_in = x.detach().requires_grad_(True)
        model_output = self.predict(x_in, timestep, batch, condition_domain, conditioning_image)
        out = self._base_step(
            model_output,
            timestep,
            x_in,
            conditioning_image,
            base_sampler=base_sampler,
            clip_denoised=clip_denoised,
        )
        x0_hat = out.pred_original_sample
        loss_y = measurement_loss(x0_hat, batch, self.global_cfg, loss_type=loss_type)
        grad = torch.autograd.grad(loss_y, x_in, retain_graph=False, create_graph=False)[0]
        return (out.prev_sample.detach() - guidance_scale * grad.detach()).detach()

    @torch.no_grad()
    def flow_loop(self, x, batch, condition_domain, conditioning_image, direction="reverse", n_steps=50):
        if direction == "forward":
            tau_grid = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=x.device)
        elif direction == "reverse":
            tau_grid = torch.linspace(1.0, 0.0, int(n_steps) + 1, device=x.device)
        else:
            raise ValueError(direction)

        for tau, next_tau in zip(tau_grid[:-1], tau_grid[1:]):
            timestep = torch.full(
                (x.shape[0],),
                float(tau) * (self.noise_scheduler.num_train_timesteps - 1),
                device=x.device,
            )
            velocity = self.predict(x, timestep, batch, condition_domain, conditioning_image)
            dt = next_tau - tau
            x = x + dt * velocity
        return x

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from .spectral import fit_power_law, rapsd, spectral_logsnr


def expand_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while x.ndim < target.ndim:
        x = x[..., None]
    return x


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor | None = None


class BaseScheduler:
    def __init__(self, num_train_timesteps=1000, prediction_type="epsilon", eps=1e-5, **_):
        self.num_train_timesteps = int(num_train_timesteps)
        self.prediction_type = prediction_type
        self.eps = float(eps)
        self.timesteps = None
        self.last_stats = {}

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            int(num_inference_steps),
            device=device,
        )
        return self.timesteps

    def scale_model_input(self, sample, timestep):
        return sample

    def timesteps_to_tau(self, timesteps):
        return timesteps.float() / float(self.num_train_timesteps - 1)

    def logsnr(self, timesteps, image=None):
        raise NotImplementedError

    def alpha_bar(self, timesteps, image=None):
        lambda_t = self.logsnr(timesteps, image=image)
        return torch.sigmoid(lambda_t).clamp(self.eps, 1.0 - self.eps)

    def add_noise(self, original_samples, noise, timesteps, image=None):
        ab = expand_to(self.alpha_bar(timesteps, image=image), original_samples)
        return ab.sqrt() * original_samples + (1.0 - ab).sqrt() * noise

    def get_velocity(self, sample, noise, timesteps, image=None):
        ab = expand_to(self.alpha_bar(timesteps, image=image), sample)
        return ab.sqrt() * noise - (1.0 - ab).sqrt() * sample

    def _next_timestep(self, timestep):
        if self.timesteps is None:
            raise ValueError("Call set_timesteps(...) before step(...)")
        idx = int((self.timesteps - timestep).abs().argmin().item())
        next_idx = min(idx + 1, len(self.timesteps) - 1)
        return self.timesteps[next_idx]

    def _predict_x0_eps(self, model_output, timestep, sample, image=None, clip_denoised=None):
        ab_t = expand_to(self.alpha_bar(timestep, image=image), sample)
        sigma_t = (1.0 - ab_t).sqrt()

        if self.prediction_type == "epsilon":
            eps_hat = model_output
            x0_hat = (sample - sigma_t * eps_hat) / ab_t.sqrt().clamp_min(1e-8)
        elif self.prediction_type == "sample":
            x0_hat = model_output
            eps_hat = (sample - ab_t.sqrt() * x0_hat) / sigma_t.clamp_min(1e-8)
        elif self.prediction_type == "v_prediction":
            v_hat = model_output
            x0_hat = ab_t.sqrt() * sample - sigma_t * v_hat
            eps_hat = sigma_t * sample + ab_t.sqrt() * v_hat
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        if clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
            eps_hat = (sample - ab_t.sqrt() * x0_hat) / sigma_t.clamp_min(1e-8)
        return x0_hat, eps_hat, ab_t

    def step(self, model_output, timestep, sample, image=None, clip_denoised=None):
        next_timestep = self._next_timestep(timestep)
        x0_hat, eps_hat, _ = self._predict_x0_eps(
            model_output,
            timestep,
            sample,
            image=image,
            clip_denoised=clip_denoised,
        )
        ab_next = expand_to(self.alpha_bar(next_timestep, image=image), sample)
        prev_sample = ab_next.sqrt() * x0_hat + (1.0 - ab_next).sqrt() * eps_hat
        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=x0_hat)

    def ddpm_step(self, model_output, timestep, sample, image=None, clip_denoised=None):
        next_timestep = self._next_timestep(timestep)
        x0_hat, _, ab_t = self._predict_x0_eps(
            model_output,
            timestep,
            sample,
            image=image,
            clip_denoised=clip_denoised,
        )
        ab_next = expand_to(self.alpha_bar(next_timestep, image=image), sample)
        alpha_t_to_next = (ab_t / ab_next.clamp_min(1e-8)).clamp(max=1.0)
        beta_t_to_next = (1.0 - alpha_t_to_next).clamp_min(0.0)
        one_minus_ab_t = (1.0 - ab_t).clamp_min(1e-8)

        mean = ab_next.sqrt() * beta_t_to_next / one_minus_ab_t * x0_hat + alpha_t_to_next.sqrt() * (1.0 - ab_next) / one_minus_ab_t * sample
        variance = ((1.0 - ab_next) * beta_t_to_next / one_minus_ab_t).clamp_min(0.0)
        prev_sample = mean
        if bool((next_timestep > 0).detach().cpu().item()):
            prev_sample = prev_sample + variance.sqrt() * torch.randn_like(sample)
        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=x0_hat)


class SpectralRAPSDScheduler(BaseScheduler):
    def __init__(self, kind="mixed", kappa_min=0.2, kappa_max=200.0, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

    def logsnr(self, timesteps, image=None):
        if image is None:
            raise ValueError("SpectralRAPSDScheduler requires image")
        tau = self.timesteps_to_tau(timesteps)
        with torch.no_grad():
            k, psd = rapsd(image)
            slope, beta = fit_power_law(k, psd)
        self.last_stats = {"slope": slope.detach(), "beta": beta.detach(), "n_freq": len(k)}
        return spectral_logsnr(
            tau=tau,
            slope=slope,
            beta=beta,
            n_freq=len(k),
            kind=self.kind,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            eps=self.eps,
        )


class DiffusersSchedulerAdapter:
    def __init__(self, scheduler_cls, **params):
        self.eps = float(params.pop("eps", 1e-5))
        self.scheduler = scheduler_cls(**params)
        if isinstance(self.scheduler, DDPMScheduler):
            self.ddpm_scheduler = self.scheduler
            self.ddim_scheduler = DDIMScheduler.from_config(self.scheduler.config)
        elif isinstance(self.scheduler, DDIMScheduler):
            self.ddim_scheduler = self.scheduler
            self.ddpm_scheduler = DDPMScheduler.from_config(self.scheduler.config)
        else:
            raise ValueError(f"Unsupported diffusers noise scheduler: {scheduler_cls.__name__}")

        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.prediction_type = self.scheduler.config.prediction_type
        self.timesteps = self.scheduler.timesteps
        self.last_stats = {}

    def set_timesteps(self, num_inference_steps, device=None):
        self.ddim_scheduler.set_timesteps(int(num_inference_steps), device=device)
        self.ddpm_scheduler.set_timesteps(int(num_inference_steps), device=device)
        self.timesteps = self.ddim_scheduler.timesteps
        return self.timesteps

    def scale_model_input(self, sample, timestep):
        return self.scheduler.scale_model_input(sample, timestep)

    def timesteps_to_tau(self, timesteps):
        return timesteps.float() / float(self.num_train_timesteps - 1)

    def alpha_bar(self, timesteps, image=None):
        del image
        if torch.is_tensor(timesteps):
            device = timesteps.device
            dtype = timesteps.dtype if timesteps.is_floating_point() else torch.float32
        else:
            device = self.ddpm_scheduler.alphas_cumprod.device
            dtype = torch.float32
        alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(device=device)
        indices = torch.as_tensor(timesteps, device=device).round().long().clamp(0, self.num_train_timesteps - 1)
        return alphas_cumprod[indices].to(dtype=dtype)

    def logsnr(self, timesteps, image=None):
        ab = self.alpha_bar(timesteps, image=image).clamp(self.eps, 1.0 - self.eps)
        return torch.log(ab) - torch.log1p(-ab)

    def add_noise(self, original_samples, noise, timesteps, image=None):
        del image
        return self.scheduler.add_noise(original_samples, noise, timesteps.long())

    def get_velocity(self, sample, noise, timesteps, image=None):
        del image
        return self.scheduler.get_velocity(sample, noise, timesteps.long())

    def _with_clip_sample(self, scheduler, clip_denoised, fn):
        if clip_denoised is None:
            return fn()
        previous = scheduler.config.clip_sample
        scheduler.config.clip_sample = bool(clip_denoised)
        try:
            return fn()
        finally:
            scheduler.config.clip_sample = previous

    def step(self, model_output, timestep, sample, image=None, clip_denoised=None):
        del image
        return self._with_clip_sample(
            self.ddim_scheduler,
            clip_denoised,
            lambda: self.ddim_scheduler.step(model_output, timestep, sample, return_dict=True),
        )

    def ddpm_step(self, model_output, timestep, sample, image=None, clip_denoised=None):
        del image
        return self._with_clip_sample(
            self.ddpm_scheduler,
            clip_denoised,
            lambda: self.ddpm_scheduler.step(model_output, timestep, sample, return_dict=True),
        )


DIFFUSERS_NOISE_SCHEDULER_REGISTRY = {
    "DDPMScheduler": DDPMScheduler,
    "DDIMScheduler": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
}

CUSTOM_NOISE_SCHEDULER_REGISTRY = {
    "SpectralRAPSDScheduler": SpectralRAPSDScheduler,
    "spectral_rapsd": SpectralRAPSDScheduler,
}

RUNTIME_CONFIG_KEYS = {"name", "n_infer_steps"}


def _coerce_scheduler_param(value):
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return value
        return int(parsed) if parsed.is_integer() and value.strip().lower().find("e") == -1 and "." not in value else parsed
    if isinstance(value, list):
        return [_coerce_scheduler_param(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_coerce_scheduler_param(item) for item in value)
    if isinstance(value, dict):
        return {key: _coerce_scheduler_param(item) for key, item in value.items()}
    return value


def build_noise_scheduler(cfg):
    params = {k: _coerce_scheduler_param(v) for k, v in cfg.items() if k not in RUNTIME_CONFIG_KEYS}
    name = cfg.name
    if name in CUSTOM_NOISE_SCHEDULER_REGISTRY:
        return CUSTOM_NOISE_SCHEDULER_REGISTRY[name](**params)
    if name in DIFFUSERS_NOISE_SCHEDULER_REGISTRY:
        return DiffusersSchedulerAdapter(DIFFUSERS_NOISE_SCHEDULER_REGISTRY[name], **params)
    known = sorted([*CUSTOM_NOISE_SCHEDULER_REGISTRY, *DIFFUSERS_NOISE_SCHEDULER_REGISTRY])
    raise ValueError(f"Unknown noise scheduler: {name}. Known schedulers: {known}")

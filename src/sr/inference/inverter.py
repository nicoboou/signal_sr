from __future__ import annotations


class Inverter:
    def __init__(self, inversion_cfg, sampling_cfg, objective, noise_scheduler, denoiser, conditioner, sampler):
        self.inversion_cfg = inversion_cfg
        self.sampling_cfg = sampling_cfg
        self.objective = objective
        self.noise_scheduler = noise_scheduler
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.sampler = sampler
        self.inversion_method = self._resolve_inversion_method(inversion_cfg.method, objective.name)
        self.sampling_method = self._resolve_sampling_method(sampling_cfg.method, objective.name)
        self._validate()

    def _resolve_inversion_method(self, method, objective_name):
        if method == "auto":
            return "ddim" if objective_name == "diffusion" else "flow_ode"
        return method

    def _resolve_sampling_method(self, method, objective_name):
        if method == "auto":
            return "ddim" if objective_name == "diffusion" else "flow_ode"
        return method

    def _validate(self):
        if self.objective.name == "diffusion" and self.inversion_method != "ddim":
            raise ValueError("Diffusion inversion supports only ddim for now")
        if self.objective.name == "diffusion" and self.sampling_method not in {"ddim", "dps", "eps"}:
            raise ValueError("Diffusion sampling supports only ddim, dps, or eps for now")
        if self.objective.name == "flow_matching" and "ddim" in {self.inversion_method, self.sampling_method}:
            raise ValueError("Flow matching cannot use DDIM")
        if self.objective.name == "flow_matching" and self.sampling_method == "dps":
            raise ValueError("DPS is disabled for flow matching")
        if self.objective.name == "flow_matching" and "flow_ode" not in {self.inversion_method, self.sampling_method}:
            raise ValueError("Flow matching requires flow_ode inversion/sampling")

    def invert(self, x_clean, batch, condition_domain="LR", conditioning_image=None):
        if conditioning_image is None:
            conditioning_image = x_clean
        if self.inversion_method == "ddim":
            return self._ddim_invert(x_clean, batch, condition_domain, conditioning_image)
        if self.inversion_method == "flow_ode":
            n_steps = int(self.inversion_cfg.get("flow", {}).get("n_steps", 50))
            return self.sampler.flow_loop(
                x_clean,
                batch,
                condition_domain,
                conditioning_image,
                direction="forward",
                n_steps=n_steps,
            )
        raise ValueError(self.inversion_method)

    def sample(self, x_terminal, batch, condition_domain="HR", conditioning_image=None):
        if conditioning_image is None:
            conditioning_image = x_terminal
        if self.sampling_method in {"ddim", "eps"}:
            return self.sampler.ddim_loop(x_terminal, batch, condition_domain, conditioning_image)
        if self.sampling_method == "dps":
            return self.sampler.dps_loop(x_terminal, batch, condition_domain, conditioning_image)
        if self.sampling_method == "flow_ode":
            n_steps = int(self.sampling_cfg.get("flow", self.inversion_cfg.get("flow", {})).get("n_steps", 50))
            return self.sampler.flow_loop(
                x_terminal,
                batch,
                condition_domain,
                conditioning_image,
                direction="reverse",
                n_steps=n_steps,
            )
        raise ValueError(self.sampling_method)

    def invert_and_sample(self, x_lr, batch, conditioning_image=None):
        if conditioning_image is None:
            conditioning_image = x_lr
        z_t = self.invert(x_lr, batch, condition_domain="LR", conditioning_image=conditioning_image)
        x_hr = self.sample(z_t, batch, condition_domain="HR", conditioning_image=conditioning_image)
        return x_hr, z_t

    def _ddim_invert(self, x_clean, batch, condition_domain, conditioning_image):
        if self.noise_scheduler.timesteps is None:
            raise ValueError("Call noise_scheduler.set_timesteps(...) before inversion")
        sample_timesteps = self.noise_scheduler.timesteps
        self.noise_scheduler.timesteps = sample_timesteps.flip(0)
        try:
            return self.sampler.ddim_loop(x_clean, batch, condition_domain, conditioning_image)
        finally:
            self.noise_scheduler.timesteps = sample_timesteps


def freeze_for_inference(*modules):
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

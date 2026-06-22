from __future__ import annotations

import torch
import torch.nn.functional as F

from sr.inference.spectral_sdedit import spectral_sdedit_init
from sr.schedules.spectral import (
    dct2,
    dct_rapsd_from_coefficients,
    estimate_effective_bandwidth,
    estimate_hr_spectral_stats,
    find_t_star_logfreq_budget,
    fit_power_law,
    hard_embed_native_lr_dct,
    idct2,
    log_frequency_bin_average,
    make_dct_frequency_grid,
    make_frequency_radius_grid,
    make_reliability_mask,
    native_lr_dct_proxy,
    native_lr_spectral_sdedit_init,
    select_t_star_by_spectral_activation,
    spectral_reliability,
)


class ToyScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = int(num_train_timesteps)
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, int(num_inference_steps), device=device)
        return self.timesteps

    def logsnr(self, timesteps, image=None):
        del image
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=self.timesteps.device if self.timesteps is not None else None)
        tau = timesteps / float(self.num_train_timesteps - 1)
        return 5.0 - 10.0 * tau

    def alpha_bar(self, timesteps, image=None):
        return torch.sigmoid(self.logsnr(timesteps, image=image)).clamp(1e-5, 1.0 - 1e-5)


def _manual_sdedit_terms(x_lr, scheduler, scale_r=4, image_size=32, rho=0.75, m_min=0.0, m_max=0.85, num_log_bins=16):
    x_lr_up = F.interpolate(x_lr, size=(image_size, image_size), mode="nearest").float()
    x_freq = dct2(x_lr_up)
    k, psd = dct_rapsd_from_coefficients(x_freq)
    slope, beta = fit_power_law(k, psd)
    k_2d = make_frequency_radius_grid(image_size, image_size, device=x_lr.device, dtype=torch.float32, transform="dct")
    t_star, t_stats = find_t_star_logfreq_budget(
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        slope=slope,
        beta=beta,
        k_2d=k_2d,
        scale_r=scale_r,
        rho=rho,
        num_log_bins=num_log_bins,
    )
    mask, _ = make_reliability_mask(
        scheduler=scheduler,
        t_star=t_star,
        slope=slope,
        beta=beta,
        k_2d=k_2d,
        m_min=m_min,
        m_max=m_max,
    )
    alpha = t_stats["alpha_star"][:, None, None, None]
    return x_freq, mask, alpha


def test_dct_idct_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 10)
    x_rec = idct2(dct2(x))
    assert torch.mean((x - x_rec).abs()) < 1e-5


def test_dct_roundtrip():
    torch.manual_seed(10)
    x = torch.randn(1, 2, 7, 9)
    assert torch.allclose(idct2(dct2(x, norm="ortho"), norm="ortho"), x, atol=1e-5)


def test_dct_frequency_grid_convention():
    k = make_dct_frequency_grid(8, 10, device="cpu", dtype=torch.float32)
    assert torch.isclose(k[0, 1], torch.tensor(0.5))
    assert torch.isclose(k[1, 0], torch.tensor(0.5))
    assert torch.isclose(k[0, 0], torch.tensor(0.5))


def test_spectral_reliability_bounds():
    snr = torch.tensor([0.0, 0.5, 1.0, 10.0, float("inf")])
    w = spectral_reliability(snr)
    assert torch.all(w >= 0)
    assert torch.all(w <= 1)
    assert torch.isclose(w[0], torch.tensor(0.0))
    assert torch.isclose(w[-1], torch.tensor(1.0))


def test_logfreq_f_is_monotonic_with_logsnr():
    scheduler = ToyScheduler()
    timesteps = scheduler.set_timesteps(20)
    k_2d = make_frequency_radius_grid(32, 32, device=timesteps.device, dtype=torch.float32, transform="dct")
    slope = torch.tensor([-2.0])
    beta = torch.tensor([1.0])
    _, stats = find_t_star_logfreq_budget(
        scheduler=scheduler,
        timesteps=timesteps,
        slope=slope,
        beta=beta,
        k_2d=k_2d,
        scale_r=4,
        rho=0.75,
        num_log_bins=16,
    )
    per_t_F = stats["per_t_F"][0]
    lambda_grid = scheduler.logsnr(timesteps)
    order = torch.argsort(lambda_grid)
    assert torch.all(torch.diff(per_t_F[order]) >= -1e-6)


def test_t_star_is_valid_scheduler_timestep_and_mask_is_bounded():
    scheduler = ToyScheduler()
    timesteps = scheduler.set_timesteps(12)
    k_2d = make_frequency_radius_grid(24, 24, device=timesteps.device, dtype=torch.float32, transform="dct")
    slope = torch.tensor([-1.5])
    beta = torch.tensor([0.7])
    t_star, _ = find_t_star_logfreq_budget(
        scheduler=scheduler,
        timesteps=timesteps,
        slope=slope,
        beta=beta,
        k_2d=k_2d,
        scale_r=3,
        rho=0.6,
        num_log_bins=12,
    )
    assert bool(torch.isin(t_star, timesteps).all())

    mask, _ = make_reliability_mask(
        scheduler=scheduler,
        t_star=t_star,
        slope=slope,
        beta=beta,
        k_2d=k_2d,
        m_min=0.1,
        m_max=0.8,
    )
    assert mask.shape == (1, 1, 24, 24)
    assert torch.all(mask >= 0.1)
    assert torch.all(mask <= 0.8)


def test_log_frequency_bin_average_excludes_dc_and_hr_corners():
    k_2d = make_frequency_radius_grid(16, 16, device="cpu", dtype=torch.float32, transform="dct")
    values = torch.ones(1, 1, 16, 16)
    values[..., 0, 0] = 1000.0
    values[..., -1, -1] = 1000.0
    F, _ = log_frequency_bin_average(values, k_2d, num_bins=8)
    assert torch.isclose(F[0], torch.tensor(1.0))


def test_spectral_sdedit_init_outputs_finite_values():
    torch.manual_seed(1)
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    x_lr = torch.randn(1, 1, 8, 8)
    z_init, t_star, stats = spectral_sdedit_init(
        x_lr=x_lr,
        hr_scheduler=scheduler,
        scale_r=4,
        image_size=32,
        rho=0.75,
        m_min=0.0,
        m_max=0.85,
        num_log_bins=16,
    )
    assert z_init.shape == (1, 1, 32, 32)
    assert bool(torch.isfinite(z_init).all())
    assert bool(torch.isin(t_star, scheduler.timesteps).all())
    assert torch.all(stats["z_init_finite"] == 1)


def test_scheduler_compatible_init_uses_scalar_noise_scale():
    seed = 7
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    x_lr = torch.randn(1, 1, 8, 8)

    torch.manual_seed(seed)
    z_init, _, stats = spectral_sdedit_init(
        x_lr=x_lr,
        hr_scheduler=scheduler,
        scale_r=4,
        image_size=32,
        rho=0.75,
        m_min=0.0,
        m_max=0.85,
        num_log_bins=16,
        init_formula="scheduler_compatible",
    )
    x_freq, mask, alpha = _manual_sdedit_terms(x_lr, scheduler)
    torch.manual_seed(seed)
    noise_freq = torch.randn_like(x_freq)
    sigma = (1.0 - alpha).clamp_min(0.0).sqrt()
    expected = idct2(alpha.sqrt() * mask * x_freq + sigma * noise_freq)
    legacy_expected = idct2(alpha.sqrt() * mask * x_freq + (1.0 - alpha * mask.square()).clamp_min(0.0).sqrt() * noise_freq)

    assert stats["init_formula"] == "scheduler_compatible"
    assert torch.allclose(stats["sigma_star"], sigma.reshape(1), atol=1e-6)
    assert sigma.shape == (1, 1, 1, 1)
    assert torch.allclose(z_init, expected, atol=1e-5)
    assert not torch.allclose(z_init, legacy_expected, atol=1e-5)


def test_legacy_frequency_dependent_init_is_opt_in():
    seed = 11
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    x_lr = torch.randn(1, 1, 8, 8)

    torch.manual_seed(seed)
    z_init, _, stats = spectral_sdedit_init(
        x_lr=x_lr,
        hr_scheduler=scheduler,
        scale_r=4,
        image_size=32,
        rho=0.75,
        m_min=0.0,
        m_max=0.85,
        num_log_bins=16,
        init_formula="frequency_dependent_legacy",
    )
    x_freq, mask, alpha = _manual_sdedit_terms(x_lr, scheduler)
    torch.manual_seed(seed)
    noise_freq = torch.randn_like(x_freq)
    expected = idct2(alpha.sqrt() * mask * x_freq + (1.0 - alpha * mask.square()).clamp_min(0.0).sqrt() * noise_freq)

    assert stats["init_formula"] == "frequency_dependent_legacy"
    assert "legacy_noise_scale_low_mean" in stats
    assert "legacy_noise_scale_high_mean" in stats
    assert torch.allclose(z_init, expected, atol=1e-5)
    assert bool(torch.isfinite(z_init).all())


def _native_stats(H=32, W=32, num_log_bins=16):
    torch.manual_seed(123)
    hr = torch.randn(4, 1, H, W)
    return estimate_hr_spectral_stats([{"hr": hr}], (H, W), num_log_bins=num_log_bins)


def test_constant_image_embedding():
    y_lr = torch.full((1, 1, 8, 8), 0.37)
    x_proxy, X_emb, Y = native_lr_dct_proxy(y_lr, (32, 32))
    assert X_emb.shape == (1, 1, 32, 32)
    assert Y.shape == (1, 1, 8, 8)
    assert torch.allclose(x_proxy, torch.full_like(x_proxy, 0.37), atol=1e-5)


def test_hard_embedding_shape():
    y_lr = torch.randn(1, 2, 8, 10)
    X_emb, Y = hard_embed_native_lr_dct(y_lr, (24, 28))
    assert X_emb.shape == (1, 2, 24, 28)
    assert Y.shape == (1, 2, 8, 10)
    assert torch.count_nonzero(X_emb[:, :, 8:, :]) == 0
    assert torch.count_nonzero(X_emb[:, :, :, 10:]) == 0


def test_q_bins_bounds():
    hr_stats = _native_stats()
    y_lr = torch.randn(1, 1, 8, 8)
    X_emb, _ = hard_embed_native_lr_dct(y_lr, (32, 32))
    _, debug = estimate_effective_bandwidth(X_emb, hr_stats, h=8, w=8, H=32, W=32)
    q_bins = debug["q_bins"]
    assert torch.all(q_bins >= 0)
    assert torch.all(q_bins <= 1)


def test_K_eff_bounds():
    hr_stats = _native_stats()
    y_lr = torch.randn(1, 1, 8, 8)
    X_emb, _ = hard_embed_native_lr_dct(y_lr, (32, 32))
    K_eff, _ = estimate_effective_bandwidth(X_emb, hr_stats, h=8, w=8, H=32, W=32)
    assert torch.all(K_eff > 0)
    assert torch.all(K_eff <= float(hr_stats["K_hr"]))


def test_lambda_star_finite():
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    hr_stats = _native_stats()
    K_eff = torch.tensor([4.0])
    _, debug = select_t_star_by_spectral_activation(scheduler, hr_stats, K_eff)
    assert torch.isfinite(debug["lambda_star"]).all()


def test_t_star_valid():
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    hr_stats = _native_stats()
    K_eff = torch.tensor([4.0])
    t_star, _ = select_t_star_by_spectral_activation(scheduler, hr_stats, K_eff)
    assert bool(torch.isin(t_star, scheduler.timesteps).all())


def test_z_init_scheduler_variance_shape():
    torch.manual_seed(321)
    scheduler = ToyScheduler()
    scheduler.set_timesteps(10)
    hr_stats = _native_stats()
    y_lr = torch.randn(1, 1, 8, 8)
    z_init, t_star, debug = native_lr_spectral_sdedit_init(y_lr, scheduler, hr_stats, (32, 32))
    assert z_init.shape == (1, 1, 32, 32)
    assert debug["z_init"].shape == (1, 1, 32, 32)
    assert debug["x_proxy"].shape == (1, 1, 32, 32)
    assert debug["alpha_star"].shape == (1,)
    assert debug["sigma_star"].shape == (1,)
    assert bool(torch.isin(t_star, scheduler.timesteps).all())

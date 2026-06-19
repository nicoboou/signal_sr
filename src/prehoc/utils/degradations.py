"""Clear image degradation functions."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


def _as_numpy_images(images):
    x = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    return x[:, 0] if x.ndim == 4 and x.shape[1] == 1 else x


def nyquist_lowpass(images, rho, batch_size=128, device="cuda"):
    """Apply a disk low-pass filter at cutoff rho in cycles/pixel."""
    images = _as_numpy_images(images).astype(np.float32)
    h, w = images.shape[-2:]
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device))
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    mask = (radius <= float(rho)).float()
    outputs = []
    for start in range(0, len(images), int(batch_size)):
        x = torch.from_numpy(images[start : start + int(batch_size)]).float().to(device)
        spectrum = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)).real
        outputs.append(filtered.clamp(0, 1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def bilinear_degrade(images, factor, batch_size=128, device="cuda"):
    """Downsample by factor and bilinear-upsample back to the original size."""
    images = _as_numpy_images(images).astype(np.float32)
    h, w = images.shape[-2:]
    low_h, low_w = max(1, int(round(h / float(factor)))), max(1, int(round(w / float(factor))))
    outputs = []
    for start in range(0, len(images), int(batch_size)):
        x = torch.from_numpy(images[start : start + int(batch_size), None]).float().to(device)
        low = F.interpolate(x, size=(low_h, low_w), mode="bilinear", align_corners=False)
        high = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        outputs.append(high[:, 0].clamp(0, 1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def make_continuous_spline(image, upsample=4):
    """Upsample a discrete image with a smooth spline representation."""
    h, w = image.shape
    spline = RectBivariateSpline(np.arange(h), np.arange(w), image)
    y_new = np.linspace(0, h - 1, h * int(upsample))
    x_new = np.linspace(0, w - 1, w * int(upsample))
    return np.clip(spline(y_new, x_new), 0, 1).astype(np.float32)


def render_monte_carlo(image_continuous, resolution_um_per_px, sigma0, native_continuous_um_per_px, n_samples, rng):
    """Exact Monte Carlo PSF rendering from the previous pre-hoc notebook."""
    sigma = float(sigma0) * (float(resolution_um_per_px) / float(native_continuous_um_per_px))
    blurred = gaussian_filter(image_continuous, sigma=sigma, mode="reflect")
    h_cont, w_cont = image_continuous.shape
    h_out = max(1, int(round(h_cont * (native_continuous_um_per_px / float(resolution_um_per_px)))))
    w_out = max(1, int(round(w_cont * (native_continuous_um_per_px / float(resolution_um_per_px)))))
    dy = rng.random((h_out, w_out, int(n_samples)), dtype=np.float32)
    dx = rng.random((h_out, w_out, int(n_samples)), dtype=np.float32)
    ys = np.minimum(((np.arange(h_out)[:, None, None] + dy) * (h_cont / h_out)).astype(np.int64), h_cont - 1)
    xs = np.minimum(((np.arange(w_out)[None, :, None] + dx) * (w_cont / w_out)).astype(np.int64), w_cont - 1)
    return blurred[ys, xs].mean(axis=2).astype(np.float32)


def mc_psf_degrade(
    images,
    resolution_um_per_px,
    native_pixel_size_um=0.1,
    continuous_upsampling_factor=4,
    sigma0=1.0,
    n_samples=8,
    seed=0,
):
    """Render every image with exact Monte Carlo PSF sampling and resize to HR shape."""
    images = _as_numpy_images(images).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    native_continuous_um_per_px = float(native_pixel_size_um) / int(continuous_upsampling_factor)
    h, w = images.shape[-2:]
    outputs = []
    for image in images:
        continuous = make_continuous_spline(image, upsample=continuous_upsampling_factor)
        rendered = render_monte_carlo(
            continuous,
            resolution_um_per_px=resolution_um_per_px,
            sigma0=sigma0,
            native_continuous_um_per_px=native_continuous_um_per_px,
            n_samples=n_samples,
            rng=rng,
        )
        restored = resize(rendered, (h, w), order=1, mode="reflect", anti_aliasing=False, preserve_range=True)
        outputs.append(np.clip(restored, 0, 1).astype(np.float32))
    return np.stack(outputs)


def psf_summary(resolution_values_um, native_pixel_size_um=0.1, continuous_upsampling_factor=4, sigma0=1.0):
    native_continuous_um_per_px = float(native_pixel_size_um) / int(continuous_upsampling_factor)
    rows = []
    for resolution in resolution_values_um:
        sigma = float(sigma0) * (float(resolution) / native_continuous_um_per_px)
        rows.append({"resolution_um_per_px": float(resolution), "sigma_continuous_px": sigma, "fwhm_continuous_px": 2.355 * sigma})
    return rows

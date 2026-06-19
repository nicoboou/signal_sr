"""RGB preprocessing and degradation transforms for pre-hoc classifiers."""

import numpy as np
import torch
from PIL import Image

from .degradations import bilinear_degrade, mc_psf_degrade, nyquist_lowpass


RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def pil_to_rgb_tensor(image):
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class NLMDegradationTransform:
    """Resize, tensorize, and degrade one RGB crop.

    ``needs_index`` lets ``NLMDataset`` pass the sample index so stochastic
    degradations are fixed per sample across epochs.
    """

    needs_index = True

    def __init__(
        self,
        image_size=128,
        degradation_type="none",
        level=1,
        seed=0,
        batch_size=128,
        native_pixel_size_um=0.1,
        continuous_upsampling_factor=4,
        mc_psf_sigma_hr_px=1.0,
        mc_n_samples=8,
    ):
        self.image_size = int(image_size)
        self.degradation_type = str(degradation_type)
        self.level = float(level)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.native_pixel_size_um = float(native_pixel_size_um)
        self.continuous_upsampling_factor = int(continuous_upsampling_factor)
        self.mc_psf_sigma_hr_px = float(mc_psf_sigma_hr_px)
        self.mc_n_samples = int(mc_n_samples)

    def __call__(self, image, index=None):
        image = image.convert("RGB").resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
        tensor = pil_to_rgb_tensor(image)
        return self.degrade(tensor, index=index)

    def degrade(self, tensor, index=None):
        if self.degradation_type == "none":
            return tensor

        array = tensor.numpy()
        if self.degradation_type == "bilinear":
            out = bilinear_degrade(array, self.level, batch_size=self.batch_size, device="cpu")
        elif self.degradation_type == "nyquist":
            out = nyquist_lowpass(array, self.level, batch_size=self.batch_size, device="cpu")
        elif self.degradation_type == "mc_psf":
            out = mc_psf_degrade(
                array,
                resolution_um_per_px=self.level,
                native_pixel_size_um=self.native_pixel_size_um,
                continuous_upsampling_factor=self.continuous_upsampling_factor,
                sigma0=self.mc_psf_sigma_hr_px,
                n_samples=self.mc_n_samples,
                seed=self.seed + (0 if index is None else int(index)),
            )
        else:
            raise ValueError(f"Unsupported degradation type: {self.degradation_type}")

        return torch.from_numpy(np.clip(out, 0.0, 1.0).astype(np.float32))


def make_transform(config, level):
    data_cfg = config.get("data", {})
    degradation_cfg = config.get("degradation", {})
    return NLMDegradationTransform(
        image_size=data_cfg.get("image_size", 128),
        degradation_type=degradation_cfg.get("type", "none"),
        level=level,
        seed=config.get("seed", 0),
        batch_size=degradation_cfg.get("batch_size", 128),
        native_pixel_size_um=degradation_cfg.get("native_pixel_size_um", 0.1),
        continuous_upsampling_factor=degradation_cfg.get("continuous_upsampling_factor", 4),
        mc_psf_sigma_hr_px=degradation_cfg.get("mc_psf_sigma_hr_px", 1.0),
        mc_n_samples=degradation_cfg.get("mc_n_samples", 8),
    )

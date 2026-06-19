from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from ..utils.import_utils import build_from_config


class AutoencoderWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from diffusers import AutoencoderKL

        kwargs = {}
        if cfg.get("subfolder", None) is not None:
            kwargs["subfolder"] = cfg.subfolder
        self.vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, **kwargs)
        self.scaling_factor = float(cfg.scaling_factor)
        self.trainable = bool(cfg.get("trainable", False))
        self.sample_posterior = bool(cfg.get("sample_posterior", False))
        self.vae.requires_grad_(self.trainable)

    def encode_to_latent(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() if self.training and self.sample_posterior else posterior.mode()
        return z * self.scaling_factor

    def decode_to_image(self, z):
        return self.vae.decode(z / self.scaling_factor).sample


class IdentityAutoencoder(nn.Module):
    """Debug autoencoder implementing the latent interface without diffusers."""

    def __init__(self, scaling_factor=1.0, downsample_factor=1, trainable=False, **_):
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.downsample_factor = int(downsample_factor)
        self.trainable = bool(trainable)

    def encode_to_latent(self, x):
        z = x
        if self.downsample_factor > 1:
            z = F.avg_pool2d(z, kernel_size=self.downsample_factor, stride=self.downsample_factor)
        return z * self.scaling_factor

    def decode_to_image(self, z):
        x = z / self.scaling_factor
        if self.downsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.downsample_factor, mode="nearest")
        return x


def build_autoencoder(cfg):
    if cfg.target == "diffusers.AutoencoderKL":
        return AutoencoderWrapper(cfg)
    autoencoder = build_from_config(cfg)
    if not hasattr(autoencoder, "encode_to_latent") or not hasattr(autoencoder, "decode_to_image"):
        raise ValueError("Autoencoder must implement encode_to_latent(...) and decode_to_image(...)")
    return autoencoder

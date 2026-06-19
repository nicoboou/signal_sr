from __future__ import annotations

import inspect
import torch
from torch import nn


def _config_value(source, key, default=None):
    if source is None:
        return default
    if hasattr(source, "get"):
        return source.get(key, default)
    return getattr(source, key, default)


def _denoiser_params(source):
    if hasattr(source, "config"):
        return source.config
    return _config_value(source, "params", source)


def denoiser_channels(source):
    params = _denoiser_params(source)
    in_channels = _config_value(params, "in_channels", None)
    out_channels = _config_value(params, "out_channels", in_channels)
    if in_channels is None or out_channels is None:
        raise ValueError("UNet2DModel config must define in_channels and out_channels")
    return int(in_channels), int(out_channels)


def denoiser_conditioning_mode(source):
    params = _denoiser_params(source)
    num_class_embeds = _config_value(params, "num_class_embeds", None)
    class_embed_type = _config_value(params, "class_embed_type", None)
    if num_class_embeds is not None:
        in_channels, out_channels = denoiser_channels(source)
        if in_channels != out_channels:
            raise ValueError("UNet2DModel config cannot combine num_class_embeds with channel concatenation")
        return "class_labels"
    if class_embed_type is not None:
        raise ValueError("Class conditioning must be selected with denoiser.params.num_class_embeds")

    in_channels, out_channels = denoiser_channels(source)
    if in_channels > out_channels:
        return "concat"
    if in_channels == out_channels:
        return "unconditional"
    raise ValueError(f"UNet2DModel in_channels={in_channels} cannot be smaller than out_channels={out_channels}")


class DenoiserAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.conditioning_mode = denoiser_conditioning_mode(model)
        in_channels, out_channels = denoiser_channels(model)
        self.concat_channels = in_channels - out_channels

    def _sample(self, output):
        return output.sample if hasattr(output, "sample") else output

    def forward(self, x_t, t_model, cond):
        if self.conditioning_mode == "unconditional":
            return self._sample(
                self.model(
                    x_t,
                    t_model,
                    return_dict=True,
                )
            )
        if self.conditioning_mode == "class_labels":
            if cond is None or cond.class_labels is None:
                raise ValueError("UNet2DModel class conditioning requires batch domain labels")
            return self._sample(
                self.model(
                    x_t,
                    t_model,
                    class_labels=cond.class_labels.to(device=x_t.device),
                    return_dict=True,
                )
            )
        if self.conditioning_mode == "concat":
            if cond is None or cond.conditioning_image is None:
                raise ValueError("UNet2DModel channel concatenation requires a conditioning image")
            conditioning_image = cond.conditioning_image.to(device=x_t.device, dtype=x_t.dtype)
            if conditioning_image.shape[0] != x_t.shape[0] or conditioning_image.shape[-2:] != x_t.shape[-2:]:
                raise ValueError(
                    f"Conditioning image shape {tuple(conditioning_image.shape)} is incompatible with model input {tuple(x_t.shape)}"
                )
            if conditioning_image.shape[1] != self.concat_channels:
                raise ValueError(f"Expected {self.concat_channels} conditioning channels, got {conditioning_image.shape[1]}")
            return self._sample(
                self.model(
                    torch.cat([x_t, conditioning_image], dim=1),
                    t_model,
                    return_dict=True,
                )
            )
        raise ValueError(f"Unsupported conditioning mode: {self.conditioning_mode}")


def validate_denoiser(model, data_channels=None):
    if model.__class__.__name__ != "UNet2DModel":
        raise ValueError("denoiser.target must build diffusers.UNet2DModel")

    forward_params = inspect.signature(model.forward).parameters
    if "class_labels" not in forward_params:
        raise ValueError("UNet2DModel forward must accept class_labels")

    mode = denoiser_conditioning_mode(model)
    if data_channels is not None:
        in_channels, out_channels = denoiser_channels(model)
        if out_channels != int(data_channels):
            raise ValueError(f"Model out_channels={out_channels} does not match data channels={data_channels}")
        if mode != "concat" and in_channels != int(data_channels):
            raise ValueError(f"Model in_channels={in_channels} does not match data channels={data_channels}")

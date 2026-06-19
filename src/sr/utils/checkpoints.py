from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .config import to_plain


def _load_weights(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    out_dir,
    denoiser,
    conditioner,
    optimizer,
    cfg,
    step: int,
    accelerator=None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    denoiser_to_save = accelerator.unwrap_model(denoiser) if accelerator is not None else denoiser
    conditioner_to_save = accelerator.unwrap_model(conditioner) if accelerator is not None else conditioner

    torch.save(denoiser_to_save.state_dict(), out_dir / "denoiser.pt")
    torch.save(conditioner_to_save.state_dict(), out_dir / "conditioner.pt")
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": int(step)},
        out_dir / "training_state.pt",
    )
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain(cfg), handle, sort_keys=False)

    model = getattr(denoiser_to_save, "model", None)
    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir / "denoiser")

    return out_dir


def load_model_weights(checkpoint_dir, denoiser=None, conditioner=None, map_location="cpu") -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if denoiser is not None:
        denoiser_path = checkpoint_dir / "denoiser.pt"
        if denoiser_path.exists():
            denoiser.load_state_dict(_load_weights(denoiser_path, map_location=map_location))
    if conditioner is not None:
        conditioner_path = checkpoint_dir / "conditioner.pt"
        if conditioner_path.exists():
            conditioner.load_state_dict(_load_weights(conditioner_path, map_location=map_location))

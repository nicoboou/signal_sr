from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F

from .data import build_dataloader
from .diagnostics import image_metrics, latent_metrics, scalarize_metrics
from .inference import Inverter, Sampler, spectral_sdedit_sr
from .inference.inverter import freeze_for_inference
from .models import build_autoencoder, build_conditioner, build_denoiser, denoiser_channels, denoiser_conditioning_mode
from .objectives import Objective
from .schedules import build_noise_scheduler
from .splits import ensure_synthetic_split
from .utils.checkpoints import load_model_weights
from .utils.config import load_config, merge_dict, to_plain
from .utils.images import save_images, tensor_to_pil_image
from .utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["invert", "sample", "invert_sample", "spectral_sdedit_sr"], default="invert_sample")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_metrics(metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _is_main_process():
    for rank_name in ("RANK", "LOCAL_RANK"):
        rank = os.environ.get(rank_name)
        if rank is not None and int(rank) != 0:
            return False
    return True


def _init_wandb(cfg, args, out_dir):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False) or not _is_main_process():
        return None
    try:
        import wandb
    except ImportError:
        print("wandb.enabled=true but wandb is not installed; skipping W&B logging")
        return None

    init_kwargs = {
        "project": wandb_cfg.get("project", "spectral_inversion"),
        "job_type": "inference",
        "config": {
            "config": to_plain(cfg),
            "inference": {
                "mode": args.mode,
                "split": args.split,
                "checkpoint": args.checkpoint,
                "num_samples": args.num_samples,
                "output_dir": str(out_dir),
            },
        },
    }
    for key in ("entity", "group", "mode", "name", "tags"):
        if key in wandb_cfg:
            init_kwargs[key] = wandb_cfg[key]
    return wandb.init(**init_kwargs)


def _wandb_images(images_dict, prefix):
    try:
        import wandb

        if wandb.run is None:
            return {}
        out = {}
        for key, tensor in images_dict.items():
            imgs = tensor.detach().cpu()
            for i in range(imgs.shape[0]):
                out[f"{prefix}/{key}/sample_{i}"] = wandb.Image(tensor_to_pil_image(imgs[i]))
        return out
    except ImportError:
        return {}


def _log_wandb_inference(run, mode, metrics, images_dict):
    if run is None:
        return
    prefix = f"infer/{mode}"
    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    payload.update(_wandb_images(images_dict, prefix))
    if payload:
        run.log(payload, step=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode == "invert_sample" and cfg.sampling.method in {"dps", "eps"}:
        raise ValueError(f"{cfg.sampling.method} is a direct sampling method; use --mode sample instead of --mode invert_sample")
    if cfg.space == "latent" and cfg.sampling.method == "eps":
        raise ValueError("EPS currently supports only pixel-space inverse problems")
    if args.mode == "spectral_sdedit_sr":
        if cfg.space != "pixel":
            raise ValueError("spectral_sdedit_sr currently supports only space=pixel")
        if cfg.objective.name != "diffusion":
            raise ValueError("spectral_sdedit_sr requires objective.name=diffusion")
        if cfg.noise_scheduler.name == "spectral_rapsd":
            raise ValueError("spectral_sdedit_sr must use the trained HR diffusion scheduler, not spectral_rapsd")
        sdedit_cfg = cfg.get("spectral_sdedit", cfg.sampling.get("spectral_sdedit", {}))
        if args.num_samples > 1 and not sdedit_cfg.get("allow_batch_size_gt_1", False):
            raise ValueError("spectral_sdedit_sr supports --num-samples 1 unless spectral_sdedit.allow_batch_size_gt_1=true")
    seed_everything(int(cfg.get("seed", 0)))
    if cfg.data.name == "synthetic_microscopy":
        ensure_synthetic_split(cfg)

    device = torch.device(args.device)
    data_cfg = merge_dict(cfg.data, cfg.get(f"{args.split}_data", {}))
    data_cfg.return_pair = True
    loader = build_dataloader(data_cfg, split=args.split, shuffle=False, batch_size=args.num_samples)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    autoencoder = None
    concat_conditioning = denoiser_conditioning_mode(cfg.denoiser) == "concat"
    if cfg.space == "latent":
        if cfg.sampling.method == "dps" and not cfg.sampling.get("dps", {}).get("enabled_in_latent", False):
            raise ValueError("Latent DPS is disabled unless sampling.dps.enabled_in_latent=true")
        autoencoder = build_autoencoder(cfg.autoencoder).to(device)
        autoencoder.eval()
        autoencoder.requires_grad_(False)
        _, data_channels = denoiser_channels(cfg.denoiser)
    else:
        data_channels = int(cfg.data.channels)

    denoiser = build_denoiser(cfg.denoiser, data_channels=data_channels).to(device)
    conditioner = build_conditioner().to(device)
    noise_scheduler = build_noise_scheduler(cfg.noise_scheduler)
    objective = Objective(cfg.objective, noise_scheduler)
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)
    inverter = None
    if args.mode != "spectral_sdedit_sr":
        inverter = Inverter(cfg.inversion, cfg.sampling, objective, noise_scheduler, denoiser, conditioner, sampler)

    if args.checkpoint:
        load_model_weights(args.checkpoint, denoiser=denoiser, conditioner=conditioner, map_location=device)
    freeze_for_inference(denoiser, conditioner)
    if autoencoder is not None:
        freeze_for_inference(autoencoder)

    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    out_dir = Path(args.output_dir or f"outputs/{Path(args.config).stem}/{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(cfg, args, out_dir)

    grad_context = torch.enable_grad() if cfg.sampling.method == "dps" else torch.no_grad()
    with grad_context:
        if args.mode == "invert_sample":
            x_lr_img = batch["lr_up"] if "lr_up" in batch else batch["image"]
            if autoencoder is None:
                x_start = x_lr_img
                conditioning_image = x_lr_img
            else:
                x_start = autoencoder.encode_to_latent(x_lr_img)
                rapsd_source = cfg.noise_scheduler.get("rapsd_source", "latent_lr")
                conditioning_image = x_start if concat_conditioning or rapsd_source == "latent_lr" else x_lr_img
            x_state, z_t = inverter.invert_and_sample(x_start, batch, conditioning_image=conditioning_image)
            x_hr = autoencoder.decode_to_image(x_state) if autoencoder is not None else x_state
            save_images(x_lr_img, out_dir, "lr_up")
            if "hr" in batch:
                save_images(batch["hr"], out_dir, "hr_gt")
            save_images(x_hr, out_dir, "lr_to_hr")
            torch.save(z_t.detach().cpu(), out_dir / "terminal_state.pt")
            metrics = image_metrics(
                x_hr,
                batch,
                lr_size=int(cfg.lr_size),
                downsample_mode=cfg.get("measurement", {}).get("downsample_mode", "nearest"),
            )
            metrics.update(latent_metrics(z_t))
            metrics = scalarize_metrics(metrics)
            save_metrics(metrics, out_dir)
            wandb_images = {"lr_up": x_lr_img, "output": x_hr}
            if "hr" in batch:
                wandb_images["hr_gt"] = batch["hr"]
            _log_wandb_inference(wandb_run, args.mode, metrics, wandb_images)
        elif args.mode == "spectral_sdedit_sr":
            x_lr = batch["lr"] if "lr" in batch else batch.get("lr_up", batch["image"])
            sdedit_cfg = cfg.get("spectral_sdedit", cfg.sampling.get("spectral_sdedit", {}))
            x_lr_up = F.interpolate(x_lr, size=(int(cfg.image_size), int(cfg.image_size)), mode=sdedit_cfg.get("upsample_mode", "nearest"))
            x_hr, z_init, sdedit_stats = spectral_sdedit_sr(
                x_lr=x_lr,
                batch=batch,
                model=denoiser,
                sampler=sampler,
                hr_scheduler=noise_scheduler,
                scale_r=cfg.get("scale", int(cfg.image_size) // int(cfg.lr_size)),
                cfg=cfg,
            )
            save_images(x_lr_up, out_dir, "lr_up")
            if "hr" in batch:
                save_images(batch["hr"], out_dir, "hr_gt")
            save_images(x_hr, out_dir, "spectral_sdedit_sr")
            torch.save(z_init.detach().cpu(), out_dir / "terminal_state.pt")
            metrics = image_metrics(
                x_hr,
                batch,
                lr_size=int(cfg.lr_size),
                downsample_mode=cfg.get("measurement", {}).get("downsample_mode", "nearest"),
            )
            metrics.update(sdedit_stats)
            metrics = scalarize_metrics(metrics)
            save_metrics(metrics, out_dir)
            wandb_images = {"lr_up": x_lr_up, "output": x_hr}
            if "hr" in batch:
                wandb_images["hr_gt"] = batch["hr"]
            _log_wandb_inference(wandb_run, args.mode, metrics, wandb_images)
        elif args.mode == "invert":
            x_img = batch["lr_up"] if "lr_up" in batch else batch["image"]
            if autoencoder is None:
                x_start = x_img
                conditioning_image = x_img
            else:
                x_start = autoencoder.encode_to_latent(x_img)
                rapsd_source = cfg.noise_scheduler.get("rapsd_source", "latent_lr")
                conditioning_image = x_start if concat_conditioning or rapsd_source == "latent_lr" else x_img
            z_t = inverter.invert(x_start, batch, condition_domain="LR", conditioning_image=conditioning_image)
            torch.save(z_t.detach().cpu(), out_dir / "terminal_state.pt")
            save_images(x_img, out_dir, "input")
            _log_wandb_inference(wandb_run, args.mode, {}, {"input": x_img})
        else:
            is_conditional_sampler = cfg.sampling.method in {"dps", "eps"}
            shape_img = batch["hr"] if "hr" in batch else batch["image"]
            cond_img = batch.get("lr_up", shape_img)
            if autoencoder is None:
                z_t = torch.randn_like(shape_img)
                conditioning_image = cond_img
            else:
                latent_shape = autoencoder.encode_to_latent(shape_img)
                z_t = torch.randn_like(latent_shape)
                cond_latent = autoencoder.encode_to_latent(cond_img)
                rapsd_source = cfg.noise_scheduler.get("rapsd_source", "latent_lr")
                conditioning_image = cond_latent if concat_conditioning or rapsd_source == "latent_lr" else cond_img
            x_state = inverter.sample(z_t, batch, condition_domain="HR", conditioning_image=conditioning_image)
            x = autoencoder.decode_to_image(x_state) if autoencoder is not None else x_state
            if is_conditional_sampler:
                x_lr_img = batch["lr_up"] if "lr_up" in batch else cond_img
                save_images(x_lr_img, out_dir, "lr_up")
                if "hr" in batch:
                    save_images(batch["hr"], out_dir, "hr_gt")
                save_images(x, out_dir, "output")
                wandb_images = {"lr_up": x_lr_img, "output": x}
                if "hr" in batch:
                    wandb_images["hr_gt"] = batch["hr"]
                wandb_mode = cfg.sampling.method
            else:
                save_images(x, out_dir, "sample")
                wandb_images = {"sample": x}
                wandb_mode = args.mode
            metrics = image_metrics(
                x,
                batch,
                lr_size=int(cfg.lr_size),
                downsample_mode=cfg.get("measurement", {}).get("downsample_mode", "nearest"),
            )
            metrics = scalarize_metrics(metrics)
            save_metrics(metrics, out_dir)
            _log_wandb_inference(wandb_run, wandb_mode, metrics, wandb_images)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

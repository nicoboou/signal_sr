from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import build_dataloader
from .diagnostics import compute_fid
from .inference import Sampler, spectral_native_lr_sdedit_sr
from .inference.inverter import freeze_for_inference
from .models import build_conditioner, build_denoiser, denoiser_channels, denoiser_conditioning_mode
from .objectives import Objective
from .schedules import build_noise_scheduler
from .schedules.spectral import load_hr_spectral_stats
from .utils.checkpoints import load_model_weights
from .utils.config import merge_dict, load_config, to_plain
from .utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID for a checkpoint with streamed generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["uncond", "dps", "eps", "spectral_native_lr_sdedit"],
        help="Generation method used for FID samples.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-samples", default="1000", help="Number of samples, or 'all' for the split size.")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for the diffusion model.")
    parser.add_argument(
        "--fid-device",
        default="cpu",
        help="Device for Inception feature extraction. Use 'cuda' for speed if VRAM allows; default 'cpu' minimizes VRAM.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weights", default="DEFAULT", help="Torchvision Inception weights, e.g. DEFAULT or none.")
    parser.add_argument("--resize-size", type=int, default=299)
    parser.add_argument("--no-cache-real-features", action="store_true")
    parser.add_argument("--empty-cache", action="store_true", help="Call torch.cuda.empty_cache() between generated batches.")

    parser.add_argument("--measurement-space", default="lr", choices=["lr", "lr_up"])
    parser.add_argument("--downsample-mode", default="area", choices=["area", "nearest", "bilinear", "bicubic"])
    parser.add_argument("--upsample-mode", default="nearest", choices=["nearest", "bilinear", "bicubic"])
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--base-sampler", default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--dps-start-step", type=int, default=None)
    parser.add_argument("--dps-end-step", type=int, default=None)
    parser.add_argument("--dps-loss", default=None, choices=["l2_norm", "mse"])
    parser.add_argument("--clip-denoised", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--hr-stats-path", default=None, help="Required for spectral_native_lr_sdedit.")
    parser.add_argument("--activation-snr", type=float, default=1.0)
    parser.add_argument("--allow-batch-size-gt-1", action="store_true")
    return parser.parse_args()


def _resolve_n_samples(value: str, dataset_len: int | None) -> int:
    if str(value).lower() in {"all", "none"}:
        if dataset_len is None:
            raise ValueError("--n-samples all requires a dataloader with a finite dataset")
        return int(dataset_len)
    n_samples = int(value)
    if n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    return min(n_samples, int(dataset_len)) if dataset_len is not None else n_samples


def _loader_len(loader) -> int | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _set_sampling_method(cfg, args):
    if args.method in {"uncond", "spectral_native_lr_sdedit"}:
        cfg.sampling.method = "ddim"
    elif args.method in {"dps", "eps"}:
        cfg.sampling.method = args.method
    else:
        raise ValueError(f"Unsupported method: {args.method}")


def _configure_dps(cfg, args):
    cfg.measurement = {"space": args.measurement_space, "downsample_mode": args.downsample_mode}
    dps_cfg = dict(cfg.sampling.get("dps", {}))
    if args.guidance_scale is not None:
        dps_cfg["guidance_scale"] = float(args.guidance_scale)
    dps_cfg.setdefault("guidance_scale", 5.0)
    if args.base_sampler is not None:
        dps_cfg["base_sampler"] = args.base_sampler
    dps_cfg.setdefault("base_sampler", "ddpm")
    if args.dps_start_step is not None:
        dps_cfg["start_step"] = int(args.dps_start_step)
    dps_cfg.setdefault("start_step", 0)
    if args.dps_end_step is not None:
        dps_cfg["end_step"] = int(args.dps_end_step)
    dps_cfg.setdefault("end_step", None)
    if args.dps_loss is not None:
        dps_cfg["loss"] = args.dps_loss
    dps_cfg.setdefault("loss", "l2_norm")
    if args.clip_denoised is not None:
        dps_cfg["clip_denoised"] = bool(args.clip_denoised)
    dps_cfg.setdefault("clip_denoised", True)
    cfg.sampling["dps"] = dps_cfg


def _clip_denoised(cfg, args):
    if args.clip_denoised is not None:
        return bool(args.clip_denoised)
    return cfg.sampling.get("clip_denoised", None)


def _interpolate(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def _move_batch(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def _make_measurement_batch(batch, cfg, args, device):
    batch = _move_batch(batch, device)
    if "hr" not in batch:
        raise ValueError(f"Method {args.method} requires dataloader batches with 'hr' images")
    x_hr = batch["hr"]
    y_lr = _interpolate(x_hr, (int(cfg.lr_size), int(cfg.lr_size)), mode=args.downsample_mode)
    y_lr_up = _interpolate(y_lr, (int(cfg.image_size), int(cfg.image_size)), mode=args.upsample_mode)
    out = dict(batch)
    out.update({"hr": x_hr, "lr": y_lr, "lr_up": y_lr_up, "image": y_lr_up})
    return out


def _real_hr_batches(loader, fid_device):
    for batch in loader:
        if "hr" not in batch:
            raise ValueError("FID real dataloader must yield batches with 'hr'")
        yield batch["hr"].to(fid_device) if fid_device.type == "cuda" else batch["hr"]


def _to_cpu_batch(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def _maybe_empty_cache(args, device):
    if args.empty_cache and device.type == "cuda":
        torch.cuda.empty_cache()


def _uncond_batches(cfg, sampler, objective, noise_scheduler, args, device, n_samples: int):
    if denoiser_conditioning_mode(cfg.denoiser) == "concat":
        raise ValueError("uncond FID does not support channel-concat conditioning")
    _, channels = denoiser_channels(cfg.denoiser)
    image_size = int(cfg.denoiser.params.sample_size)
    rng = torch.Generator(device=device).manual_seed(int(args.seed))
    emitted = 0
    dummy_batch = {}
    clip_d = _clip_denoised(cfg, args)
    while emitted < n_samples:
        current = min(int(args.generation_batch_size), int(n_samples) - emitted)
        z = torch.randn(current, channels, image_size, image_size, generator=rng, device=device)
        with torch.no_grad():
            if objective.name == "diffusion":
                x = sampler.ddim_loop(z, dummy_batch, "HR", conditioning_image=None, clip_denoised=clip_d)
            else:
                flow_cfg = cfg.sampling.get("flow", cfg.inversion.get("flow", {}))
                x = sampler.flow_loop(z, dummy_batch, "HR", conditioning_image=None, direction="reverse", n_steps=int(flow_cfg.get("n_steps", 50)))
        emitted += current
        yield _to_cpu_batch(x)
        del z, x
        _maybe_empty_cache(args, device)


def _dps_batches(loader, cfg, sampler, args, device):
    if denoiser_conditioning_mode(cfg.denoiser) == "concat":
        raise ValueError("dps FID does not support channel-concat conditioning in this script")
    for raw_batch in loader:
        batch = _make_measurement_batch(raw_batch, cfg, args, device)
        z = torch.randn_like(batch["hr"])
        with torch.enable_grad():
            x = sampler.dps_loop(z, batch, condition_domain="HR", conditioning_image=None)
        yield _to_cpu_batch(x)
        del raw_batch, batch, z, x
        _maybe_empty_cache(args, device)


def _eps_batches(loader, cfg, sampler, args, device):
    if denoiser_conditioning_mode(cfg.denoiser) == "concat":
        raise ValueError("eps FID does not support channel-concat conditioning in this script")
    for raw_batch in loader:
        batch = _make_measurement_batch(raw_batch, cfg, args, device)
        z = torch.randn_like(batch["hr"])
        with torch.no_grad():
            x = sampler.ddim_loop(z, batch, condition_domain="HR", conditioning_image=None, clip_denoised=_clip_denoised(cfg, args))
        yield _to_cpu_batch(x)
        del raw_batch, batch, z, x
        _maybe_empty_cache(args, device)


def _spectral_native_batches(loader, cfg, denoiser, sampler, noise_scheduler, hr_stats, args, device):
    if denoiser_conditioning_mode(cfg.denoiser) == "concat":
        raise ValueError("spectral_native_lr_sdedit FID does not support channel-concat conditioning")
    for raw_batch in loader:
        batch = _make_measurement_batch(raw_batch, cfg, args, device)
        if batch["lr"].shape[0] > 1 and not args.allow_batch_size_gt_1:
            raise ValueError("spectral_native_lr_sdedit supports batch size 1 unless --allow-batch-size-gt-1 is set")
        with torch.no_grad():
            x, _ = spectral_native_lr_sdedit_sr(
                y_lr=batch["lr"],
                model=denoiser,
                scheduler=noise_scheduler,
                hr_stats=hr_stats,
                target_hr_size=(int(cfg.image_size), int(cfg.image_size)),
                num_inference_steps=None,
                sampler=sampler,
                batch=batch,
                condition_domain="HR",
                clip_denoised=_clip_denoised(cfg, args),
                activation_snr=float(args.activation_snr),
                allow_batch_size_gt_1=bool(args.allow_batch_size_gt_1),
            )
        yield _to_cpu_batch(x)
        del raw_batch, batch, x
        _maybe_empty_cache(args, device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.space != "pixel":
        raise ValueError("compute_fid.py currently supports only pixel-space checkpoints")
    args.seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(args.seed)
    _set_sampling_method(cfg, args)
    if args.method == "dps":
        _configure_dps(cfg, args)
    if args.method == "spectral_native_lr_sdedit" and not args.allow_batch_size_gt_1 and args.generation_batch_size != 1:
        print("spectral_native_lr_sdedit V1 uses batch size 1; overriding --generation-batch-size to 1")
        args.generation_batch_size = 1

    device = torch.device(args.device)
    fid_device = torch.device(args.fid_device)
    output_dir = Path(args.output_dir or f"outputs/fid/{Path(args.config).stem}/{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    real_cfg = merge_dict(cfg.data, cfg.get(f"{args.split}_data", {}))
    real_cfg.return_pair = True
    real_loader = build_dataloader(real_cfg, split=args.split, shuffle=False, batch_size=args.feature_batch_size, num_workers=args.num_workers)
    n_samples = _resolve_n_samples(args.n_samples, _loader_len(real_loader))

    gen_loader = None
    if args.method != "uncond":
        gen_cfg = merge_dict(cfg.data, cfg.get(f"{args.split}_data", {}))
        gen_cfg.return_pair = True
        gen_loader = build_dataloader(gen_cfg, split=args.split, shuffle=False, batch_size=args.generation_batch_size, num_workers=args.num_workers)

    denoiser = build_denoiser(cfg.denoiser, data_channels=int(cfg.data.channels)).to(device)
    conditioner = build_conditioner().to(device)
    load_model_weights(args.checkpoint, denoiser=denoiser, conditioner=conditioner, map_location=device)
    freeze_for_inference(denoiser, conditioner)

    noise_scheduler = build_noise_scheduler(cfg.noise_scheduler)
    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    objective = Objective(cfg.objective, noise_scheduler)
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)

    if args.method == "uncond":
        generated = _uncond_batches(cfg, sampler, objective, noise_scheduler, args, device, n_samples)
    elif args.method == "dps":
        generated = _dps_batches(gen_loader, cfg, sampler, args, device)
    elif args.method == "eps":
        generated = _eps_batches(gen_loader, cfg, sampler, args, device)
    elif args.method == "spectral_native_lr_sdedit":
        if args.hr_stats_path is None:
            raise ValueError("--hr-stats-path is required for --method spectral_native_lr_sdedit")
        hr_stats = load_hr_spectral_stats(args.hr_stats_path, map_location=device)
        generated = _spectral_native_batches(gen_loader, cfg, denoiser, sampler, noise_scheduler, hr_stats, args, device)
    else:
        raise ValueError(args.method)

    real_cache = None if args.no_cache_real_features else output_dir / f"real_features_{args.split}_n{n_samples}_{args.weights}_s{args.resize_size}.pt"
    fid, details = compute_fid(
        generated,
        _real_hr_batches(real_loader, fid_device=fid_device),
        n_samples=n_samples,
        batch_size=args.feature_batch_size,
        device=fid_device,
        weights=None if str(args.weights).lower() in {"none", "random"} else args.weights,
        resize_size=args.resize_size,
        real_features_cache_path=real_cache,
        cache_real_features=not args.no_cache_real_features,
        return_details=True,
    )

    summary = {
        "fid": float(fid.detach().cpu()),
        "method": args.method,
        "n_samples": int(details["n_samples"]),
        "feature_dim": int(details["feature_dim"]),
        "real_feature_cache_hit": bool(details["real_feature_cache_hit"]),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": int(args.seed),
        "generation_batch_size": int(args.generation_batch_size),
        "feature_batch_size": int(args.feature_batch_size),
        "device": str(device),
        "fid_device": str(fid_device),
        "weights": args.weights,
        "resize_size": int(args.resize_size),
        "measurement": to_plain(cfg.get("measurement", {})),
        "sampling": to_plain(cfg.get("sampling", {})),
    }
    if args.hr_stats_path is not None:
        summary["hr_stats_path"] = str(args.hr_stats_path)
    out_path = output_dir / "fid.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from .data import build_dataloader
from .diagnostics import (
    collect_image_features_stream,
    compute_fid,
    image_metrics,
    latent_metrics,
    load_or_compute_real_mind_features,
    monge_inception_distance_torch,
    scalarize_metrics,
)
from .inference.eps import eps_observation_from_hr, eps_pivot, validate_eps_config
from .inference import Inverter, Sampler
from .models import build_conditioner, build_denoiser, denoiser_channels, denoiser_conditioning_mode
from .objectives import Objective
from .schedules import build_noise_scheduler
from .splits import ensure_synthetic_split
from .utils.checkpoints import save_checkpoint
from .utils.config import load_config, merge_dict, to_plain
from .utils.images import make_image_grid, save_images, tensor_to_pil_image
from .utils.logging import make_logs
from .utils.outputs import default_train_output_base, make_run_output_dir, resolve_path
from .utils.seed import seed_everything

FIXED_VAL_NOISE_SEED = 9999


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_cfg(cfg):
    if cfg.space != "pixel":
        raise ValueError("train_pixel.py only supports space=pixel")
    if cfg.objective.name not in {"diffusion", "flow_matching"}:
        raise ValueError("Unsupported objective for pixel training")
    if cfg.objective.prediction_type != cfg.noise_scheduler.prediction_type:
        raise ValueError("objective.prediction_type must match noise_scheduler.prediction_type")
    _, out_channels = denoiser_channels(cfg.denoiser)
    if int(cfg.data.channels) != out_channels:
        raise ValueError("data.channels must match denoiser.params.out_channels")
    if cfg.sampling.method == "eps":
        in_channels, out_channels = denoiser_channels(cfg.denoiser)
        if in_channels != 2 * out_channels:
            raise ValueError("EPS expects denoiser.params.in_channels == 2 * out_channels for [mu_star, y_up]")
        validate_eps_config(cfg, denoiser_mode=denoiser_conditioning_mode(cfg.denoiser))
        if cfg.objective.name != "diffusion" or cfg.objective.prediction_type != "sample":
            raise ValueError("EPS training requires objective.name=diffusion and prediction_type=sample")


def _uses_concat_conditioning(cfg):
    return denoiser_conditioning_mode(cfg.denoiser) == "concat"


def _uses_eps(cfg):
    return cfg.sampling.method == "eps"


def _ddim_clip_denoised(cfg):
    ddim_cfg = cfg.sampling.get("ddim", {})
    if "clip_denoised" in ddim_cfg:
        return bool(ddim_cfg.get("clip_denoised"))
    if "clip_sample" in cfg.noise_scheduler:
        return bool(cfg.noise_scheduler.get("clip_sample"))
    return None


def _training_tensors(batch, concat_conditioning):
    if not concat_conditioning:
        return batch["image"], None
    if "hr" not in batch or "lr_up" not in batch:
        raise ValueError("Channel-concat conditioning requires batches with hr and lr_up tensors")
    return batch["hr"], batch["lr_up"]


def _move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "_")


def _metric_cfg(cfg, name):
    return cfg.get("evaluation", {}).get(name, {})


def _metric_enabled(cfg, name):
    return bool(_metric_cfg(cfg, name).get("enabled", False))


def _metric_every(cfg, name):
    return int(_metric_cfg(cfg, name).get("every", cfg.train.get("save_every", 0)))


def _loader_num_samples(loader):
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _wandb_init_kwargs(cfg):
    wandb_cfg = cfg.get("wandb", {})
    init_kwargs = {}
    run_id = cfg.train.get("run_id", None)
    if run_id:
        init_kwargs["id"] = str(wandb_cfg.get("id", run_id))
        init_kwargs["name"] = str(wandb_cfg.get("name", run_id))
        init_kwargs["resume"] = wandb_cfg.get("resume", "allow")
    for key in ("entity", "group", "tags", "notes", "job_type"):
        if key in wandb_cfg:
            init_kwargs[key] = wandb_cfg[key]
    return {"wandb": init_kwargs} if init_kwargs else None


def _effective_metric_n_samples(metric_cfg, loader, default=50000):
    requested = metric_cfg.get("n_samples", default)
    available = _loader_num_samples(loader)
    if requested is None or (isinstance(requested, str) and requested.lower() in {"all", "none"}):
        return available
    requested = int(requested)
    return min(requested, available) if available is not None else requested


def _metric_real_split(metric_cfg):
    return str(metric_cfg.get("real_split", "train"))


def _real_feature_cache_path(cfg, output_dir, metric_name, n_samples, real_split):
    metric_cfg = _metric_cfg(cfg, metric_name)
    weights = _safe_name(metric_cfg.get("weights", "DEFAULT"))
    resize_size = int(metric_cfg.get("resize_size", 299))
    sample_tag = "all" if n_samples is None else str(int(n_samples))
    return Path(output_dir) / "metrics" / f"{metric_name}_real_features_{real_split}_n{sample_tag}_{weights}_s{resize_size}.pt"


def _real_hr_batches(loader):
    for batch in loader:
        yield batch["hr"]


def _random_hr_batches(cfg, sampler, objective, noise_scheduler, device, n_samples, batch_size, rng_seed):
    in_channels, channels = denoiser_channels(cfg.denoiser)
    concat_channels = in_channels - channels
    image_size = int(cfg.denoiser.params.sample_size)
    concat_conditioning = _uses_concat_conditioning(cfg)
    rng = torch.Generator(device=device).manual_seed(int(rng_seed))
    emitted = 0
    clip_d = _ddim_clip_denoised(cfg)
    dummy_batch = {"domain": None}
    while emitted < n_samples:
        current = min(int(batch_size), int(n_samples) - emitted)
        z_t = torch.randn(current, channels, image_size, image_size, generator=rng, device=device)
        conditioning_image = torch.randn(current, concat_channels, image_size, image_size, generator=rng, device=device) if concat_conditioning else z_t
        if objective.name == "diffusion":
            x_out = sampler.ddim_loop(z_t, dummy_batch, "HR", conditioning_image=conditioning_image, clip_denoised=clip_d)
        else:
            flow_cfg = cfg.sampling.get("flow", cfg.inversion.get("flow", {}))
            x_out = sampler.flow_loop(z_t, dummy_batch, "HR", conditioning_image=conditioning_image, direction="reverse", n_steps=int(flow_cfg.get("n_steps", 50)))
        emitted += current
        yield x_out


def _log_tensor_stats(x, name, step, accelerator=None):
    xf = x.detach().float()
    min_v = xf.min().item()
    max_v = xf.max().item()
    mean_v = xf.mean().item()
    std_v = xf.std().item()
    below = (xf < -1.0).float().mean().item()
    above = (xf > 1.0).float().mean().item()
    nan_frac = (xf != xf).float().mean().item()
    if accelerator is not None:
        log_payload = {
            f"diagnostics/{name}/min": min_v,
            f"diagnostics/{name}/max": max_v,
            f"diagnostics/{name}/mean": mean_v,
            f"diagnostics/{name}/std": std_v,
            f"diagnostics/{name}/fraction_below_minus_1": below,
            f"diagnostics/{name}/fraction_above_1": above,
            f"diagnostics/{name}/fraction_nan": nan_frac,
        }
        channel_msg = ""
        if xf.ndim == 4:
            channel_mean = xf.mean(dim=(0, 2, 3))
            channel_std = xf.std(dim=(0, 2, 3))
            channel_min = xf.amin(dim=(0, 2, 3))
            channel_max = xf.amax(dim=(0, 2, 3))
            for channel in range(min(xf.shape[1], 4)):
                log_payload[f"diagnostics/{name}/channel_{channel}/mean"] = channel_mean[channel].item()
                log_payload[f"diagnostics/{name}/channel_{channel}/std"] = channel_std[channel].item()
                log_payload[f"diagnostics/{name}/channel_{channel}/min"] = channel_min[channel].item()
                log_payload[f"diagnostics/{name}/channel_{channel}/max"] = channel_max[channel].item()
            channel_msg = " channels=" + ",".join(
                f"c{channel}:mean={channel_mean[channel].item():.4f},std={channel_std[channel].item():.4f},min={channel_min[channel].item():.4f},max={channel_max[channel].item():.4f}"
                for channel in range(min(xf.shape[1], 4))
            )
        accelerator.print(
            f"step={step} {name} tensor: min={min_v:.6f} max={max_v:.6f} "
            f"mean={mean_v:.6f} std={std_v:.6f} below_-1={below:.6f} above_+1={above:.6f} nan={nan_frac:.6f}"
            f"{channel_msg}"
        )
        accelerator.log(log_payload, step=step)


def _log_wandb_images(images_dict, step):
    try:
        import wandb

        if wandb.run is None:
            return {}
        out = {}
        for key, tensor in images_dict.items():
            imgs = tensor.detach().cpu()
            for i in range(imgs.shape[0]):
                out[f"val/{key}/sample_{i}"] = wandb.Image(tensor_to_pil_image(imgs[i]))
        return out
    except ImportError:
        return {}


def run_random_hr_validation(
    cfg,
    denoiser,
    conditioner,
    objective,
    noise_scheduler,
    output_dir,
    step,
    device,
    accelerator=None,
):
    was_training = denoiser.training
    denoiser.eval()
    conditioner.eval()
    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    in_channels, channels = denoiser_channels(cfg.denoiser)
    concat_channels = in_channels - channels
    image_size = int(cfg.denoiser.params.sample_size)
    concat_conditioning = _uses_concat_conditioning(cfg)
    batch_size = 16
    rng = torch.Generator(device=device).manual_seed(FIXED_VAL_NOISE_SEED)
    z_t = torch.randn(batch_size, channels, image_size, image_size, generator=rng, device=device)
    conditioning_image = torch.randn(batch_size, concat_channels, image_size, image_size, generator=rng, device=device) if concat_conditioning else z_t
    dummy_batch = {"domain": None}
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)
    clip_d = _ddim_clip_denoised(cfg)
    with torch.no_grad():
        x_out = sampler.ddim_loop(z_t, dummy_batch, "HR", conditioning_image=conditioning_image, clip_denoised=clip_d)
    _log_tensor_stats(x_out, "hr_random", step, accelerator=accelerator)
    sample_dir = Path(output_dir) / "samples" / f"step-{step:08d}"
    save_images(x_out, sample_dir, "hr_random")
    grid = make_image_grid(x_out, nrow=4)
    pil_grid = tensor_to_pil_image(grid)
    grid_path = sample_dir / "grid_4x4.png"
    pil_grid.save(grid_path)
    if accelerator is not None:
        import wandb

        if wandb.run is not None:
            grid_wandb = wandb.Image(pil_grid, caption=f"HR random samples step {step}")
            accelerator.log({"val/hr_random_grid": grid_wandb}, step=step)
        accelerator.log(_log_wandb_images({"hr_random": x_out}, step), step=step)
    if was_training:
        denoiser.train()
        conditioner.train()
    return {}


def run_validation_samples(
    cfg,
    val_loader,
    denoiser,
    conditioner,
    objective,
    noise_scheduler,
    output_dir,
    step,
    device,
    accelerator=None,
):
    if val_loader is None:
        return {}
    was_training = denoiser.training
    denoiser.eval()
    conditioner.eval()
    batch = _move_batch(next(iter(val_loader)), device)
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)
    inverter = Inverter(
        cfg.inversion,
        cfg.sampling,
        objective,
        noise_scheduler,
        denoiser,
        conditioner,
        sampler,
    )
    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    x_lr = batch["lr_up"] if "lr_up" in batch else batch["image"]
    extra_metrics = {}
    if cfg.sampling.method in {"dps", "eps"}:
        shape_img = batch["hr"] if "hr" in batch else batch["image"]
        rng = torch.Generator(device=device).manual_seed(FIXED_VAL_NOISE_SEED)
        z_t = torch.randn(shape_img.shape, generator=rng, device=device, dtype=shape_img.dtype)
        context = torch.enable_grad() if cfg.sampling.method == "dps" else torch.no_grad()
        with context:
            x_out = inverter.sample(z_t, batch, condition_domain="HR", conditioning_image=x_lr)
    else:
        with torch.no_grad():
            x_out, z_t = inverter.invert_and_sample(x_lr, batch, conditioning_image=x_lr)
        extra_metrics.update(latent_metrics(z_t))
    _log_tensor_stats(x_out, "output", step, accelerator=accelerator)
    _log_tensor_stats(x_lr, "lr_up", step, accelerator=accelerator)
    if "hr" in batch:
        _log_tensor_stats(batch["hr"], "hr_gt", step, accelerator=accelerator)
    sample_dir = Path(output_dir) / "samples" / f"step-{step:08d}"
    save_images(x_lr, sample_dir, "lr_up")
    if "hr" in batch:
        save_images(batch["hr"], sample_dir, "hr_gt")
    save_images(x_out, sample_dir, "output")
    if accelerator is not None:
        wandb_images = {"lr_up": x_lr, "output": x_out}
        if "hr" in batch:
            wandb_images["hr_gt"] = batch["hr"]
        accelerator.log(_log_wandb_images(wandb_images, step), step=step)
    metrics = image_metrics(
        x_out,
        batch,
        lr_size=int(cfg.lr_size),
        downsample_mode=cfg.get("measurement", {}).get("downsample_mode", "nearest"),
    )
    metrics.update(extra_metrics)
    scalars = scalarize_metrics(metrics)
    with (sample_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(scalars, handle, indent=2)
    if was_training:
        denoiser.train()
        conditioner.train()
    return scalars


def run_mind_validation(
    cfg,
    mind_loader,
    denoiser,
    conditioner,
    objective,
    noise_scheduler,
    output_dir,
    step,
    device,
    accelerator=None,
):
    if mind_loader is None:
        return {}
    mind_cfg = _metric_cfg(cfg, "mind")
    n_samples = _effective_metric_n_samples(mind_cfg, mind_loader, default=50000)
    n_projections = int(mind_cfg.get("n_projections", 1000))
    rng_seed = int(mind_cfg.get("rng_seed", 0))
    generation_batch_size = int(mind_cfg.get("generation_batch_size", 16))
    feature_batch_size = int(mind_cfg.get("feature_batch_size", 32))
    weights = mind_cfg.get("weights", "DEFAULT")
    resize_size = int(mind_cfg.get("resize_size", 299))
    cache_real_features = bool(mind_cfg.get("cache_real_features", True))
    real_split = _metric_real_split(mind_cfg)

    was_training = denoiser.training
    conditioner_was_training = conditioner.training
    denoiser.eval()
    conditioner.eval()
    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)

    cache_path = _real_feature_cache_path(cfg, output_dir, "mind", n_samples, real_split) if cache_real_features else None
    with torch.no_grad():
        real_features, cache_hit = load_or_compute_real_mind_features(
            _real_hr_batches(mind_loader),
            n_samples=n_samples,
            cache_path=cache_path,
            cache=cache_real_features,
            batch_size=feature_batch_size,
            device=device,
            weights=weights,
            resize_size=resize_size,
        )
        n_samples = int(real_features.shape[0])
        generated_features = collect_image_features_stream(
            _random_hr_batches(cfg, sampler, objective, noise_scheduler, device, n_samples, generation_batch_size, rng_seed),
            n_samples=n_samples,
            batch_size=feature_batch_size,
            device=device,
            weights=weights,
            resize_size=resize_size,
        )
        mind = monge_inception_distance_torch(generated_features, real_features, rng_seed=rng_seed, n_projections=n_projections)

    scalars = {
        "mind": float(mind.detach().cpu()),
        "mind/n_samples": float(n_samples),
        "mind/n_projections": float(n_projections),
        "mind/feature_dim": float(generated_features.shape[1]),
        "mind/real_feature_cache_hit": float(cache_hit),
    }
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"mind_step-{step:08d}.json").open("w", encoding="utf-8") as handle:
        json.dump(scalars, handle, indent=2)
    if was_training:
        denoiser.train()
    if conditioner_was_training:
        conditioner.train()
    if accelerator is not None:
        accelerator.print(f"step={step} mind={scalars['mind']:.6f}")
    return scalars


def run_fid_validation(
    cfg,
    fid_loader,
    denoiser,
    conditioner,
    objective,
    noise_scheduler,
    output_dir,
    step,
    device,
    accelerator=None,
):
    if fid_loader is None:
        return {}
    fid_cfg = _metric_cfg(cfg, "fid")
    n_samples = _effective_metric_n_samples(fid_cfg, fid_loader, default=50000)
    rng_seed = int(fid_cfg.get("rng_seed", 0))
    generation_batch_size = int(fid_cfg.get("generation_batch_size", 16))
    feature_batch_size = int(fid_cfg.get("feature_batch_size", 32))
    weights = fid_cfg.get("weights", "DEFAULT")
    resize_size = int(fid_cfg.get("resize_size", 299))
    cache_real_features = bool(fid_cfg.get("cache_real_features", True))
    real_split = _metric_real_split(fid_cfg)

    if n_samples is None:
        raise ValueError("FID n_samples='all' requires a dataloader with a finite dataset")

    was_training = denoiser.training
    conditioner_was_training = conditioner.training
    denoiser.eval()
    conditioner.eval()
    noise_scheduler.set_timesteps(cfg.noise_scheduler.n_infer_steps, device=device)
    sampler = Sampler(cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=cfg)

    cache_path = _real_feature_cache_path(cfg, output_dir, "fid", n_samples, real_split) if cache_real_features else None
    with torch.no_grad():
        fid, details = compute_fid(
            _random_hr_batches(cfg, sampler, objective, noise_scheduler, device, n_samples, generation_batch_size, rng_seed),
            _real_hr_batches(fid_loader),
            n_samples=n_samples,
            batch_size=feature_batch_size,
            device=device,
            weights=weights,
            resize_size=resize_size,
            real_features_cache_path=cache_path,
            cache_real_features=cache_real_features,
            return_details=True,
        )

    scalars = {
        "fid": float(fid.detach().cpu()),
        "fid/n_samples": float(details["n_samples"]),
        "fid/feature_dim": float(details["feature_dim"]),
        "fid/real_feature_cache_hit": float(details["real_feature_cache_hit"]),
    }
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"fid_step-{step:08d}.json").open("w", encoding="utf-8") as handle:
        json.dump(scalars, handle, indent=2)
    if was_training:
        denoiser.train()
    if conditioner_was_training:
        conditioner.train()
    if accelerator is not None:
        accelerator.print(f"step={step} fid={scalars['fid']:.6f}")
    return scalars


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_cfg(cfg)
    seed_everything(int(cfg.get("seed", 0)))

    if cfg.data.name == "synthetic_microscopy":
        ensure_synthetic_split(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg.train.get("mixed_precision", "no"),
        log_with="wandb" if cfg.get("wandb", {}).get("enabled", False) else None,
    )

    base_output_dir = resolve_path(args.output_dir or cfg.train.get("output_dir", default_train_output_base(args.config)))
    output_info = [str(base_output_dir), None, None]
    if accelerator.is_main_process:
        output_dir, run_id = make_run_output_dir(base_output_dir, args.run_id or cfg.train.get("run_id", None))
        output_info = [str(base_output_dir), str(output_dir), run_id]
        with (output_dir / "argv.json").open("w", encoding="utf-8") as handle:
            json.dump({"argv": sys.argv}, handle, indent=2)
    broadcast_object_list(output_info)
    base_output_dir = Path(output_info[0])
    output_dir = Path(output_info[1])
    cfg.train.base_output_dir = str(base_output_dir)
    cfg.train.output_dir = str(output_dir)
    cfg.train.run_id = output_info[2]

    if cfg.get("wandb", {}).get("enabled", False):
        wandb_init_kwargs = _wandb_init_kwargs(cfg)
        if wandb_init_kwargs:
            accelerator.init_trackers(cfg.wandb.project, config=to_plain(cfg), init_kwargs=wandb_init_kwargs)
        else:
            accelerator.init_trackers(cfg.wandb.project, config=to_plain(cfg))

    concat_conditioning = _uses_concat_conditioning(cfg)
    train_data_cfg = merge_dict(cfg.data, {})
    if concat_conditioning:
        train_data_cfg.return_pair = True

    batch_size = args.batch_size or int(cfg.train.batch_size)
    loader = build_dataloader(
        train_data_cfg,
        split="train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    val_cfg = merge_dict(cfg.data, cfg.get("val_data", {}))
    val_cfg.return_pair = True
    val_loader = build_dataloader(
        val_cfg,
        split="val",
        shuffle=False,
        batch_size=min(4, batch_size),
        num_workers=args.num_workers,
    )
    mind_loader = None
    if _metric_enabled(cfg, "mind"):
        mind_cfg = _metric_cfg(cfg, "mind")
        mind_real_cfg = merge_dict(cfg.data, mind_cfg.get("real_data", {}))
        mind_real_cfg.return_pair = True
        mind_loader = build_dataloader(
            mind_real_cfg,
            split=_metric_real_split(mind_cfg),
            shuffle=False,
            batch_size=int(mind_cfg.get("real_batch_size", mind_cfg.get("generation_batch_size", 16))),
            num_workers=args.num_workers,
        )
    fid_loader = None
    if _metric_enabled(cfg, "fid"):
        fid_cfg = _metric_cfg(cfg, "fid")
        fid_real_cfg = merge_dict(cfg.data, fid_cfg.get("real_data", {}))
        fid_real_cfg.return_pair = True
        fid_loader = build_dataloader(
            fid_real_cfg,
            split=_metric_real_split(fid_cfg),
            shuffle=False,
            batch_size=int(fid_cfg.get("real_batch_size", fid_cfg.get("generation_batch_size", 16))),
            num_workers=args.num_workers,
        )

    denoiser = build_denoiser(cfg.denoiser, data_channels=cfg.data.channels)
    conditioner = build_conditioner()
    noise_scheduler = build_noise_scheduler(cfg.noise_scheduler)
    objective = Objective(cfg.objective, noise_scheduler)

    params = list(denoiser.parameters()) + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg.train.lr))
    denoiser, conditioner, optimizer, loader = accelerator.prepare(denoiser, conditioner, optimizer, loader)

    max_steps = int(args.max_steps or cfg.train.max_steps)
    log_every = int(cfg.train.get("log_every", 50))
    save_every = int(cfg.train.get("save_every", 10000))
    sample_every = int(cfg.train.get("sample_every", 0))
    mind_every = _metric_every(cfg, "mind") if _metric_enabled(cfg, "mind") else 0
    fid_every = _metric_every(cfg, "fid") if _metric_enabled(cfg, "fid") else 0
    step = 0

    denoiser.train()
    conditioner.train()
    while step < max_steps:
        for batch in loader:
            if _uses_eps(cfg):
                if "hr" not in batch:
                    raise ValueError("EPS training requires train_data_cfg.return_pair=true so batches contain hr")
                x0 = batch["hr"]
                conditioning_image = None
            else:
                x0, conditioning_image = _training_tensors(batch, concat_conditioning)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (x0.shape[0],), device=x0.device)
            if _uses_eps(cfg):
                noise = torch.randn_like(x0)
                noisy_x = noise_scheduler.add_noise(x0, noise, timesteps, image=x0)
                y_lr = eps_observation_from_hr(x0, cfg, noisy=True)
                mu_star, conditioning_image = eps_pivot(noisy_x, y_lr, timesteps, noise_scheduler, cfg)
                model_input = noise_scheduler.scale_model_input(mu_star, timesteps)
                target = x0
            elif objective.name == "diffusion":
                noise = torch.randn_like(x0)
                noisy_x = noise_scheduler.add_noise(x0, noise, timesteps, image=x0)
                model_input = noise_scheduler.scale_model_input(noisy_x, timesteps)
                target = objective.training_target(x0, noise, timesteps, image=x0)
            else:
                model_input, target = objective.prepare_flow_training_input(x0, timesteps)
            cond = conditioner(batch, conditioning_image=conditioning_image)
            pred = denoiser(model_input, timesteps, cond)
            loss = objective.loss(pred, target)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                logs = make_logs(loss, noise_scheduler, timesteps, x0)
                accelerator.log(logs, step=step)
                _log_tensor_stats(x0, "train_x0", step, accelerator=accelerator)
                if accelerator.is_main_process:
                    accelerator.print(f"step={step} loss={loss.detach().float().item():.6f}")

            if sample_every > 0 and step % sample_every == 0 and accelerator.is_main_process:
                if cfg.data.get("split_strategy") == "train_hr_val_test_lr" and not _uses_eps(cfg):
                    val_logs = run_random_hr_validation(
                        cfg,
                        denoiser,
                        conditioner,
                        objective,
                        noise_scheduler,
                        output_dir,
                        step,
                        accelerator.device,
                        accelerator=accelerator,
                    )
                else:
                    val_logs = run_validation_samples(
                        cfg,
                        val_loader,
                        denoiser,
                        conditioner,
                        objective,
                        noise_scheduler,
                        output_dir,
                        step,
                        accelerator.device,
                        accelerator=accelerator,
                    )
                if val_logs:
                    accelerator.log({f"val/{k}": v for k, v in val_logs.items()}, step=step)

            if mind_every > 0 and step > 0 and step % mind_every == 0 and accelerator.is_main_process:
                mind_logs = run_mind_validation(
                    cfg,
                    mind_loader,
                    denoiser,
                    conditioner,
                    objective,
                    noise_scheduler,
                    output_dir,
                    step,
                    accelerator.device,
                    accelerator=accelerator,
                )
                if mind_logs:
                    accelerator.log({f"val/{k}": v for k, v in mind_logs.items()}, step=step)

            if fid_every > 0 and step > 0 and step % fid_every == 0 and accelerator.is_main_process:
                fid_logs = run_fid_validation(
                    cfg,
                    fid_loader,
                    denoiser,
                    conditioner,
                    objective,
                    noise_scheduler,
                    output_dir,
                    step,
                    accelerator.device,
                    accelerator=accelerator,
                )
                if fid_logs:
                    accelerator.log({f"val/{k}": v for k, v in fid_logs.items()}, step=step)

            if step > 0 and step % save_every == 0 and accelerator.is_main_process:
                save_checkpoint(
                    output_dir / f"checkpoint-{step}",
                    denoiser,
                    conditioner,
                    optimizer,
                    cfg,
                    step,
                    accelerator,
                )

            step += 1
            if step >= max_steps:
                break

    if accelerator.is_main_process:
        save_checkpoint(
            output_dir / "checkpoint-final",
            denoiser,
            conditioner,
            optimizer,
            cfg,
            step,
            accelerator,
        )
        if mind_every > 0:
            mind_logs = run_mind_validation(
                cfg,
                mind_loader,
                denoiser,
                conditioner,
                objective,
                noise_scheduler,
                output_dir,
                step,
                accelerator.device,
                accelerator=accelerator,
            )
            if mind_logs:
                accelerator.log({f"val/{k}": v for k, v in mind_logs.items()}, step=step)
        if fid_every > 0:
            fid_logs = run_fid_validation(
                cfg,
                fid_loader,
                denoiser,
                conditioner,
                objective,
                noise_scheduler,
                output_dir,
                step,
                accelerator.device,
                accelerator=accelerator,
            )
            if fid_logs:
                accelerator.log({f"val/{k}": v for k, v in fid_logs.items()}, step=step)
    accelerator.end_training()


if __name__ == "__main__":
    main()

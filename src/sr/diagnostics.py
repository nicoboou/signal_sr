from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .schedules.spectral import rapsd


_INCEPTION_FEATURE_EXTRACTORS = {}


def _load_tensor(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_n_samples(n_samples):
    if n_samples is None:
        return None
    if isinstance(n_samples, str) and n_samples.lower() in {"all", "none"}:
        return None
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return n_samples


def latent_metrics(z_t, k_lf=3):
    k, psd = rapsd(z_t)
    logp = torch.log(psd + 1e-12)
    flatness = logp.std(dim=1)
    lf_leakage = logp[:, :k_lf].mean(dim=1) - logp.mean(dim=1)
    return {"latent/flatness": flatness, "latent/lf_leakage": lf_leakage, "k": k, "psd": psd}


def downsample_l1(output, target_lr, lr_size, mode="nearest"):
    if mode == "nearest":
        pred_lr = F.interpolate(output, size=(lr_size, lr_size), mode="nearest")
    elif mode == "area":
        pred_lr = F.interpolate(output, size=(lr_size, lr_size), mode="area")
    else:
        raise ValueError(mode)
    return (pred_lr - target_lr).abs().mean(dim=(1, 2, 3))


def roundtrip_l1(reconstruction, target):
    return (reconstruction - target).abs().mean(dim=(1, 2, 3))


def rapsd_error(output, target, eps=1e-12):
    _, psd_out = rapsd(output)
    _, psd_target = rapsd(target)
    return (torch.log(psd_out + eps) - torch.log(psd_target + eps)).abs().mean(dim=1)


def monge_inception_distance_torch(x, y, rng_seed, n_projections=1000):
    """Monge Inception Distance on precomputed features.

    Args:
        x: Generated features with shape [N,D].
        y: Ground-truth features with shape [N,D].
        rng_seed: Seed for random projection directions.
        n_projections: Number of random unit projections.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected feature tensors [N,D], got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape != y.shape:
        raise ValueError(f"MIND requires equal feature shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] == 0:
        raise ValueError("MIND requires at least one sample")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    if x.device != y.device:
        raise ValueError("MIND feature tensors must be on the same device")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise ValueError("MIND feature tensors must be floating point")

    y = y.to(dtype=x.dtype)
    _, d = x.shape
    alpha = 3 * d
    generator = torch.Generator(device=x.device).manual_seed(int(rng_seed))
    u_proj = torch.randn((int(n_projections), d), generator=generator, dtype=x.dtype, device=x.device)
    u_proj = u_proj / torch.linalg.norm(u_proj, dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)

    x_proj = u_proj @ x.T
    y_proj = u_proj @ y.T
    dists = (torch.sort(x_proj, dim=-1).values - torch.sort(y_proj, dim=-1).values).square().mean(dim=1)
    return alpha * dists.mean()


def _feature_mean_and_cov(features):
    if features.ndim != 2:
        raise ValueError(f"Expected feature tensor [N,D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("FID requires at least two samples")
    features = features.to(dtype=torch.float64)
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T @ centered / (features.shape[0] - 1)
    return mean, cov


def _matrix_sqrt_psd(matrix):
    matrix = (matrix + matrix.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(0.0)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.T


def frechet_inception_distance_torch(x, y):
    """Frechet Inception Distance on precomputed features.

    Args:
        x: Generated features with shape [N,D].
        y: Real features with shape [M,D].
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected feature tensors [N,D], got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"FID feature dimensions must match, got {x.shape[1]} and {y.shape[1]}")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise ValueError("FID feature tensors must be floating point")
    if x.device != y.device:
        y = y.to(x.device)

    mu_x, cov_x = _feature_mean_and_cov(x)
    mu_y, cov_y = _feature_mean_and_cov(y)
    diff = mu_x - mu_y
    sqrt_cov_x = _matrix_sqrt_psd(cov_x)
    covmean = _matrix_sqrt_psd(sqrt_cov_x @ cov_y @ sqrt_cov_x)
    fid = diff.dot(diff) + torch.trace(cov_x) + torch.trace(cov_y) - 2.0 * torch.trace(covmean)
    return fid.clamp_min(0.0)


def _load_torchvision_models():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError("Inception image features require torchvision. Install torchvision or pass a custom feature_extractor.") from exc
    return models


def _resolve_inception_weights(models, weights):
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights.lower() in {"none", "random"}:
            return None
        enum = models.Inception_V3_Weights
        if weights.upper() == "DEFAULT":
            return enum.DEFAULT
        return enum[weights]
    return weights


def _get_inception_feature_extractor(device, weights="DEFAULT", progress=False):
    models = _load_torchvision_models()
    resolved_weights = _resolve_inception_weights(models, weights)
    weights_key = "none" if resolved_weights is None else str(resolved_weights)
    device = torch.device(device)
    cache_key = (str(device), weights_key)
    if cache_key in _INCEPTION_FEATURE_EXTRACTORS:
        return _INCEPTION_FEATURE_EXTRACTORS[cache_key]

    if resolved_weights is None:
        model = models.inception_v3(weights=None, aux_logits=False, transform_input=False, init_weights=True)
    else:
        model = models.inception_v3(
            weights=None,
            aux_logits=True,
            transform_input=False,
            init_weights=False,
            num_classes=len(resolved_weights.meta["categories"]),
        )
        model.load_state_dict(resolved_weights.get_state_dict(progress=progress, check_hash=True))
        model.aux_logits = False
        model.AuxLogits = None
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _INCEPTION_FEATURE_EXTRACTORS[cache_key] = model
    return model


def _prepare_inception_images(images, size=299):
    if images.ndim != 4:
        raise ValueError(f"Expected images [N,C,H,W], got {tuple(images.shape)}")
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] != 3:
        raise ValueError(f"Inception features require 1 or 3 channels, got {images.shape[1]}")
    images = (images.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    images = F.interpolate(images, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)[None, :, None, None]
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)[None, :, None, None]
    return (images - mean) / std


@torch.no_grad()
def inception_features(images, batch_size=32, device=None, weights="DEFAULT", resize_size=299, progress=False):
    device = torch.device(device) if device is not None else images.device
    model = _get_inception_feature_extractor(device=device, weights=weights, progress=progress)
    features = []
    for start in range(0, images.shape[0], int(batch_size)):
        batch = _prepare_inception_images(images[start : start + int(batch_size)].to(device), size=resize_size)
        out = model(batch)
        if hasattr(out, "logits"):
            out = out.logits
        features.append(out.flatten(1))
    return torch.cat(features, dim=0)


@torch.no_grad()
def extract_image_features(
    images,
    batch_size=32,
    device=None,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
):
    if feature_extractor is not None:
        features = feature_extractor(images)
        if features.ndim != 2:
            raise ValueError(f"Expected feature_extractor to return [N,D], got {tuple(features.shape)}")
        return features
    return inception_features(images, batch_size=batch_size, device=device, weights=weights, resize_size=resize_size)


@torch.no_grad()
def collect_image_features_stream(
    image_batches,
    n_samples=5000,
    batch_size=32,
    device=None,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
):
    """Collect image features from an iterable of image batches.

    This follows FID-style evaluation practice: each image batch is featurized and
    immediately discarded; only CPU features are retained. Pass n_samples=None
    or "all" to consume the iterable until exhaustion.
    """
    n_samples = _resolve_n_samples(n_samples)
    features = []
    seen = 0
    for images in image_batches:
        if n_samples is not None and seen >= n_samples:
            break
        if not torch.is_tensor(images):
            raise ValueError("image_batches must yield tensors")
        take = images.shape[0] if n_samples is None else min(images.shape[0], n_samples - seen)
        if take <= 0:
            continue
        current = images[:take]
        if device is not None:
            current = current.to(device)
        batch_features = extract_image_features(
            current,
            batch_size=batch_size,
            device=current.device,
            weights=weights,
            resize_size=resize_size,
            feature_extractor=feature_extractor,
        )
        features.append(batch_features.detach().float().cpu())
        seen += take
    if seen == 0:
        raise ValueError("Expected at least one sample, but collected none")
    if n_samples is not None and seen != n_samples:
        raise ValueError(f"Expected {n_samples} samples, but collected {seen}")
    return torch.cat(features, dim=0)


def load_or_compute_real_image_features(
    image_batches,
    n_samples=5000,
    cache_path=None,
    cache=True,
    batch_size=32,
    device=None,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
):
    n_samples = _resolve_n_samples(n_samples)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache and cache_path is not None and cache_path.exists():
        payload = _load_tensor(cache_path, map_location="cpu")
        features = payload.get("features", payload) if isinstance(payload, dict) else payload
        if torch.is_tensor(features) and features.ndim == 2 and (n_samples is None or features.shape[0] >= n_samples):
            return (features if n_samples is None else features[:n_samples]).float().cpu(), True

    features = collect_image_features_stream(
        image_batches,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        weights=weights,
        resize_size=resize_size,
        feature_extractor=feature_extractor,
    )
    if cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features,
                "n_samples": int(features.shape[0]),
                "weights": None if weights is None else str(weights),
                "resize_size": int(resize_size),
            },
            cache_path,
        )
    return features, False


def load_or_compute_real_mind_features(*args, **kwargs):
    return load_or_compute_real_image_features(*args, **kwargs)


def load_or_compute_real_fid_features(*args, **kwargs):
    return load_or_compute_real_image_features(*args, **kwargs)


@torch.no_grad()
def fid_image_metric(
    output,
    target,
    batch_size=32,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
):
    output_features = extract_image_features(
        output,
        batch_size=batch_size,
        device=output.device,
        weights=weights,
        resize_size=resize_size,
        feature_extractor=feature_extractor,
    )
    target_features = extract_image_features(
        target.to(output.device),
        batch_size=batch_size,
        device=output.device,
        weights=weights,
        resize_size=resize_size,
        feature_extractor=feature_extractor,
    )
    return frechet_inception_distance_torch(output_features, target_features)


@torch.no_grad()
def compute_fid(
    generated_image_batches,
    real_image_batches,
    n_samples=50000,
    batch_size=32,
    device=None,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
    real_features_cache_path=None,
    cache_real_features=True,
    return_details=False,
):
    """Compute FID from streamed generated and real image batches.

    Pass n_samples=None or "all" to use every image yielded by real_image_batches;
    generated_image_batches must then yield at least the same number of samples.
    Images are expected in the repository's normalized [-1, 1] tensor format.
    """
    n_samples = _resolve_n_samples(n_samples)
    real_features, cache_hit = load_or_compute_real_fid_features(
        real_image_batches,
        n_samples=n_samples,
        cache_path=real_features_cache_path,
        cache=cache_real_features,
        batch_size=batch_size,
        device=device,
        weights=weights,
        resize_size=resize_size,
        feature_extractor=feature_extractor,
    )
    effective_n_samples = int(real_features.shape[0])
    generated_features = collect_image_features_stream(
        generated_image_batches,
        n_samples=effective_n_samples,
        batch_size=batch_size,
        device=device,
        weights=weights,
        resize_size=resize_size,
        feature_extractor=feature_extractor,
    )
    fid = frechet_inception_distance_torch(generated_features, real_features)
    if return_details:
        return fid, {
            "n_samples": effective_n_samples,
            "feature_dim": int(generated_features.shape[1]),
            "real_feature_cache_hit": bool(cache_hit),
        }
    return fid


@torch.no_grad()
def mind_image_metric(
    output,
    target,
    rng_seed=0,
    n_projections=1000,
    batch_size=32,
    weights="DEFAULT",
    resize_size=299,
    feature_extractor=None,
):
    if output.shape[0] != target.shape[0]:
        raise ValueError(f"MIND requires equal sample counts, got {output.shape[0]} and {target.shape[0]}")
    if feature_extractor is None:
        output_features = inception_features(output, batch_size=batch_size, device=output.device, weights=weights, resize_size=resize_size)
        target_features = inception_features(target, batch_size=batch_size, device=output.device, weights=weights, resize_size=resize_size)
    else:
        output_features = feature_extractor(output)
        target_features = feature_extractor(target.to(output.device))
    return monge_inception_distance_torch(output_features, target_features, rng_seed=rng_seed, n_projections=n_projections)


def image_metrics(
    output,
    batch,
    lr_size,
    downsample_mode="nearest",
    compute_mind=False,
    mind_rng_seed=0,
    mind_n_projections=1000,
    mind_batch_size=32,
    mind_weights="DEFAULT",
    mind_resize_size=299,
    mind_feature_extractor=None,
):
    metrics = {}
    if "lr" in batch:
        metrics["output/downsample_l1"] = downsample_l1(output, batch["lr"].to(output.device), lr_size, downsample_mode)
    if "hr" in batch:
        target_hr = batch["hr"].to(output.device)
        metrics["output/rapsd_error"] = rapsd_error(output, target_hr)
        if compute_mind:
            metrics["output/mind"] = mind_image_metric(
                output,
                target_hr,
                rng_seed=mind_rng_seed,
                n_projections=mind_n_projections,
                batch_size=mind_batch_size,
                weights=mind_weights,
                resize_size=mind_resize_size,
                feature_extractor=mind_feature_extractor,
            )
    if "lr_up" in batch:
        metrics["roundtrip/l1"] = roundtrip_l1(output, batch["lr_up"].to(output.device))
    return metrics


def scalarize_metrics(metrics):
    return {key: float(value.detach().mean().cpu()) for key, value in metrics.items() if torch.is_tensor(value)}

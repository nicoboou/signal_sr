from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

from prehoc.models.classifier import ClassifierImage

from sr.utils.config import load_config as load_sr_config
from sr.utils.seed import seed_everything

from posthoc.parasite_detector.detector import detect_parasites
from posthoc.utils.config import resolve_path, load_yaml, apply_overrides, make_run_output_dir
from posthoc.data import build_eval_loader
from posthoc.utils.misc import safe_tag, move_to_device, choose_device
from posthoc.utils.metrics import binary_metrics, mask_iou, plot_curve, parasite_labels
from posthoc.utils.degradations import degrade_batch, to_unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/posthoc/outputs/sr_signal_sweep"
DEFAULT_NLM_MASK_ROOT = Path("/projects/compures/nicolas/cell_annotator/outputs")
PLOT_METRICS = ("accuracy", "precision", "recall", "f1")


def ids_from_batch(batch, start):
    if "sample_id" not in batch:
        return list(range(start, start + int(batch["hr"].shape[0])))
    ids = batch["sample_id"].detach().cpu().numpy().tolist() if torch.is_tensor(batch["sample_id"]) else list(batch["sample_id"])
    return [int(item) if isinstance(item, (int, np.integer, float, np.floating)) else item for item in ids]


def metadata_values(batch, key, count):
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, dict) or key not in metadata:
        return [None] * count
    value = metadata[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()[:count]
    if isinstance(value, (list, tuple)):
        return list(value)[:count]
    return [value] * count


def slice_value(value, keep, batch_size):
    if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
        return value[:keep]
    if isinstance(value, dict):
        return {key: slice_value(item, keep, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return value[:keep]
    return value


def resize_mask(mask, size):
    mask = torch.as_tensor(np.asarray(mask).astype(np.float32))[None, None]
    if tuple(mask.shape[-2:]) != (int(size), int(size)):
        mask = F.interpolate(mask, size=(int(size), int(size)), mode="nearest")
    return mask[0, 0].numpy() > 0.5


def true_masks(batch, data_name, mask_root, image_size):
    if data_name == "synthetic_microscopy":
        masks = batch["masks"]["parasite"].detach().cpu().numpy().astype(bool)
        return np.stack([resize_mask(mask, image_size) for mask in masks])
    if data_name == "nlm":
        paths = metadata_values(batch, "crop_path", int(batch["hr"].shape[0]))
        rel_paths = metadata_values(batch, "relative_path", int(batch["hr"].shape[0]))
        stems = [Path(path or rel_path).stem for path, rel_path in zip(paths, rel_paths)]
        masks = []
        for stem in stems:
            path = Path(mask_root) / "masks_npy" / f"{stem}.npy"
            masks.append(resize_mask(np.load(path) if path.exists() else np.zeros((image_size, image_size), dtype=bool), image_size))
        return np.stack(masks)
    raise ValueError(f"Unsupported posthoc dataset: {data_name}")


def build_sr(sr_cfg, method, checkpoint, device):
    from sr.inference import Inverter, Sampler
    from sr.inference.inverter import freeze_for_inference
    from sr.models import build_conditioner, build_denoiser
    from sr.objectives import Objective
    from sr.schedules import build_noise_scheduler
    from sr.utils.checkpoints import load_model_weights

    if sr_cfg.space != "pixel":
        raise ValueError("posthoc.run currently supports only pixel-space SR configs")
    if method in {"dps", "eps"}:
        sr_cfg.sampling.method = method
    if method == "spectral_sdedit_sr":
        sr_cfg.sampling.method = "ddim"
    denoiser = build_denoiser(sr_cfg.denoiser, data_channels=int(sr_cfg.data.channels)).to(device)
    conditioner = build_conditioner().to(device)
    noise_scheduler = build_noise_scheduler(sr_cfg.noise_scheduler)
    objective = Objective(sr_cfg.objective, noise_scheduler)
    sampler = Sampler(sr_cfg.sampling, objective, noise_scheduler, denoiser, conditioner, global_cfg=sr_cfg)
    inverter = (
        Inverter(sr_cfg.inversion, sr_cfg.sampling, objective, noise_scheduler, denoiser, conditioner, sampler) if method == "invert_sample" else None
    )
    if checkpoint:
        load_model_weights(checkpoint, denoiser=denoiser, conditioner=conditioner, map_location=device)
    freeze_for_inference(denoiser, conditioner)
    noise_scheduler.set_timesteps(sr_cfg.noise_scheduler.n_infer_steps, device=device)
    return denoiser, sampler, inverter, noise_scheduler


def run_sr(method, batch, sr_cfg, denoiser, sampler, inverter):
    if method == "dps":
        with torch.enable_grad():
            return sampler.dps_loop(torch.randn_like(batch["hr"]), batch, condition_domain="HR", conditioning_image=None)
    if method == "eps":
        with torch.no_grad():
            return sampler.ddim_loop(
                torch.randn_like(batch["hr"]),
                batch,
                condition_domain="HR",
                conditioning_image=None,
                clip_denoised=sr_cfg.sampling.get("clip_denoised", None),
            )
    if method == "spectral_sdedit_sr":
        from sr.inference import spectral_sdedit_sr

        native_cfg = sr_cfg.get("spectral_native_lr_sdedit", sr_cfg.sampling.get("spectral_native_lr_sdedit", {}))
        x_lr = batch["lr"] if native_cfg.get("enabled", False) else batch.get("lr", batch["lr_up"])
        with torch.no_grad():
            x_sr, _, _ = spectral_sdedit_sr(
                x_lr=x_lr,
                batch=batch,
                model=denoiser,
                sampler=sampler,
                hr_scheduler=sampler.noise_scheduler,
                scale_r=sr_cfg.get("scale", int(sr_cfg.image_size) / int(sr_cfg.lr_size)),
                cfg=sr_cfg,
            )
        return x_sr
    if method == "invert_sample":
        if inverter is None:
            raise ValueError("invert_sample requires an Inverter")
        with torch.no_grad():
            x_state, _ = inverter.invert_and_sample(batch["lr_up"], batch, conditioning_image=batch["lr_up"])
        return x_state
    raise ValueError(f"Unsupported SR method: {method}")


def classifier_scores(classifier, images_01):
    x = images_01.detach().cpu().float()
    if x.shape[1] == 1 and classifier.in_channels == 3:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3 and classifier.in_channels == 1:
        x = x.mean(dim=1, keepdim=True)
    probs = []
    classifier.model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], classifier.batch_size):
            logits = classifier.model(x[start : start + classifier.batch_size].to(classifier.device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def image_array(image_01):
    arr = image_01.detach().cpu().float().clamp(0, 1)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    return (arr[:3].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def synthetic_detect(image_01, cfg):
    gray = image_01.detach().cpu().float().mean(dim=0).numpy()
    cell = binary_fill_holes(gray > float(cfg.get("synthetic_cell_threshold", 0.12)))
    cell = morphology.remove_small_objects(cell.astype(bool), min_size=int(cfg.get("synthetic_min_cell_area", 256)))
    dark = cell & (gray < float(cfg.get("synthetic_dark_threshold", 0.35)))
    dark = morphology.remove_small_objects(dark, min_size=int(cfg.get("synthetic_min_area", 8)))
    labels = measure.label(dark)
    mask = np.zeros_like(dark, dtype=bool)
    for region in measure.regionprops(labels):
        keep = region.area <= int(cfg.get("synthetic_max_area", 220)) and region.eccentricity <= float(cfg.get("synthetic_max_eccentricity", 0.97))
        if keep:
            mask[labels == region.label] = True
    return mask, int(mask.any()), "parasitized" if mask.any() else "uninfected"


def detector_outputs(images_01, data_name, detector_cfg):
    masks, labels, statuses = [], [], []
    for image in images_01:
        if data_name == "synthetic_microscopy":
            mask, label, status = synthetic_detect(image, detector_cfg)
        else:
            result = detect_parasites(image_array(image), config=detector_cfg, return_debug=False)
            mask = result["parasite_mask"].astype(bool)
            label = int(result["inferred_label"] == "parasitized")
            status = result["segmentation_status"]
        masks.append(mask)
        labels.append(label)
        statuses.append(status)
    return masks, np.asarray(labels, dtype=int), statuses


def save_tensor_images(x, output_dir, names):
    from sr.utils.images import tensor_to_pil_image

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for image, name in zip(x.detach().cpu(), names):
        path = output_dir / f"{name}.png"
        tensor_to_pil_image(image).save(path)
        paths.append(str(path))
    return paths


def save_masks(masks, output_dir, names):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mask, name in zip(masks, names):
        path = output_dir / f"{name}.png"
        Image.fromarray((np.asarray(mask).astype(np.uint8) * 255)).save(path)
        paths.append(str(path))
    return paths


def run(config, config_path=None):
    seed_everything(int(config.get("seed", 0)))
    device = torch.device(choose_device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    base_output_dir = resolve_path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir, run_id = make_run_output_dir(base_output_dir, config.get("run_id"))
    config["run_id"] = run_id
    config["base_output_dir"] = str(base_output_dir)
    config["output_dir"] = str(output_dir)
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_dir}")

    with (output_dir / "argv.json").open("w", encoding="utf-8") as handle:
        json.dump({"argv": sys.argv}, handle, indent=2)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    if config_path is not None:
        shutil.copy2(config_path, output_dir / "source_config.yaml")

    sr_run_cfg = config["sr"]
    sr_cfg = load_sr_config(resolve_path(sr_run_cfg["config"]))
    method = sr_run_cfg.get("method", sr_cfg.sampling.get("method", "dps"))
    split = sr_run_cfg.get("split", "test")
    sr_checkpoint = resolve_path(sr_run_cfg["checkpoint"]) if sr_run_cfg.get("checkpoint") else None
    classifier = ClassifierImage.from_checkpoint(resolve_path(config["classifier"]["checkpoint"]), device=str(device))
    threshold = float(config.get("classifier", {}).get("threshold", classifier.checkpoint_metadata.get("threshold", 0.5)))
    loader = build_eval_loader(sr_cfg, config, split)
    denoiser, sampler, inverter, _ = build_sr(sr_cfg, method, sr_checkpoint, device)
    base_lr_size = int(sr_cfg.get("lr_size", sr_cfg.data.get("lr_size", max(1, int(sr_cfg.image_size) // int(sr_cfg.get("scale", 1))))))

    rows, metric_rows = [], []
    max_samples = config.get("eval", {}).get("max_samples")
    save_images = bool(config.get("eval", {}).get("save_images", True))
    data_name = sr_cfg.data.name
    mask_root = resolve_path(
        config.get("data", {}).get("nlm_mask_outputs_root", config.get("data", {}).get("mask_outputs_root", DEFAULT_NLM_MASK_ROOT))
    )
    detector_cfg = config.get("detector", {}).get("config", {}) or {}

    for level in config["degradation"]["levels"]:
        print(f"Posthoc {config['degradation']['type']} level={level}")
        level_tag = f"level_{safe_tag(float(level))}"
        level_dir = output_dir / "images" / level_tag
        y_true_all, clf_prob_all, det_pred_all, ious, ious_positive = [], [], [], [], []
        seen = 0
        for batch_index, raw_batch in enumerate(loader):
            if max_samples is not None and seen >= int(max_samples):
                break
            raw_batch = move_to_device(raw_batch, device)
            sample_ids = ids_from_batch(raw_batch, start=seen)
            if max_samples is not None:
                keep = min(int(max_samples) - seen, int(raw_batch["hr"].shape[0]))
                raw_batch = {key: slice_value(value, keep, len(sample_ids)) for key, value in raw_batch.items()}
                sample_ids = sample_ids[:keep]
            degradation_cfg = dict(config.get("degradation", {}))
            degradation_cfg.setdefault("lr_size", base_lr_size)
            lr, lr_up = degrade_batch(raw_batch["hr"], level, degradation_cfg, sr_cfg, sample_ids)
            sr_cfg.lr_size = int(lr.shape[-1])
            sr_cfg.scale = float(raw_batch["hr"].shape[-1]) / float(lr.shape[-1])
            batch = dict(raw_batch)
            batch.update({"lr": lr, "lr_up": lr_up, "image": lr_up})
            sr = run_sr(method, batch, sr_cfg, denoiser, sampler, inverter).detach().clamp(-1, 1)
            sr01 = to_unit(sr)
            labels = parasite_labels(raw_batch)
            masks_true = true_masks(raw_batch, data_name, mask_root, int(sr.shape[-1]))
            clf_prob = classifier_scores(classifier, sr01)
            det_masks, det_pred, det_status = detector_outputs(sr01, data_name, detector_cfg)
            names = [f"{seen + i:06d}_{safe_tag(sample_id)}" for i, sample_id in enumerate(sample_ids)]

            sr_paths = lr_paths = lrup_paths = hr_paths = true_mask_paths = pred_mask_paths = [None] * len(names)
            if save_images:
                sr_paths = save_tensor_images(sr, level_dir / "sr", names)
                lr_paths = save_tensor_images(lr, level_dir / "lr", names)
                lrup_paths = save_tensor_images(lr_up, level_dir / "lr_up", names)
                hr_paths = save_tensor_images(raw_batch["hr"], level_dir / "hr_gt", names)
                true_mask_paths = save_masks(masks_true, level_dir / "true_masks", names)
                pred_mask_paths = save_masks(det_masks, level_dir / "detector_masks", names)

            for i, sample_id in enumerate(sample_ids):
                iou = mask_iou(det_masks[i], masks_true[i])
                ious.append(iou)
                if masks_true[i].any():
                    ious_positive.append(iou)
                rows.append(
                    {
                        "degradation_type": config["degradation"]["type"],
                        "level": float(level),
                        "sample_id": sample_id,
                        "true_label": int(labels[i]),
                        "classifier_probability": float(clf_prob[i]),
                        "classifier_prediction": int(clf_prob[i] >= threshold),
                        "detector_prediction": int(det_pred[i]),
                        "detector_status": det_status[i],
                        "detector_iou": float(iou),
                        "true_mask_area": int(np.asarray(masks_true[i]).sum()),
                        "detector_mask_area": int(np.asarray(det_masks[i]).sum()),
                        "sr_path": sr_paths[i],
                        "lr_path": lr_paths[i],
                        "lr_up_path": lrup_paths[i],
                        "hr_path": hr_paths[i],
                        "true_mask_path": true_mask_paths[i],
                        "detector_mask_path": pred_mask_paths[i],
                    }
                )
            y_true_all.extend(labels.tolist())
            clf_prob_all.extend(clf_prob.tolist())
            det_pred_all.extend(det_pred.tolist())
            seen += len(sample_ids)

        clf_metrics = binary_metrics(y_true_all, clf_prob_all, threshold=threshold)
        det_metrics = binary_metrics(y_true_all, det_pred_all, threshold=0.5)
        metric_rows.append(
            {
                "degradation_type": config["degradation"]["type"],
                "level": float(level),
                "n_samples": int(len(y_true_all)),
                "mean_iou": float(np.nanmean(ious_positive)) if ious_positive else np.nan,
                "mean_iou_all": float(np.mean(ious)) if ious else np.nan,
                **{f"classifier_{key}": value for key, value in clf_metrics.items()},
                **{f"detector_{key}": value for key, value in det_metrics.items()},
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / "sample_results.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)

    metrics = pd.DataFrame(metric_rows).sort_values("level")
    samples = pd.DataFrame(rows)
    samples.to_csv(output_dir / "sample_results.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    plot_curve(metrics, ["mean_iou"], output_dir / "mean_iou.pdf", ylabel="Mean IoU")
    plot_curve(metrics, [f"detector_{metric}" for metric in PLOT_METRICS], output_dir / "detector_metrics.pdf")
    plot_curve(metrics, [f"classifier_{metric}" for metric in PLOT_METRICS], output_dir / "classifier_metrics.pdf")
    print(f"Wrote {output_dir / 'sample_results.csv'}")
    print(f"Wrote {output_dir / 'metrics.csv'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run posthoc SR signal analysis over degradation levels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--levels", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sr-checkpoint", default=None)
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    run(apply_overrides(load_yaml(config_path), args), config_path=config_path)


if __name__ == "__main__":
    main()

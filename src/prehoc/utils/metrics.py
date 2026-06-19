"""Small metrics for binary tasks, calibration, masks, and local signal."""

import numpy as np
import pandas as pd
from skimage import morphology


def expected_calibration_error(y_true, positive_probability, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(positive_probability).astype(float)
    pred = (prob >= 0.5).astype(int)
    confidence = np.maximum(prob, 1 - prob)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, int(n_bins) + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return float(ece)


def calibration_table(y_true, positive_probability, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(positive_probability).astype(float)
    pred = (prob >= 0.5).astype(int)
    confidence = np.maximum(prob, 1 - prob)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, int(n_bins) + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "bin_center": 0.5 * (low + high),
                "count": int(in_bin.sum()),
                "accuracy": float(correct[in_bin].mean()) if in_bin.any() else np.nan,
                "confidence": float(confidence[in_bin].mean()) if in_bin.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def roc_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(score)
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true, positive_probability, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(positive_probability).astype(float)
    pred = (prob >= 0.5).astype(int)
    confidence = np.maximum(prob, 1 - prob)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "auc": roc_auc(y_true, prob),
        "mean_confidence": float(confidence.mean()),
        "ece": expected_calibration_error(y_true, prob, n_bins=n_bins),
        "positive_rate": float(pred.mean()),
        "n": int(len(y_true)),
    }


def mask_iou(pred, target):
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    union = np.logical_or(pred, target).sum()
    return float(np.logical_and(pred, target).sum() / (union + 1e-8))


def local_snr(image, object_mask, cell_mask, margin=8):
    object_mask = np.asarray(object_mask).astype(bool)
    cell_mask = np.asarray(cell_mask).astype(bool)
    if object_mask.sum() == 0:
        return np.nan
    background = morphology.dilation(object_mask, morphology.disk(int(margin))) & cell_mask
    background = background & ~morphology.dilation(object_mask, morphology.disk(2))
    if background.sum() == 0:
        return np.nan
    contrast = float(image[background].mean() - image[object_mask].mean())
    return contrast / float(image[background].std() + 1e-6)

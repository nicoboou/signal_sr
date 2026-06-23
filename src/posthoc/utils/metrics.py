import numpy as np
import torch
import matplotlib.pyplot as plt


def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mask_iou(pred, true):
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    union = np.logical_or(pred, true).sum()
    return float(np.logical_and(pred, true).sum() / union) if union else 1.0


def parasite_labels(batch):
    labels = batch.get("labels")
    if labels is None:
        raise ValueError("Posthoc evaluation requires labels in dataloader batches")
    labels = labels.detach().cpu() if torch.is_tensor(labels) else torch.as_tensor(labels)
    return labels[:, 0].numpy().astype(int) if labels.ndim > 1 else labels.numpy().astype(int)


def plot_curve(summary, y_columns, output_path, ylabel="Score"):
    summary = summary.sort_values("level")
    x = summary["level"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for column in y_columns:
        if column in summary:
            ax.plot(x, summary[column].to_numpy(dtype=float), marker="o", linewidth=1.8, markersize=4.5, label=column)
    ax.set_xlabel("Degradation level")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    if len(x) > 2 and np.all(x > 0):
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def make_image_grid(tensors, nrow=4):
    tensors = tensors.detach().cpu()
    n = tensors.shape[0]
    ncol = (n + nrow - 1) // nrow
    c, h, w = tensors.shape[1], tensors.shape[2], tensors.shape[3]
    grid = torch.zeros(c, nrow * h, ncol * w)
    for i in range(n):
        r, col = divmod(i, ncol)
        grid[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = tensors[i]
    return grid


def to_display_tensor(x):
    x = x.detach().cpu().float()
    x = x.nan_to_num().clamp(-1, 1)
    return (x + 1.0) * 0.5


def tensor_to_uint8_image(image):
    image = to_display_tensor(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(image.shape)}")
    if image.shape[0] == 1:
        return (image[0].numpy() * 255.0).round().clip(0, 255).astype(np.uint8), "L"
    return (image[:3].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8), "RGB"


def tensor_to_pil_image(image):
    arr, mode = tensor_to_uint8_image(image)
    return Image.fromarray(arr, mode=mode)


def save_images(x, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu()
    paths = []
    for i, image in enumerate(x):
        pil = tensor_to_pil_image(image)
        path = out_dir / f"{prefix}_{i:03d}.png"
        pil.save(path)
        paths.append(path)
    return paths

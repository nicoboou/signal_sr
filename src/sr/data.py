from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2


@lru_cache(maxsize=None)
def make_rgb_image_transform(image_size: int):
    return v2.Compose(
        [
            v2.Resize((int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def resize_nearest(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.ndim == 3:
        return F.interpolate(x[None], size=(size, size), mode="nearest")[0]
    if x.ndim == 4:
        return F.interpolate(x, size=(size, size), mode="nearest")
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got shape {tuple(x.shape)}")


def make_lr_pair(hr: torch.Tensor, image_size: int, scale: int):
    lr_size = image_size // scale
    lr = resize_nearest(hr, lr_size)
    lr_up = resize_nearest(lr, image_size)
    return lr, lr_up


def load_rgb_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
    return make_rgb_image_transform(image_size)(image)


class NLMSRDataset(Dataset):
    def __init__(
        self,
        root,
        split_dir=None,
        split_csv=None,
        split="train",
        image_size=224,
        scale=2,
        return_pair=False,
        return_labels=False,
        return_metadata=False,
    ):
        if split_csv is None:
            if split_dir is None:
                raise ValueError("NLMSRDataset requires split_dir or split_csv")
            split_csv = Path(split_dir) / f"{split}.csv"

        split_csv = Path(split_csv)
        if not split_csv.is_file():
            raise FileNotFoundError(f"Missing NLM split CSV: {split_csv}")

        table = pd.read_csv(split_csv)
        if "split" in table.columns:
            table = table[table["split"] == split]
        table = table.reset_index(drop=True)
        if table.empty:
            raise ValueError(f"No rows found for split {split!r} in {split_csv}")

        required = {"domain", "label"}
        if not ({"crop_path", "relative_path"} & set(table.columns)):
            required.add("crop_path or relative_path")
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Missing required NLM split columns: {sorted(missing)}")

        self.table = table
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.scale = int(scale)
        self.lr_size = self.image_size // self.scale
        self.return_pair = bool(return_pair)
        self.return_labels = bool(return_labels)
        self.return_metadata = bool(return_metadata)

        if self.image_size % self.scale != 0:
            raise ValueError("image_size must be divisible by scale")

    def __len__(self):
        return len(self.table)

    def _path_from_row(self, row):
        path = Path(row["crop_path"] if "crop_path" in row and pd.notna(row["crop_path"]) else row["relative_path"])
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        domain = int(row["domain"])
        if domain not in (0, 1):
            raise ValueError(f"Expected domain 0 or 1, got {domain}")

        sample_id = int(row["sample_id"]) if "sample_id" in row and pd.notna(row["sample_id"]) else int(idx)
        hr = load_rgb_image(self._path_from_row(row), self.image_size)
        lr, lr_up = make_lr_pair(hr, self.image_size, self.scale)
        image = lr_up if domain == 0 else hr
        out = {
            "image": image,
            "domain": torch.tensor(domain, dtype=torch.long),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "split": self.split,
        }
        if self.return_pair:
            out.update({"lr": lr, "lr_up": lr_up, "hr": hr})
        if self.return_labels:
            out["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.return_metadata:
            out["metadata"] = row.to_dict()
        return out


class SyntheticMicroscopyDataset(Dataset):
    def __init__(
        self,
        split_dir,
        split="train",
        image_size=128,
        scale=16,
        channels=1,
        return_pair=False,
        return_labels=True,
        return_masks=False,
        return_metadata=False,
    ):
        split_dir = Path(split_dir)
        self.images = np.load(split_dir / f"{split}_images.npy")
        self.domains = np.load(split_dir / f"{split}_domains.npy")
        self.sample_ids = np.load(split_dir / f"{split}_sample_ids.npy")
        self.labels = np.load(split_dir / f"{split}_labels.npy") if return_labels else None

        self.cell_masks = None
        self.parasite_masks = None
        self.filament_masks = None
        if return_masks:
            self.cell_masks = np.load(split_dir / f"{split}_cell_masks.npy")
            self.parasite_masks = np.load(split_dir / f"{split}_parasite_masks.npy")
            self.filament_masks = np.load(split_dir / f"{split}_filament_masks.npy")

        self.metadata = None
        if return_metadata:
            self.metadata = np.load(split_dir / f"{split}_metadata.npy", allow_pickle=True)

        self.split = split
        self.image_size = int(image_size)
        self.scale = int(scale)
        self.lr_size = self.image_size // self.scale
        self.channels = int(channels)
        self.return_pair = bool(return_pair)
        self.return_labels = bool(return_labels)
        self.return_masks = bool(return_masks)
        self.return_metadata = bool(return_metadata)

        if not (len(self.images) == len(self.domains) == len(self.sample_ids)):
            raise ValueError("Synthetic split arrays must have matching lengths")
        if self.image_size % self.scale != 0:
            raise ValueError("image_size must be divisible by scale")
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        domain = int(self.domains[idx])
        sample_id = int(self.sample_ids[idx])

        hr = torch.from_numpy(self.images[idx]).float()[None]
        if self.channels == 3:
            hr = hr.repeat(3, 1, 1)
        hr = hr * 2.0 - 1.0

        lr, lr_up = make_lr_pair(hr, self.image_size, self.scale)
        image = hr if domain == 1 else lr_up

        out = {
            "image": image,
            "domain": torch.tensor(domain, dtype=torch.long),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "split": self.split,
        }
        if self.return_pair:
            out.update({"lr": lr, "lr_up": lr_up, "hr": hr})

        if self.return_labels:
            out["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_masks:
            out["masks"] = {
                "cell": torch.from_numpy(self.cell_masks[idx]).bool(),
                "parasite": torch.from_numpy(self.parasite_masks[idx]).bool(),
                "filament": torch.from_numpy(self.filament_masks[idx]).bool(),
            }

        if self.return_metadata:
            item = self.metadata[idx]
            out["metadata"] = item.item() if hasattr(item, "item") else item

        return out


def build_dataset(cfg, split="train"):
    name = cfg.name
    if name == "synthetic_microscopy":
        return SyntheticMicroscopyDataset(
            split_dir=cfg.split_dir,
            split=split,
            image_size=cfg.image_size,
            scale=cfg.scale,
            channels=cfg.channels,
            return_pair=cfg.get("return_pair", False),
            return_labels=cfg.get("return_labels", True),
            return_masks=cfg.get("return_masks", False),
            return_metadata=cfg.get("return_metadata", False),
        )
    if name == "nlm":
        return NLMSRDataset(
            root=cfg.root,
            split_dir=cfg.get("split_dir", None),
            split_csv=cfg.get("split_csv", None),
            split=split,
            image_size=cfg.image_size,
            scale=cfg.scale,
            return_pair=cfg.get("return_pair", False),
            return_labels=cfg.get("return_labels", False),
            return_metadata=cfg.get("return_metadata", False),
        )
    raise ValueError(f"Unknown dataset: {name}")


def build_dataloader(cfg, split="train", shuffle=True, batch_size=None, num_workers=0):
    dataset = build_dataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.get("batch_size", 1),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

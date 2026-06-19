import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class NLMDataset(Dataset):
    """Dataset for preprocessed NLM cell crops stored by class."""

    classes = ("uninfected", "parasitized")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    def __init__(self, root="/projects/compures/datasets/NLM/crops", transform=None, split_csv=None):
        self.root = Path(root)
        self.transform = transform if transform is not None else pil_to_tensor
        self.samples = self._load_split_csv(split_csv) if split_csv is not None else self._load_folder_samples()

        if not self.samples:
            raise ValueError(f"No PNG crops found for NLMDataset under {self.root}")

    def _load_folder_samples(self):
        samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing NLM crop directory: {class_dir}")
            samples.extend((path, label) for path in sorted(class_dir.glob("*.png")))
        return samples

    def _load_split_csv(self, split_csv):
        split_csv = Path(split_csv)
        if not split_csv.is_file():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        samples = []
        with split_csv.open(newline="") as file:
            for row in csv.DictReader(file):
                path = Path(row.get("crop_path") or row["relative_path"])
                if not path.is_absolute():
                    path = self.root / path
                label = int(row["label"]) if row.get("label") else self.class_to_idx[row["label_name"]]
                samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if getattr(self.transform, "needs_index", False):
            image = self.transform(image, index=index)
        else:
            image = self.transform(image)
        return image, label

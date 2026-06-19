"""Small image and feature classifiers with train/eval methods."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..utils.metrics import binary_metrics


class SmallCNN(nn.Module):
    def __init__(self, width=24, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * width, 2)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class ClassifierImage:
    def __init__(
        self, device="cuda", epochs=5, batch_size=64, lr=1e-3, weight_decay=1e-4, width=24, backbone="small_cnn", pretrained=True, in_channels=3
    ):
        self.device = torch.device(device)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.width = int(width)
        self.backbone = backbone
        self.pretrained = bool(pretrained)
        self.in_channels = int(in_channels)
        self.model = self._new_model().to(self.device)

    def _new_model(self):
        if self.backbone == "resnet18":
            return make_resnet18(pretrained=self.pretrained, in_channels=self.in_channels)
        return SmallCNN(width=self.width, in_channels=self.in_channels)

    def _tensor(self, images):
        array = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
        if array.ndim == 3:
            array = array[:, None]
        return torch.from_numpy(array.astype("float32"))

    def _new_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _loader(self, images, labels, indices, shuffle=False):
        x = self._tensor(images[indices])
        y = torch.as_tensor(np.asarray(labels)[indices]).long()
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=shuffle)

    def train(self, images, labels, train_idx, val_idx, seed=0):
        torch.manual_seed(int(seed))
        self.model = self._new_model().to(self.device)
        optimizer = self._new_optimizer()
        train_loader = self._loader(images, labels, train_idx, shuffle=True)

        best_accuracy = -1.0
        best_state = deepcopy(self.model.state_dict())

        for _ in range(self.epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(self.model(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            accuracy = self.eval(images, labels, val_idx)["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        return self

    def fit_loader(self, train_loader, val_loader, seed=0):
        torch.manual_seed(int(seed))
        self.model = self._new_model().to(self.device)
        optimizer = self._new_optimizer()
        best_accuracy = -1.0
        best_state = deepcopy(self.model.state_dict())

        for _ in tqdm(range(self.epochs), desc="Epochs", leave=False):
            self.model.train()
            for x, y in tqdm(train_loader, desc="Batch", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(self.model(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            accuracy = self.eval_loader(val_loader)["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        return self

    def probabilities_loader(self, loader):
        probs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                logits = self.model(x.to(self.device))
                probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                labels.append(y.cpu().numpy())
        return np.concatenate(probs), np.concatenate(labels).astype(int)

    def eval_loader(self, loader):
        prob, labels = self.probabilities_loader(loader)
        return binary_metrics(labels, prob)

    def probabilities(self, images, indices):
        loader = DataLoader(TensorDataset(self._tensor(images[indices])), batch_size=self.batch_size)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for (x,) in loader:
                logits = self.model(x.to(self.device))
                probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)

    def predictions(self, images, indices):
        return (self.probabilities(images, indices) >= 0.5).astype(int)

    def eval(self, images, labels, indices):
        prob = self.probabilities(images, indices)
        return binary_metrics(np.asarray(labels)[indices], prob)

    def plot(self, results, x, y="accuracy", title=None):
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        for mode, sub in results.groupby("mode"):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o", label=mode)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or y)
        ax.legend()
        return fig


def make_resnet18(pretrained=True, in_channels=3):
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_channels = int(in_channels)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        if pretrained:
            weight = old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            model.conv1.weight.data.copy_(weight / max(1, in_channels))
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

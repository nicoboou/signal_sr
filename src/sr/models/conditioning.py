from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Conditioning:
    class_labels: torch.Tensor | None = None
    conditioning_image: torch.Tensor | None = None


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, conditioning_image=None):
        domain = batch.get("domain", None)
        class_labels = None
        if domain is not None:
            if torch.is_tensor(domain):
                class_labels = domain.long()
            else:
                class_labels = torch.as_tensor(domain).long()
        return Conditioning(class_labels=class_labels, conditioning_image=conditioning_image)


def build_conditioner():
    return Conditioner()

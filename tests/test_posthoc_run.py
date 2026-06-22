import numpy as np
import torch

from posthoc.run import binary_metrics, degrade_batch, mask_iou
from prehoc.models.classifier import ClassifierImage


def test_classifier_checkpoint_roundtrip(tmp_path):
    classifier = ClassifierImage(device="cpu", width=2, in_channels=1, pretrained=False)
    path = classifier.save_checkpoint(tmp_path / "classifier.pt", metadata={"threshold": 0.4})
    loaded = ClassifierImage.from_checkpoint(path, device="cpu")

    assert loaded.in_channels == 1
    assert loaded.checkpoint_metadata["threshold"] == 0.4
    for key, value in classifier.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])


def test_bilinear_degradation_uses_level_as_scale_factor():
    hr = torch.linspace(-1, 1, 16, dtype=torch.float32).reshape(1, 1, 4, 4)
    lr, lr_up = degrade_batch(hr, level=2, cfg={"type": "bilinear"}, sr_cfg={}, sample_ids=[0])

    assert lr.shape == (1, 1, 2, 2)
    assert lr_up.shape == hr.shape
    assert float(lr.min()) >= -1.0
    assert float(lr.max()) <= 1.0


def test_binary_metrics_and_mask_iou():
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5

    pred = np.array([[1, 0], [1, 0]], dtype=bool)
    true = np.array([[1, 1], [0, 0]], dtype=bool)
    assert mask_iou(pred, true) == 1 / 3

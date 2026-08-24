#!/usr/bin/env python3
"""CPU contracts for the opt-in official UNI2-H input normalization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.dataset_input import (  # noqa: E402
    LEGACY_HISTOLOGY_NORMALIZATION,
    UNI2_HISTOLOGY_NORMALIZATION,
    normalize_histology_tensor_after_joint_transform,
    normalize_uni2_histology_tensor,
    resolve_foundation_model_input_normalization,
    validate_foundation_model_input_normalization,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_uni2_known_rgb_values():
    rgb = torch.tensor(
        [
            [[0.0, 255.0, 123.0]],
            [[0.0, 255.0, 45.0]],
            [[0.0, 255.0, 200.0]],
        ]
    )
    actual = normalize_uni2_histology_tensor(rgb)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (rgb / 255.0 - mean) / std
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


def test_joint_transform_preserves_mask_ids_and_rgb():
    nuclei = np.array([[0, 4], [7, 7]], dtype=np.uint32)
    cell_types = np.array([[0, 2], [3, 3]], dtype=np.uint32)
    histology = np.array(
        [
            [[0, 0, 0], [123, 45, 200]],
            [[255, 255, 255], [10, 20, 30]],
        ],
        dtype=np.float32,
    )
    joint = np.concatenate(
        [nuclei[..., None], cell_types[..., None], histology], axis=-1
    )
    assert joint.dtype == np.float64
    transform = v2.Compose(
        [v2.ToImage(), v2.RandomHorizontalFlip(p=1.0), v2.ToDtype(torch.float32)]
    )
    transformed = transform(joint)

    nuclei_after = transformed[0].long()
    types_after = transformed[1].long()
    expected_nuclei = torch.from_numpy(np.fliplr(nuclei).copy()).long()
    expected_types = torch.from_numpy(np.fliplr(cell_types).copy()).long()
    assert torch.equal(nuclei_after, expected_nuclei)
    assert torch.equal(types_after, expected_types)
    assert set(torch.unique(nuclei_after).tolist()) == {0, 4, 7}

    actual_hist = normalize_histology_tensor_after_joint_transform(
        transformed[2:], UNI2_HISTOLOGY_NORMALIZATION
    )
    flipped_hist = torch.from_numpy(np.fliplr(histology).copy()).permute(2, 0, 1)
    expected_hist = normalize_uni2_histology_tensor(flipped_hist)
    torch.testing.assert_close(actual_hist, expected_hist, rtol=0.0, atol=1e-7)


def test_legacy_flag_is_exact_identity():
    assert (
        resolve_foundation_model_input_normalization(ns())
        == LEGACY_HISTOLOGY_NORMALIZATION
    )
    histology = torch.randn(3, 5, 7)
    actual = normalize_histology_tensor_after_joint_transform(
        histology, LEGACY_HISTOLOGY_NORMALIZATION
    )
    assert actual is histology
    assert torch.equal(actual, histology)


def test_foundation_guard_and_invalid_mode():
    data_cfg = ns(foundation_model_input_normalization=UNI2_HISTOLOGY_NORMALIZATION)
    try:
        validate_foundation_model_input_normalization(
            data_cfg, ns(foundation_model=ns(enabled=False))
        )
    except ValueError as exc:
        assert "foundation_model.enabled=true" in str(exc)
    else:
        raise AssertionError("UNI2 normalization must reject a disabled foundation model")

    assert (
        validate_foundation_model_input_normalization(
            data_cfg, ns(foundation_model=ns(enabled=True))
        )
        == UNI2_HISTOLOGY_NORMALIZATION
    )
    try:
        resolve_foundation_model_input_normalization(
            ns(foundation_model_input_normalization="not-a-mode")
        )
    except ValueError as exc:
        assert "foundation_model_input_normalization" in str(exc)
    else:
        raise AssertionError("Invalid normalization mode must be rejected")


def test_full_config_uses_uni2_input_contract():
    candidate = json.loads(
        (ROOT / "configs/breast2_ablation9/full.json").read_text()
    )
    assert (
        candidate["data"]["foundation_model_input_normalization"]
        == UNI2_HISTOLOGY_NORMALIZATION
    )
    assert candidate["data"]["num_workers"] == 24
    assert candidate["data"]["pin_memory"] is False
    assert candidate["model"]["foundation_model"] == {
        "enabled": True,
        "pretrained": True,
        "train_adapter": False,
    }
    assert candidate["model"]["ecrm"]["ablation_off"] is False
    assert candidate["model"]["hurdle"]["enabled"] is True
    assert candidate["model"]["vq_patch"]["enabled"] is True


def run():
    tests = [
        test_uni2_known_rgb_values,
        test_joint_transform_preserves_mask_ids_and_rgb,
        test_legacy_flag_is_exact_identity,
        test_foundation_guard_and_invalid_mode,
        test_full_config_uses_uni2_input_contract,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} UNI2-H input normalization contract groups")


if __name__ == "__main__":
    run()

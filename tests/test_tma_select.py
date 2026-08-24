#!/usr/bin/env python3
"""Golden contracts for C from ``HEAD:train_tma_select.py``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.tma_select as tma_select  # noqa: E402


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeCanonicalUnion(Dataset):
    """Canonical 9-field sample with original-C observable state."""

    def __init__(self, *, mode="train", slide_idx=7):
        self.mode = mode
        self.slide_idx = slide_idx
        self.hsize = 2
        self.wsize = 2
        self.coords_starts = [(0, 0), (0, 4), (4, 0), (4, 4), (8, 8)]
        self.n_patches = len(self.coords_starts)
        self.stain_aug = True
        self.tfs = object()
        self.tfs_test = object()
        self.patch_weights = [1.0] * self.n_patches
        self.df_ct = {"whole_slide": True}
        self.all_intersect = [1, 2, 3, 4, 5]
        self.read_observations = []
        self.last_multipliers = None

    def __len__(self):
        return self.n_patches

    def __getitem__(self, index):
        self.read_observations.append(
            (bool(self.stain_aug), self.tfs is self.tfs_test)
        )
        max_cells = 4
        n_genes = 3
        nuclei = torch.tensor([[1, 1], [2, 2]], dtype=torch.long)
        types = torch.tensor([[1, 1], [2, 2]], dtype=torch.long)
        hist = torch.full((3, 2, 2), float(index), dtype=torch.float32)
        expr = torch.zeros(max_cells, n_genes, dtype=torch.float32)
        expr[:2] = float(index + 1)
        n_cells = torch.tensor([2], dtype=torch.long)
        cell_types = torch.tensor([index % 2, (index + 1) % 2, 0, 0])
        patch_ids = torch.tensor([1, 2, 0, 0], dtype=torch.long)
        expr_mask = torch.ones_like(expr)
        slide_id = torch.tensor(self.slide_idx, dtype=torch.long)
        return (
            nuclei,
            types,
            hist,
            expr,
            n_cells,
            cell_types,
            patch_ids,
            expr_mask,
            slide_id,
        )

    def _compute_patch_sampling_weights(self, boost_factor):
        return [
            float(boost_factor) * float(index + 1)
            for index in range(len(self.coords_starts))
        ]

    def set_immune_sampling_multipliers(self, multipliers):
        self.last_multipliers = dict(multipliers)

    def refresh_patch_sampling_weights(self, boost_factor):
        self.patch_weights = self._compute_patch_sampling_weights(boost_factor)
        return self.patch_weights


class DummyVQModel(nn.Module):
    def __init__(self, *, expose_idx=True, fail=False):
        super().__init__()
        self.child = nn.Dropout(0.5)
        self.use_ecrm = False
        self.expose_idx = expose_idx
        self.fail = fail
        self.calls = 0
        self.last_aux_losses = {}

    def forward(
        self,
        batch_he_img,
        _batch_nuclei,
        _batch_n_cells,
        _expr_ref_batch,
        _batch_ct,
        _batch_expr,
        **_kwargs,
    ):
        self.calls += 1
        torch.rand(1)
        if self.fail:
            raise RuntimeError("selector fixture failure")
        patch_value = batch_he_img[:, 0, 0, 0].long()
        self.last_aux_losses = {
            "vq_patch_err": 5.0 - patch_value.float(),
        }
        if self.expose_idx:
            self.last_aux_losses["vq_patch_idx"] = patch_value.remainder(2)
        return ()


def selector_opts(**data_overrides):
    data = {
        "punch_select_enabled": True,
        "punch_filter_splits": "train",
        "roi_size_um": 8.0,
        "pixel_size_um": 1.0,
        "punch_num_workers": 0,
        "pin_memory": False,
        "punch_stage2_min_patches_for_mol": 2,
    }
    data.update(data_overrides)
    return ns(
        data=ns(**data),
        training=ns(batch_size=2, seed=17, batch_sampler_seed=99),
        model=ns(vq_patch=ns(n_codes=2)),
    )


def original_statistics_fixture():
    return {
        "coords": np.array(
            [[1, 1], [1, 5], [5, 1], [5, 5], [9, 9]],
            dtype=np.float32,
        ),
        "n_cells": np.array([2, 3, 4, 2, 1], dtype=np.int64),
        "expr_sum": np.array([2, 3, 4, 2, 1], dtype=np.float32),
        "vq_err": np.array([5, 4, 3, 2, 1], dtype=np.float32),
        "vq_idx": np.array([0, 1, 1, 0, 1], dtype=np.int64),
        "ct_counts": np.array(
            [[2, 0], [0, 3], [1, 3], [2, 0], [0, 1]],
            dtype=np.int64,
        ),
        "expr_mean": np.array(
            [[1, 0, 0], [0, 1, 0], [0, 2, 1], [1, 0, 1], [0, 1, 2]],
            dtype=np.float32,
        ),
    }


def test_golden_original_two_stage_formula_and_defaults():
    center, metadata = tma_select._choose_punch(
        original_statistics_fixture(),
        selector_opts(),
        ["A", "B"],
        7,
        8.0,
    )

    np.testing.assert_array_equal(center, np.array([5.0, 5.0], np.float32))
    expected = {
        "stage1_score": 0.8636926042917028,
        "balance": 0.9805806875228882,
        "coverage": 1.0,
        "size": 0.8807970779778823,
        "n_qc_patches": 5,
        "n_total_patches": 5,
        "stage2_score": 0.5929290602098665,
        "stage2_ct": 0.9966393709182739,
        "stage2_cells": 0.8807970779778823,
        "stage2_mol": 0.6754431965620077,
        "stage2_cells_roi": 12,
    }
    assert metadata.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            np.testing.assert_allclose(metadata[key], value, rtol=1e-7, atol=1e-8)
        else:
            assert metadata[key] == value


def test_golden_original_min_vq_fallbacks_and_fail_open_none():
    statistics = original_statistics_fixture()
    statistics["ct_counts"] = None
    statistics["expr_mean"] = None

    no_qc_opts = selector_opts(punch_qc_min_cells=99)
    center, metadata = tma_select._choose_punch(
        statistics, no_qc_opts, ["A", "B"], 7, 8.0
    )
    np.testing.assert_array_equal(center, np.array([9.0, 9.0], np.float32))
    assert metadata == {"fallback": "min_vq_err_no_qc"}

    statistics["vq_idx"] = None
    center, metadata = tma_select._choose_punch(
        statistics, selector_opts(), ["A", "B"], 7, 8.0
    )
    np.testing.assert_array_equal(center, np.array([9.0, 9.0], np.float32))
    assert metadata == {"fallback": "min_vq_err_no_idx"}

    statistics["vq_err"] = None
    assert (
        tma_select._choose_punch(
            statistics, selector_opts(), ["A", "B"], 7, 8.0
        )
        is None
    )


def test_original_cache_is_existence_only_and_metadata_has_no_redesign_fields(
    tmp_path,
):
    dataset = FakeCanonicalUnion()
    model = DummyVQModel(fail=True)
    cache_path = tmp_path / "punch_slide7.pt"
    legacy = {"anything": "is accepted when cache exists"}
    torch.save(legacy, cache_path)

    result = tma_select.preselect_tma_punch_with_vq(
        model,
        dataset,
        selector_opts(roi_size_um=999.0),
        torch.device("cpu"),
        torch.zeros(2, 3),
        {7: torch.zeros(2, 3)},
        ["A", "B"],
        cache_path=cache_path,
    )
    assert result is None
    assert model.calls == 0
    assert torch.load(cache_path, weights_only=False) == legacy

    cache_path.unlink()
    model = DummyVQModel()
    result = tma_select.preselect_tma_punch_with_vq(
        model,
        dataset,
        selector_opts(),
        torch.device("cpu"),
        torch.zeros(2, 3),
        {7: torch.zeros(2, 3)},
        ["A", "B"],
        cache_path=cache_path,
    )
    assert result is None
    metadata = torch.load(cache_path, weights_only=False)
    assert set(metadata).issuperset(
        {"punch_center", "window_px", "punch_select_method", "slide_idx"}
    )
    assert "schema_version" not in metadata
    assert "dataset_patch_signature" not in metadata
    assert "selector_config_sha256" not in metadata


def test_original_enable_flags_and_invalid_window_are_noop(tmp_path):
    dataset = FakeCanonicalUnion()
    model = DummyVQModel(fail=True)

    disabled = selector_opts(
        punch_select_enabled=False,
        tma_select_enabled=False,
    )
    assert (
        tma_select.preselect_tma_punch_with_vq(
            model,
            dataset,
            disabled,
            torch.device("cpu"),
            torch.zeros(2, 3),
            {7: torch.zeros(2, 3)},
            ["A", "B"],
            cache_path=tmp_path / "disabled.pt",
        )
        is None
    )
    assert model.calls == 0

    invalid_window = selector_opts(roi_size_um=0.0)
    assert (
        tma_select.preselect_tma_punch_with_vq(
            model,
            dataset,
            invalid_window,
            torch.device("cpu"),
            torch.zeros(2, 3),
            {7: torch.zeros(2, 3)},
            ["A", "B"],
            cache_path=tmp_path / "invalid.pt",
        )
        is None
    )
    assert model.calls == 0


def test_original_filter_split_gate_and_all_fail_open_paths(tmp_path):
    dataset = FakeCanonicalUnion()
    opts = selector_opts().data
    cache_path = tmp_path / "punch_slide7.pt"

    assert (
        tma_select.apply_cached_punch_filter(dataset, opts, cache_path)
        is dataset
    )
    cache_path.write_bytes(b"not a torch cache")
    assert (
        tma_select.apply_cached_punch_filter(dataset, opts, cache_path)
        is dataset
    )
    torch.save({"punch_center": [1000.0, 1000.0], "window_px": 2.0}, cache_path)
    assert (
        tma_select.apply_cached_punch_filter(dataset, opts, cache_path)
        is dataset
    )
    torch.save({"punch_center": [1.0, 1.0], "window_px": 2.0}, cache_path)

    val_dataset = FakeCanonicalUnion(mode="val")
    assert (
        tma_select.apply_cached_punch_filter(val_dataset, opts, cache_path)
        is val_dataset
    )

    filtered = tma_select.apply_cached_punch_filter(
        dataset, opts, cache_path, immune_sampler_boost=3.0
    )
    assert isinstance(filtered, tma_select.PunchSubset)
    assert filtered.coords_starts == [(0, 0)]
    assert filtered.coords_starts_unfiltered == dataset.coords_starts
    assert filtered.all_intersect is dataset.all_intersect
    assert filtered.df_ct is dataset.df_ct
    assert filtered.patch_weights == [3.0]
    assert dataset.coords_starts == [(0, 0), (0, 4), (4, 0), (4, 4), (8, 8)]


def test_original_coordinate_adapter_and_model_rng_mode_side_effects(tmp_path):
    dataset = FakeCanonicalUnion()
    original_transform = dataset.tfs
    sample = tma_select.CoordinateDatasetView(dataset)[1]
    assert len(sample) == 9
    assert torch.equal(sample[7], torch.tensor([1.0, 5.0]))
    assert int(sample[8]) == 7
    assert dataset.stain_aug is True
    assert dataset.tfs is original_transform
    assert dataset.read_observations == [(False, True)]

    model = DummyVQModel()
    model.train()
    model.child.eval()
    torch.manual_seed(31)
    state_before = torch.random.get_rng_state()
    tma_select.preselect_tma_punch_with_vq(
        model,
        dataset,
        selector_opts(),
        torch.device("cpu"),
        torch.zeros(2, 3),
        {7: torch.zeros(2, 3)},
        ["A", "B"],
        cache_path=tmp_path / "punch_slide7.pt",
    )
    assert model.training is True
    assert model.child.training is True
    assert not torch.equal(torch.random.get_rng_state(), state_before)

    failing_model = DummyVQModel(fail=True)
    failing_model.train()
    with pytest.raises(RuntimeError, match="selector fixture failure"):
        tma_select.preselect_tma_punch_with_vq(
            failing_model,
            dataset,
            selector_opts(),
            torch.device("cpu"),
            torch.zeros(2, 3),
            {7: torch.zeros(2, 3)},
            ["A", "B"],
            cache_path=tmp_path / "failing_punch_slide7.pt",
        )
    assert failing_model.training is False

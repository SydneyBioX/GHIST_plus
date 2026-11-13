#!/usr/bin/env python3
"""CPU contracts for the lookup-table fast path in DataProcessing.__getitem__.

The fast path replaced per-patch Python sets, per-id boolean-mask loops and
DataFrame.loc gathers. Each case here is checked against a deliberately slow
reference that spells out the original semantics, so any drift in the returned
tensors (ids, cell types, expression, padding, dtypes) fails.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import default_collate
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio import dataset_input  # noqa: E402
from dataio.dataset_input import (  # noqa: E402
    LEGACY_HISTOLOGY_NORMALIZATION,
    DataProcessing,
    _has_stable_stain_stats,
    _has_sufficient_tissue,
)
from dataio.collate import collate_compact_batch  # noqa: E402
from dataio.dataset_input_union import DataProcessingUnion  # noqa: E402

H = W = 48
N_GENES = 5
MAX_CELLS = 6


def _make_dataset(mode, seed=0, union=False, nuclei_dtype=np.uint32, compact=False):
    rng = np.random.RandomState(seed)
    # ids 1..8 labelled in blocks; 4 and 7 are ineligible (not in all_intersect)
    nuclei = np.zeros((H, 2 * W), dtype=nuclei_dtype)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 1000003]
    for k, cid in enumerate(ids):
        r, c = divmod(k, 3)
        nuclei[r * 16 : r * 16 + 10, c * 16 : c * 16 + 10] = cid
    # second patch (columns W..2W) holds just two cells, so batches are ragged
    nuclei[2:12, W + 2 : W + 12] = 2
    nuclei[20:30, W + 20 : W + 30] = 5
    hist = rng.randint(0, 256, size=(H, 2 * W, 3), dtype=np.uint8)

    all_intersect = [1, 2, 3, 5, 6, 8, 1000003, 999]  # 999 is not in the patch
    df_expr = pd.DataFrame(
        rng.rand(len(all_intersect) + 2, N_GENES) * 3.0,
        columns=[f"g{i}" for i in range(N_GENES)],
        index=all_intersect + [4, 7],  # df_expr may hold more cells than all_intersect
    )
    df_ct = pd.DataFrame(
        {"ct": rng.randint(1, 5, size=len(all_intersect))}, index=all_intersect
    )

    ds = object.__new__(DataProcessingUnion if union else DataProcessing)
    ds.mode = mode
    ds.hsize, ds.wsize = H, W
    ds.nuclei, ds.hist = nuclei, hist
    ds.coords_starts = [(0, 0), (0, W)]
    ds.compact_batches = compact
    ds.gene_names = list(df_expr.columns)
    ds.max_cells_per_patch = MAX_CELLS
    ds.opts_data = SimpleNamespace(random_cell_subsample_train=False)
    ds.normstain = False
    ds.stain_aug = False
    ds.foundation_model_input_normalization = LEGACY_HISTOLOGY_NORMALIZATION
    ds.norms_hist = np.array([[100.0, 110.0, 120.0], [50.0, 60.0, 70.0]])
    ds.all_intersect = all_intersect
    ds.comps_celltype = mode != "test"
    ds.df_ct = df_ct if ds.comps_celltype else None
    ds.df_expr = df_expr if mode != "test" else None
    ds._all_intersect_set = set(all_intersect)
    ds.tfs = ds.tfs_test = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    if union:
        ds.gene_mask = torch.ones(N_GENES)
        ds.slide_idx = 3
    ds._build_lookup_tables()
    return ds


def _reference(ds, hs=0, ws=0):
    """Original semantics, written the slow obvious way."""
    nuclei_patch = ds.nuclei[hs : hs + H, ws : ws + W]
    ids_seg = [int(i) for i in np.unique(nuclei_patch) if i != 0]
    valid = sorted(i for i in ids_seg if i in set(ds.all_intersect))
    nuclei_valid = nuclei_patch.copy()
    for i in ids_seg:
        if i not in valid:
            nuclei_valid[nuclei_patch == i] = 0
    if ds.comps_celltype and ds.mode != "test":
        types = nuclei_valid.copy()
        for i in valid:
            types[nuclei_valid == i] = int(ds.df_ct.loc[i, "ct"])
    else:
        types = np.where(nuclei_valid > 0, 1, 0)
    patch_ids = valid[:MAX_CELLS]
    n = len(patch_ids)
    expr = np.zeros((MAX_CELLS, N_GENES))
    if ds.mode != "test":
        for k, i in enumerate(patch_ids):
            expr[k] = ds.df_expr.loc[i].to_numpy()
    gt = np.zeros(MAX_CELLS)
    if ds.comps_celltype and ds.mode != "test":
        for k, i in enumerate(patch_ids):
            gt[k] = ds.df_ct.loc[i, "ct"] - 1
    pid = np.zeros(MAX_CELLS)
    pid[:n] = patch_ids
    return nuclei_valid, types, torch.from_numpy(expr).float(), torch.from_numpy(gt).long(), torch.from_numpy(pid).long(), n


@pytest.mark.parametrize("mode", ["train", "val", "test"])
@pytest.mark.parametrize("nuclei_dtype", [np.uint32, np.int32, np.int64])
@pytest.mark.parametrize("index", [0, 1])
def test_getitem_matches_reference(mode, nuclei_dtype, index):
    ds = _make_dataset(mode, nuclei_dtype=nuclei_dtype)
    nuclei_valid, types, expr, gt, pid, n = _reference(ds, ws=index * W)
    out = ds[index]
    nuclei_t, types_t, hist_t, expr_t, n_t, gt_t, pid_t = out
    # the joint transform casts to float32 before ids are cast back to long
    assert torch.equal(nuclei_t, torch.from_numpy(nuclei_valid.astype(np.float32)).long())
    assert torch.equal(types_t, torch.from_numpy(types.astype(np.float32)).long())
    assert hist_t.shape == (3, H, W)
    assert n_t.tolist() == [n]
    assert expr_t.dtype == torch.float32 and expr_t.shape == (MAX_CELLS, N_GENES)
    assert torch.equal(expr_t, expr)
    assert gt_t.dtype == torch.int64 and torch.equal(gt_t, gt)
    assert pid_t.dtype == torch.int64 and torch.equal(pid_t, pid)


def test_no_eligible_cells_is_all_background():
    ds = _make_dataset("val")
    ds.all_intersect = [999]
    ds._build_lookup_tables()
    nuclei_t, types_t, _, expr_t, n_t, gt_t, pid_t = ds[0]
    assert n_t.tolist() == [0]
    assert int(nuclei_t.sum()) == 0 and int(types_t.sum()) == 0
    assert float(expr_t.abs().sum()) == 0.0 and int(pid_t.sum()) == 0


def test_union_mask_is_float32_zero_padded():
    ds = _make_dataset("val", union=True)
    *_, expr_t, n_t, _, _, mask_t, slide_t = ds[0]
    n = int(n_t.item())
    assert mask_t.dtype == torch.float32 and mask_t.shape == expr_t.shape
    assert float(mask_t[:n].min()) == 1.0 and float(mask_t[n:].abs().sum()) == 0.0
    assert slide_t.item() == 3


def test_missing_cell_raises_like_loc():
    ds = _make_dataset("val")
    # cell 7 is in the patch and marked eligible, but has no df_ct/df_expr row
    ds.all_intersect = ds.all_intersect + [7]
    ds.df_expr = ds.df_expr.drop(index=7)
    ds._build_lookup_tables()
    with pytest.raises(KeyError):
        ds[0]


def test_is_valid_cell_edges():
    ds = _make_dataset("val")
    ids = np.array([0, 1, 4, 8, 999, 1000, 1000003, 2**31], dtype=np.int64)
    assert ds._is_valid_cell(ids).tolist() == [False, True, False, True, True, False, True, False]
    assert ds._is_valid_cell(ids[:0]).tolist() == []


def test_tissue_gate_accepts_precomputed_gray():
    """Regression: `if not gray:` raised ValueError on any ndarray (ambiguous truth value)."""
    rng = np.random.RandomState(1)
    img = rng.randint(0, 200, size=(64, 64, 3), dtype=np.uint8)
    gray = img.mean(axis=2)
    assert _has_sufficient_tissue(img, gray=gray) == _has_sufficient_tissue(img)
    assert _has_stable_stain_stats(img, gray=gray) == _has_stable_stain_stats(img)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    gray_w = white.mean(axis=2)
    assert not _has_sufficient_tissue(white, gray=gray_w)
    assert not _has_stable_stain_stats(white, gray=gray_w)


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_compact_items_hold_only_real_cells(mode):
    full = _make_dataset(mode)
    compact = _make_dataset(mode, compact=True)
    for index in (0, 1):
        a, b = full[index], compact[index]
        n = int(a[4].item())
        assert b[4].tolist() == [n]
        for j in (0, 1, 2, 4):  # image-shaped / scalar fields are identical
            assert torch.equal(a[j], b[j])
        for j in (3, 5, 6):  # per-cell fields: compact == padded[:n], same dtype
            assert b[j].shape[0] == n and b[j].dtype == a[j].dtype
            assert torch.equal(a[j][:n], b[j])


@pytest.mark.parametrize("mode", ["train", "val"])
def test_collate_compact_equals_padded_default_collate(mode):
    """Padding to the batch's largest cell count == default collate trimmed to it."""
    full = _make_dataset(mode, union=True)
    compact = _make_dataset(mode, union=True, compact=True)
    base = default_collate([full[0], full[1], full[1]])
    got = collate_compact_batch([compact[0], compact[1], compact[1]])
    rows = int(base[4].max())
    assert rows > 0 and got[3].shape[1] == rows
    assert len(got) == len(base) == 9
    for j, (x, y) in enumerate(zip(base, got)):
        if j in (3, 5, 6, 7):
            assert y.dtype == x.dtype
            assert torch.equal(x[:, :rows], y)
            assert float(x[:, rows:].abs().sum()) == 0.0  # what was dropped is padding
        else:
            assert torch.equal(x, y)


def test_collate_accepts_padded_items_and_all_empty_batches():
    ds = _make_dataset("val", union=True)
    ds.all_intersect = [999]  # no eligible cells -> n_cells == 0 everywhere
    ds._build_lookup_tables()
    got = collate_compact_batch([ds[0], ds[1]])
    assert got[4].tolist() == [[0], [0]]
    assert got[3].shape[:2] == (2, 1) and float(got[3].abs().sum()) == 0.0  # one zero row, not zero rows
    ds2 = _make_dataset("val", union=True)
    padded = collate_compact_batch([ds2[0], ds2[1]])  # padded (2048-style) items are trimmed
    assert padded[3].shape[1] == int(padded[4].max())

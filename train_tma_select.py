"""Training entry point and evaluation utilities for GHIST+."""

import argparse
import logging
import os
import sys
import shutil
import hashlib
from typing import NamedTuple
from types import SimpleNamespace
import inspect
import json
import csv
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort

if __package__:
    from .dataio.dataset_input_union_tma_select import DataProcessingUnion as DataProcessing
    from .dataio.dataset_input_tma_select import DataProcessing as DataProcessingBase
    from .model.framework import Framework
    from .utils.utils import *
else:
    from dataio.dataset_input_union_tma_select import DataProcessingUnion as DataProcessing
    from dataio.dataset_input_tma_select import DataProcessing as DataProcessingBase
    from model.framework import Framework
    from utils.utils import *


class CellGraph(NamedTuple):
    coords: torch.Tensor
    patch_index: torch.Tensor
    edge_index: torch.Tensor
    cells_per_patch: torch.Tensor


def _to_namespace(obj):
    if obj is None:
        return None
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, "_asdict"):
            return _to_namespace(obj._asdict())
        return tuple(_to_namespace(v) for v in obj)
    return obj


def _to_serialisable(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_serialisable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_knn_graph(coords: torch.Tensor, patch_index: torch.Tensor, k: int) -> torch.Tensor:
    if coords.numel() == 0 or patch_index.numel() == 0:
        device = coords.device if coords.numel() else patch_index.device
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edges = []
    unique_patches = torch.unique(patch_index, sorted=True)
    for pid in unique_patches:
        idx = torch.where(patch_index == pid)[0]
        n = idx.numel()
        if n <= 1:
            continue
        coords_patch = coords[idx]
        dist = torch.cdist(coords_patch, coords_patch, p=2)
        dist.fill_diagonal_(float("inf"))
        k_eff = min(max(int(k), 1), n - 1)
        if k_eff <= 0:
            continue
        nbr = dist.topk(k_eff, largest=False).indices
        src = idx.unsqueeze(1).expand(-1, k_eff)
        dst = idx[nbr]
        edges.append(torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0))

    if edges:
        return torch.cat(edges, dim=1)
    device = coords.device if coords.numel() else patch_index.device
    return torch.zeros((2, 0), dtype=torch.long, device=device)


class _InterleavedSlideBatchSampler:
    """
    Keep each batch slide-pure while interleaving slide batches across the epoch.

    This preserves the per-slide avgexp assumption used later in training while
    preventing AdamW from seeing long same-slide blocks.
    """

    def __init__(
        self,
        datasets,
        batch_size: int,
        seed: int = 0,
        shuffle_within_slide: bool = True,
        shuffle_slide_order: bool = True,
        slide_weights=None,
        weight_cap: float = 0.0,
    ):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_slide_order = bool(shuffle_slide_order)
        self.slide_weights = list(slide_weights) if slide_weights is not None else None
        self.weight_cap = float(weight_cap)
        self.offsets = []
        self.lengths = []
        acc = 0
        for ds in datasets:
            self.offsets.append(acc)
            length = len(ds)
            self.lengths.append(length)
            acc += length
        self.total_batches = sum(
            (length + self.batch_size - 1) // self.batch_size for length in self.lengths
        )
        self._epoch_index = 0

    def __len__(self):
        return self.total_batches

    def _sample_slide_indices(self, rng, length: int, weights):
        if length <= 0:
            return np.zeros((0,), dtype=np.int64)
        if weights is None:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        try:
            probs = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception:
            probs = None
        if probs is None or probs.shape[0] != length:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, a_min=0.0, a_max=None)
        if self.weight_cap > 0:
            probs = np.minimum(probs, self.weight_cap)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        probs = probs / total
        return rng.choice(length, size=length, replace=True, p=probs).astype(np.int64)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch_index)
        self._epoch_index += 1

        slide_batches = []
        for slide_idx, (offset, length) in enumerate(zip(self.offsets, self.lengths)):
            weights = None
            if self.slide_weights is not None and slide_idx < len(self.slide_weights):
                weights = self.slide_weights[slide_idx]
            local_idxs = self._sample_slide_indices(rng, length, weights)
            idxs = local_idxs + offset
            batches = [
                idxs[i : i + self.batch_size].tolist()
                for i in range(0, length, self.batch_size)
            ]
            slide_batches.append(batches)

        positions = [0] * len(slide_batches)
        active = [i for i, batches in enumerate(slide_batches) if batches]
        while active:
            round_order = list(active)
            if self.shuffle_slide_order and len(round_order) > 1:
                rng.shuffle(round_order)
            next_active = []
            for slide_idx in round_order:
                pos = positions[slide_idx]
                batches = slide_batches[slide_idx]
                if pos >= len(batches):
                    continue
                yield batches[pos]
                positions[slide_idx] += 1
                if positions[slide_idx] < len(batches):
                    next_active.append(slide_idx)
            active = next_active


def _build_knn_graph_global(coords: torch.Tensor, k: int) -> torch.Tensor:
    if coords.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    n = int(coords.shape[0])
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    dist = torch.cdist(coords, coords, p=2)
    dist.fill_diagonal_(float("inf"))
    k_eff = min(max(int(k), 1), n - 1)
    nbr = dist.topk(k_eff, largest=False).indices
    src = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k_eff)
    dst = nbr
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)


def _coalesce_edges(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device if edge_index is not None else "cpu")
    src = edge_index[0].long()
    dst = edge_index[1].long()
    valid = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes)
    src = src[valid]
    dst = dst[valid]
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    key = src * n_nodes + dst
    perm = torch.argsort(key)
    key_sorted = key[perm]
    keep = torch.ones_like(key_sorted, dtype=torch.bool)
    keep[1:] = key_sorted[1:] != key_sorted[:-1]
    uniq_idx = perm[keep]
    return torch.stack([src[uniq_idx], dst[uniq_idx]], dim=0)


def build_cell_graph(
    nuclei_batch: torch.Tensor,
    patch_ids_batch: torch.Tensor,
    k_neighbors: int,
    coords_batch: torch.Tensor | None = None,
    cell_coord_map: dict | None = None,
    cross_patch: bool = False,
    cross_patch_k: int | None = None,
) -> CellGraph:
    """
    Construct per-cell centroids and an intra-patch kNN graph so ECRM can mix
    logits at the cell granularity. The order of cells follows the sorted nuclei
    IDs to stay aligned with batch_ct/batch_expr tensors.
    """
    device = nuclei_batch.device
    B, H, W = nuclei_batch.shape
    coords_list = []
    patch_assign_list = []
    cells_per_patch = []
    coords_batch_valid_global = (
        isinstance(coords_batch, torch.Tensor)
        and coords_batch.ndim == 3
        and coords_batch.shape[0] == B
        and coords_batch.shape[2] >= 2
    )
    has_coord_map = isinstance(cell_coord_map, dict) and len(cell_coord_map) > 0
    has_global_coords = coords_batch_valid_global or has_coord_map

    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

    for b in range(B):
        mask = nuclei_batch[b]
        ids = torch.unique(mask, sorted=True)
        ids = ids[ids > 0]
        n_valid = int(ids.numel())
        cells_per_patch.append(n_valid)
        if n_valid == 0:
            continue

        local_rank = 0

        for cid in ids:
            c_mask = (mask == cid)
            area = c_mask.sum()
            if area.item() == 0:
                continue
            cid_int = int(cid.item())
            if has_coord_map and cid_int in cell_coord_map:
                cyx = cell_coord_map[cid_int]
                cy = torch.tensor(float(cyx[0]), dtype=torch.float32, device=device)
                cx = torch.tensor(float(cyx[1]), dtype=torch.float32, device=device)
            elif coords_batch_valid_global:
                if local_rank < coords_batch.shape[1]:
                    cxy = coords_batch[b, local_rank, :2].float().to(device)
                    cy = cxy[1]
                    cx = cxy[0]
                else:
                    c_mask = c_mask.float()
                    area = area.float()
                    cy = (c_mask * yy).sum() / area
                    cx = (c_mask * xx).sum() / area
                    cy = (cy / max(H - 1, 1)) * 2 - 1
                    cx = (cx / max(W - 1, 1)) * 2 - 1
            else:
                c_mask = c_mask.float()
                area = area.float()
                cy = (c_mask * yy).sum() / area
                cx = (c_mask * xx).sum() / area
                cy = (cy / max(H - 1, 1)) * 2 - 1
                cx = (cx / max(W - 1, 1)) * 2 - 1
            coords_list.append(torch.stack([cy, cx]))
            patch_assign_list.append(torch.tensor(b, dtype=torch.long, device=device))
            local_rank += 1

    if coords_list:
        coords = torch.stack(coords_list, dim=0)
        patch_index = torch.stack(patch_assign_list, dim=0)
    else:
        coords = torch.zeros((0, 2), device=device)
        patch_index = torch.zeros((0,), dtype=torch.long, device=device)

    edge_index = _build_knn_graph(coords, patch_index, k_neighbors)
    if cross_patch and has_global_coords and coords.shape[0] > 1:
        k_cross = int(cross_patch_k) if cross_patch_k is not None else int(k_neighbors)
        edge_cross = _build_knn_graph_global(coords, k_cross)
        edge_index = torch.cat([edge_index, edge_cross], dim=1)
        edge_index = _coalesce_edges(edge_index, n_nodes=int(coords.shape[0]))
    cells_per_patch_tensor = torch.tensor(
        cells_per_patch if cells_per_patch else [0] * B,
        dtype=torch.long,
        device=device,
    )
    return CellGraph(coords, patch_index, edge_index, cells_per_patch_tensor)


def _morans_many(expr: np.ndarray, coords: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Compute Moran’s I for all genes at once on a kNN graph.

    expr:  (N, G) float32
    coords:(N, 2) float32 (pixel coords; any consistent scale is fine)
    """
    from scipy.spatial import cKDTree

    if expr.ndim != 2 or coords.ndim != 2 or coords.shape[0] != expr.shape[0]:
        raise ValueError("expr/coords shape mismatch for Moran's I")
    n_cells, n_genes = expr.shape
    if n_cells < 3:
        return np.zeros((n_genes,), dtype=np.float64)

    k_eff = max(1, min(int(k), n_cells - 1))
    tree = cKDTree(coords.astype(np.float32, copy=False))
    idx = tree.query(coords, k=k_eff + 1)[1][:, 1:]  # (N,k)

    vc = expr.astype(np.float32, copy=False) - expr.mean(axis=0, keepdims=True)
    den = (vc**2).sum(axis=0).astype(np.float64)  # (G,)
    # (N,k,G) -> (G,)
    neigh = vc[idx, :]
    num = (vc[:, None, :] * neigh).sum(axis=(0, 1)).astype(np.float64)
    w = float(k_eff * n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = (n_cells / w) * (num / den)
        I[~np.isfinite(I)] = 0.0
        I[den <= 0] = 0.0
    return I


def giotto_rank_scores(expr, coords, k=8):
    """
    Giotto "rank" analog:
      1) rank expression per gene
      2) compute neighbourhood-mean rank on kNN graph
      3) score = corr(centered ranks, centered neigh-mean ranks)
    """
    from scipy.spatial import cKDTree
    from scipy.stats import rankdata

    expr = np.asarray(expr, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if expr.ndim != 2 or coords.ndim != 2 or expr.shape[0] != coords.shape[0]:
        raise ValueError("giotto_rank_scores expects expr (N,G) and coords (N,2)")
    n_cells, n_genes = expr.shape
    if n_cells < 3 or n_genes <= 0:
        return np.zeros((max(n_genes, 0),), dtype=np.float32)

    k_eff = max(1, min(int(k), n_cells - 1))
    tree = cKDTree(coords)
    neigh_idx = tree.query(coords, k=k_eff + 1)[1][:, 1:]  # drop self

    scores = np.zeros(n_genes, dtype=np.float32)
    for g in range(n_genes):
        ranks = rankdata(expr[:, g], method="average").astype(np.float32)
        neigh_mean = ranks[neigh_idx].mean(axis=1)
        r_center = ranks - ranks.mean()
        n_center = neigh_mean - neigh_mean.mean()
        denom = np.sqrt((r_center**2).sum() * (n_center**2).sum())
        if denom > 0:
            scores[g] = float(np.dot(r_center, n_center) / denom)
    return scores


def _resolve_divisions_fold(opts_regions, fold_id: int):
    if opts_regions is None:
        return None
    divisions = getattr(opts_regions, "divisions", None)
    if not isinstance(divisions, (list, tuple)) or len(divisions) == 0:
        return None
    idx = max(0, min(int(fold_id) - 1, len(divisions) - 1))
    div = divisions[idx]
    if not isinstance(div, (list, tuple)) or len(div) < 2:
        return None
    try:
        return float(div[0]), float(div[1])
    except Exception:
        return None


def _read_image_hw(fp_img: str):
    try:
        import tifffile

        with tifffile.TiffFile(fp_img) as tf:
            shape = tf.series[0].shape
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if len(shape) >= 3:
            if int(shape[0]) <= 5:
                return int(shape[1]), int(shape[2])
            return int(shape[0]), int(shape[1])
    except Exception:
        pass
    img = load_image(fp_img)
    return int(img.shape[0]), int(img.shape[1])


def _select_region_rows(y_coords: np.ndarray, whole_h: int, divisions_fold, mode: str):
    if divisions_fold is None:
        return np.ones(y_coords.shape[0], dtype=bool)
    div_a = int(round(float(divisions_fold[0]) * whole_h))
    div_b = int(round(float(divisions_fold[1]) * whole_h))
    in_band = (y_coords >= div_a) & (y_coords < div_b)
    if str(mode).lower() == "train":
        return ~in_band
    return in_band


def _compute_svg_rank_gene_indices_by_slide(
    sources,
    regions_obj,
    fold_id: int,
    mode_name: str,
    gene_names,
    *,
    k_neighbors: int = 8,
    sample_cap: int = 3000,
):
    """
    Precompute per-slide Giotto-ranked gene index orders once per run.
    """
    ranks_by_slide = {}
    divisions_fold = _resolve_divisions_fold(regions_obj, fold_id)

    for src in sources:
        slide_id = int(getattr(src, "slide_idx", -1))
        fp_expr = getattr(src, "fp_expr", None)
        if fp_expr is None or not os.path.isfile(fp_expr):
            logging.warning("SVG rank skipped for slide %s: missing fp_expr", slide_id)
            continue

        try:
            df_expr = pd.read_csv(fp_expr, index_col=0).reindex(columns=gene_names)
        except Exception as exc:
            logging.warning("SVG rank skipped for slide %s: failed reading expr (%s)", slide_id, exc)
            continue

        try:
            df_expr.index = pd.to_numeric(df_expr.index, errors="coerce").astype("Int64")
            df_expr = df_expr[~df_expr.index.isna()]
            df_expr.index = df_expr.index.astype(np.int64)
        except Exception:
            pass

        if df_expr.empty:
            logging.warning("SVG rank skipped for slide %s: empty expression table", slide_id)
            continue

        coord_map = _load_histology_coord_map_from_source(src)
        coords_df = None
        if coord_map:
            coords_df = pd.DataFrame.from_dict(coord_map, orient="index", columns=["y", "x"])
            coords_df.index = pd.to_numeric(coords_df.index, errors="coerce")
            coords_df = coords_df[~coords_df.index.isna()]
            coords_df.index = coords_df.index.astype(np.int64)

        if coords_df is None or coords_df.empty:
            # Fallback to segmentation-centroid extraction if coordinate table is unavailable.
            fp_seg = getattr(src, "fp_nuc_seg", None)
            if fp_seg is None or not os.path.isfile(fp_seg):
                logging.warning("SVG rank skipped for slide %s: no coords and no fp_nuc_seg", slide_id)
                continue
            ids_all = df_expr.index.to_numpy(dtype=np.int64, copy=False)
            if ids_all.size == 0:
                continue
            rng = np.random.default_rng(1701 + slide_id * 10007 + int(fold_id))
            sample_n = min(int(sample_cap), int(ids_all.size))
            ids_pick = (
                rng.choice(ids_all, size=sample_n, replace=False)
                if sample_n < ids_all.size
                else ids_all
            )
            kept_ids, coords_yx = _centroids_from_label_image(fp_seg, ids_pick, chunk_rows=256)
            if kept_ids.size < 3:
                logging.warning("SVG rank skipped for slide %s: too few centroid-matched cells", slide_id)
                continue
            idx = pd.Index(kept_ids.astype(np.int64))
            expr_arr = df_expr.reindex(idx).to_numpy(dtype=np.float32)
            coords_arr = coords_yx.astype(np.float32, copy=False)
            fp_hist = getattr(src, "fp_hist", None)
            if fp_hist and os.path.isfile(fp_hist):
                whole_h, _ = _read_image_hw(fp_hist)
                keep_region = _select_region_rows(
                    coords_arr[:, 0],
                    whole_h,
                    divisions_fold,
                    mode_name,
                )
                expr_arr = expr_arr[keep_region]
                coords_arr = coords_arr[keep_region]
        else:
            idx = df_expr.index.intersection(coords_df.index)
            if idx.empty:
                logging.warning("SVG rank skipped for slide %s: no expr/coord overlap", slide_id)
                continue
            expr_arr = df_expr.loc[idx].to_numpy(dtype=np.float32)
            coords_arr = coords_df.loc[idx, ["y", "x"]].to_numpy(dtype=np.float32)

            # Region filter (same semantics as dataset split).
            fp_hist = getattr(src, "fp_hist", None)
            if fp_hist and os.path.isfile(fp_hist):
                whole_h, _ = _read_image_hw(fp_hist)
                keep_region = _select_region_rows(
                    coords_arr[:, 0],
                    whole_h,
                    divisions_fold,
                    mode_name,
                )
                expr_arr = expr_arr[keep_region]
                coords_arr = coords_arr[keep_region]

            if coords_arr.shape[0] > sample_cap:
                rng = np.random.default_rng(1701 + slide_id * 10007 + int(fold_id))
                keep = rng.choice(coords_arr.shape[0], size=int(sample_cap), replace=False)
                expr_arr = expr_arr[keep]
                coords_arr = coords_arr[keep]

        if expr_arr.shape[0] < 3:
            logging.warning("SVG rank skipped for slide %s: <3 cells after filtering", slide_id)
            continue

        expr_arr = np.nan_to_num(expr_arr, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            scores = giotto_rank_scores(expr_arr, coords_arr, k=k_neighbors)
        except Exception as exc:
            logging.warning("SVG rank failed for slide %s: %s", slide_id, exc)
            continue
        scores = np.nan_to_num(scores.astype(np.float64), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        order = np.argsort(-scores, kind="stable").astype(np.int64)
        ranks_by_slide[slide_id] = order

    return ranks_by_slide


def _summarize_gene_pcc_distribution(corr: np.ndarray):
    vals = np.asarray(corr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "median": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "n_genes": 0,
        }
    return {
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "n_genes": int(vals.size),
    }


def _format_gene_pcc_triplet(stats: dict):
    if not isinstance(stats, dict):
        return "med=nan max=nan min=nan n=0"
    med = stats.get("median", float("nan"))
    mx = stats.get("max", float("nan"))
    mn = stats.get("min", float("nan"))
    n = int(stats.get("n_genes", 0) or 0)
    return "med={:.4f} max={:.4f} min={:.4f} n={}".format(
        float(med) if med is not None else float("nan"),
        float(mx) if mx is not None else float("nan"),
        float(mn) if mn is not None else float("nan"),
        n,
    )


def _log_gene_pcc_epoch(metrics: dict, *, split_tag: str, epoch: int, svg_topk=(20, 50)):
    if not isinstance(metrics, dict):
        return
    dist = metrics.get("gene_pcc_distribution_per_slide") or {}
    if not isinstance(dist, dict) or len(dist) == 0:
        logging.info("%s GenePCC epoch=%d: unavailable", split_tag, int(epoch))
        return
    for sid in sorted(dist):
        sid_stats = dist.get(sid) or {}
        all_s = _format_gene_pcc_triplet(sid_stats.get("all", {}))
        parts = [f"ALL({all_s})"]
        for k_svg in svg_topk:
            key = f"svg{int(k_svg)}"
            parts.append(f"SVG{int(k_svg)}({_format_gene_pcc_triplet(sid_stats.get(key, {}))})")
        logging.info(
            "%s GenePCC epoch=%d slide=%s %s",
            str(split_tag).upper(),
            int(epoch),
            sid,
            " ".join(parts),
        )


def _centroids_from_label_image(
    fp_label_tif: str,
    cell_ids: np.ndarray,
    *,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute centroids (y,x) for a subset of labels in a label TIFF efficiently.

    Returns:
      (kept_cell_ids, coords_yx) where coords_yx is float32 (N,2) in pixel units.
    """
    import tifffile

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    cell_ids = cell_ids[cell_ids > 0]
    if cell_ids.size == 0:
        return cell_ids, np.zeros((0, 2), dtype=np.float32)

    lab = tifffile.memmap(fp_label_tif)
    h, w = lab.shape
    # Avoid scanning the whole TIFF to find lab.max(); we only need to map the
    # labels we care about, and we can ignore out-of-range labels while scanning.
    max_label = int(cell_ids.max())

    # Map only requested labels -> compact 0..N-1 indices (fast filtering while scanning).
    idx_map = np.full(max_label + 1, -1, dtype=np.int32)
    idx_map[cell_ids] = np.arange(cell_ids.size, dtype=np.int32)

    counts = np.zeros(cell_ids.size, dtype=np.int64)
    sum_x = np.zeros(cell_ids.size, dtype=np.float64)
    sum_y = np.zeros(cell_ids.size, dtype=np.float64)

    for y0 in range(0, h, int(chunk_rows)):
        block = np.asarray(lab[y0 : y0 + int(chunk_rows)])
        # Map only labels within 0..max_label; anything larger is irrelevant (-1).
        inrange = block <= max_label
        if not np.any(inrange):
            continue
        mapped = np.full(block.shape, -1, dtype=np.int32)
        mapped[inrange] = idx_map[block[inrange]]
        valid = mapped >= 0
        if not np.any(valid):
            continue
        ys, xs = np.nonzero(valid)
        idx = mapped[ys, xs].astype(np.int64)
        counts += np.bincount(idx, minlength=cell_ids.size)
        sum_x += np.bincount(idx, weights=xs.astype(np.float64), minlength=cell_ids.size)
        sum_y += np.bincount(
            idx, weights=(ys.astype(np.float64) + float(y0)), minlength=cell_ids.size
        )

    keep = counts > 0
    kept_ids = cell_ids[keep]
    coords = np.stack([sum_y[keep] / counts[keep], sum_x[keep] / counts[keep]], axis=1).astype(
        np.float32
    )
    return kept_ids, coords


def flatten_expr_mask(mask_batch: torch.Tensor, n_cells_batch: torch.Tensor):
    """
    Collapse a per-patch expression mask (B, max_cells, n_genes) to a per-cell
    mask aligned with concatenated outputs shaped (total_cells, n_genes).
    """
    if mask_batch is None:
        return None
    if mask_batch.ndim == 2:
        return mask_batch

    device = mask_batch.device
    n_genes = mask_batch.shape[-1]
    n_cells_flat = n_cells_batch.view(-1)
    masks = []
    for idx in range(mask_batch.shape[0]):
        n_valid = int(n_cells_flat[idx].item()) if idx < n_cells_flat.numel() else 0
        if n_valid <= 0:
            continue
        masks.append(mask_batch[idx, :n_valid, :])
    if masks:
        return torch.cat(masks, dim=0)
    return torch.zeros((0, n_genes), device=device, dtype=mask_batch.dtype)


def flatten_expr(expr_batch: torch.Tensor, n_cells_batch: torch.Tensor):
    """
    Collapse a per-patch expression tensor (B, max_cells, n_genes) to a per-cell
    tensor aligned with concatenated outputs shaped (total_cells, n_genes).
    """
    if expr_batch is None:
        return None
    if expr_batch.ndim == 2:
        return expr_batch

    device = expr_batch.device
    n_genes = expr_batch.shape[-1]
    n_cells_flat = n_cells_batch.view(-1)
    exprs = []
    for idx in range(expr_batch.shape[0]):
        n_valid = int(n_cells_flat[idx].item()) if idx < n_cells_flat.numel() else 0
        if n_valid <= 0:
            continue
        exprs.append(expr_batch[idx, :n_valid, :])
    if exprs:
        return torch.cat(exprs, dim=0)
    return torch.zeros((0, n_genes), device=device, dtype=expr_batch.dtype)


def _load_histology_coord_map_from_source(src_obj):
    """
    Load a mapping {id_histology: (y_coord, x_coord)} when available.
    Uses matched nuclei CSV (`fp_nuc_sizes`) + `cell_coords.csv` in the same folder.
    """
    fp_match = getattr(src_obj, "fp_nuc_sizes", None)
    if fp_match is None or not os.path.isfile(fp_match):
        return {}

    fp_coords = os.path.join(os.path.dirname(fp_match), "cell_coords.csv")
    if not os.path.isfile(fp_coords):
        return {}

    try:
        df_match = pd.read_csv(fp_match)
        if not {"id_histology", "id_xenium"}.issubset(set(df_match.columns)):
            return {}
        df_coords = pd.read_csv(fp_coords)
        id_col = next((c for c in ("cell_id", "id_xenium", "id") if c in df_coords.columns), None)
        x_col = next((c for c in ("x_coord", "x", "X") if c in df_coords.columns), None)
        y_col = next((c for c in ("y_coord", "y", "Y") if c in df_coords.columns), None)
        if id_col is None or x_col is None or y_col is None:
            return {}

        left = df_match[["id_histology", "id_xenium"]].copy()
        right = df_coords[[id_col, x_col, y_col]].copy()
        left["id_histology"] = pd.to_numeric(left["id_histology"], errors="coerce")
        left["id_xenium"] = pd.to_numeric(left["id_xenium"], errors="coerce")
        right[id_col] = pd.to_numeric(right[id_col], errors="coerce")
        right[x_col] = pd.to_numeric(right[x_col], errors="coerce")
        right[y_col] = pd.to_numeric(right[y_col], errors="coerce")
        merged = left.merge(right, left_on="id_xenium", right_on=id_col, how="inner")
        merged = merged.dropna(subset=["id_histology", x_col, y_col])
        if merged.empty:
            return {}

        coord_map = {}
        for r in merged.itertuples(index=False):
            hid = int(getattr(r, "id_histology"))
            xv = float(getattr(r, x_col))
            yv = float(getattr(r, y_col))
            coord_map[hid] = (yv, xv)
        return coord_map
    except Exception:
        return {}


def _load_ct_series_for_classes(fp_ct, classes):
    if fp_ct is None or not os.path.isfile(fp_ct):
        return None

    df_ct = pd.read_csv(fp_ct, index_col="c_id")
    ct_numeric = pd.to_numeric(df_ct["ct"], errors="coerce")
    is_all_numbers = ct_numeric.notna().all()
    unassigned_idx = (
        classes.index("Unassigned")
        if any(str(c).strip().lower() == "unassigned" for c in classes)
        else (len(classes) - 1)
    )
    if not is_all_numbers:
        ct_dict = {name: idx for idx, name in enumerate(classes)}
        mapped = (
            df_ct["ct"]
            .astype(str)
            .str.strip()
            .map(ct_dict)
            .fillna(unassigned_idx)
            .astype(int)
        )
        return mapped

    ct_vals = ct_numeric.astype(int)
    if ct_vals.min() >= 1 and ct_vals.max() <= len(classes):
        ct_vals = ct_vals - 1
    return ct_vals.clip(lower=0, upper=len(classes) - 1)


def _source_domain_id(src):
    try:
        return int(getattr(src, "domain_id", 0))
    except (TypeError, ValueError):
        return 0


def build_avgexp_df_by_slide(
    all_sources,
    stats_sources,
    gene_names,
    classes,
    expr_scale: float,
    *,
    holdout_mask_by_slide=None,
    expr_per_source=None,
    domain_specific: bool = False,
):
    """
    Build per-slide avgexp priors directly from raw expression + cell-type files.

    `all_sources` defines which slides receive a prior; `stats_sources` defines
    which slides contribute statistics. Keeping `stats_sources` train/val only
    preserves leak-free priors for test-time use while still allowing a union
    panel over all known sources.

    When `domain_specific` is enabled, each target slide uses cell-type priors
    estimated from train/val slides with the same `domain_id`. If a domain has
    no train/val statistics for a gene/cell-type, the code falls back to the
    global train/val statistic for that missing entry.
    """
    if not all_sources or not stats_sources or not gene_names or not classes:
        return {}

    holdout_mask_by_slide = holdout_mask_by_slide or {}
    expr_per_source = expr_per_source or {}

    n_classes_local = len(classes)
    n_genes_local = len(gene_names)
    slide_ct_sums_map = {}
    slide_ct_counts_map = {}
    global_ct_sums = np.zeros((n_classes_local, n_genes_local), dtype=np.float64)
    global_ct_counts = np.zeros((n_classes_local, n_genes_local), dtype=np.int64)
    domain_ct_sums_map = {}
    domain_ct_counts_map = {}

    for src in stats_sources:
        slide_id = int(getattr(src, "slide_idx", -1))
        domain_id = _source_domain_id(src)
        fp_expr_key = getattr(src, "fp_expr", None)
        if fp_expr_key is None or not os.path.isfile(fp_expr_key):
            continue
        ct_series_tmp = _load_ct_series_for_classes(getattr(src, "fp_cell_type", None), classes)
        if ct_series_tmp is None:
            continue
        if fp_expr_key in expr_per_source:
            df_expr_raw = expr_per_source[fp_expr_key].reindex(columns=gene_names)
        else:
            df_expr_raw = pd.read_csv(fp_expr_key, index_col=0).reindex(columns=gene_names)
        try:
            df_expr_raw.index = df_expr_raw.index.astype(int)
        except Exception:
            pass
        try:
            ct_series_tmp.index = ct_series_tmp.index.astype(int)
        except Exception:
            pass
        idx = df_expr_raw.index.intersection(ct_series_tmp.index)
        if idx.empty:
            continue
        expr_arr = df_expr_raw.loc[idx].to_numpy(dtype=np.float64)
        expr_arr = np.log1p(np.clip(expr_arr, 0.0, None)) * float(expr_scale)
        ct_arr = ct_series_tmp.loc[idx].to_numpy(dtype=np.int64)
        sums = np.zeros((n_classes_local, n_genes_local), dtype=np.float64)
        counts = np.zeros((n_classes_local, n_genes_local), dtype=np.int64)
        valid_mask = np.isfinite(expr_arr)
        for ct_val in np.unique(ct_arr):
            if ct_val < 0 or ct_val >= n_classes_local:
                continue
            rows = ct_arr == ct_val
            if not rows.any():
                continue
            expr_rows = expr_arr[rows]
            valid_rows = valid_mask[rows]
            sums[ct_val] += np.nansum(np.where(valid_rows, expr_rows, 0.0), axis=0)
            counts[ct_val] += valid_rows.sum(axis=0)
        slide_ct_sums_map[slide_id] = sums
        slide_ct_counts_map[slide_id] = counts
        global_ct_sums += sums
        global_ct_counts += counts
        if domain_specific:
            if domain_id not in domain_ct_sums_map:
                domain_ct_sums_map[domain_id] = np.zeros_like(global_ct_sums)
                domain_ct_counts_map[domain_id] = np.zeros_like(global_ct_counts)
            domain_ct_sums_map[domain_id] += sums
            domain_ct_counts_map[domain_id] += counts

    if not slide_ct_sums_map:
        return {}

    with np.errstate(divide="ignore", invalid="ignore"):
        global_ct_means = np.full_like(global_ct_sums, np.nan, dtype=np.float64)
        np.divide(
            global_ct_sums,
            global_ct_counts,
            out=global_ct_means,
            where=global_ct_counts > 0,
        )

    gene_sums_global = global_ct_sums.sum(axis=0)
    gene_counts_global = global_ct_counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_mean_global = np.full_like(gene_sums_global, np.nan, dtype=np.float64)
        np.divide(
            gene_sums_global,
            gene_counts_global,
            out=gene_mean_global,
            where=gene_counts_global > 0,
        )
    gene_mean_global = np.where(np.isfinite(gene_mean_global), gene_mean_global, 0.0)

    domain_ct_means_map = {}
    domain_gene_mean_map = {}
    if domain_specific:
        for domain_id, domain_sums in domain_ct_sums_map.items():
            domain_counts = domain_ct_counts_map[domain_id]
            with np.errstate(divide="ignore", invalid="ignore"):
                domain_means = np.full_like(domain_sums, np.nan, dtype=np.float64)
                np.divide(
                    domain_sums,
                    domain_counts,
                    out=domain_means,
                    where=domain_counts > 0,
                )
            domain_gene_sums = domain_sums.sum(axis=0)
            domain_gene_counts = domain_counts.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                domain_gene_mean = np.full_like(domain_gene_sums, np.nan, dtype=np.float64)
                np.divide(
                    domain_gene_sums,
                    domain_gene_counts,
                    out=domain_gene_mean,
                    where=domain_gene_counts > 0,
                )
            domain_ct_means_map[domain_id] = domain_means
            domain_gene_mean_map[domain_id] = domain_gene_mean

    avgexp_df_by_slide = {}
    for src in all_sources:
        slide_id = int(getattr(src, "slide_idx", -1))
        domain_id = _source_domain_id(src)
        sums = slide_ct_sums_map.get(slide_id)
        counts = slide_ct_counts_map.get(slide_id)
        base_ct_sums = global_ct_sums
        base_ct_counts = global_ct_counts
        base_ct_means = global_ct_means
        base_gene_mean = gene_mean_global
        if domain_specific and domain_id in domain_ct_means_map:
            base_ct_sums = domain_ct_sums_map[domain_id]
            base_ct_counts = domain_ct_counts_map[domain_id]
            base_ct_means = domain_ct_means_map[domain_id]
            base_gene_mean = domain_gene_mean_map[domain_id]

        if sums is not None and counts is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                slide_means = np.full_like(sums, np.nan, dtype=np.float64)
                np.divide(sums, counts, out=slide_means, where=counts > 0)
        else:
            slide_means = None

        ref = base_ct_means.copy()
        present_mask = (
            counts.sum(axis=0) > 0 if counts is not None else np.zeros(n_genes_local, dtype=bool)
        )
        holdout_mask = holdout_mask_by_slide.get(slide_id)
        if holdout_mask is None:
            holdout_mask_bool = np.zeros(n_genes_local, dtype=bool)
        else:
            holdout_mask_bool = np.asarray(holdout_mask, dtype=bool)

        use_slide_mask = present_mask & (~holdout_mask_bool)
        if slide_means is not None and use_slide_mask.any():
            ref[:, use_slide_mask] = slide_means[:, use_slide_mask]

        if holdout_mask_bool.any() and sums is not None and counts is not None:
            excl_sums = base_ct_sums - sums
            excl_counts = base_ct_counts - counts
            with np.errstate(divide="ignore", invalid="ignore"):
                excl_means = np.full_like(excl_sums, np.nan, dtype=np.float64)
                np.divide(
                    excl_sums,
                    excl_counts,
                    out=excl_means,
                    where=excl_counts > 0,
                )
            excl_gene_sums = excl_sums.sum(axis=0)
            excl_gene_counts = excl_counts.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                excl_gene_mean = np.full_like(excl_gene_sums, np.nan, dtype=np.float64)
                np.divide(
                    excl_gene_sums,
                    excl_gene_counts,
                    out=excl_gene_mean,
                    where=excl_gene_counts > 0,
                )

            hold_idx = np.where(holdout_mask_bool)[0]
            excl_block = excl_means[:, hold_idx]
            fallback = np.broadcast_to(excl_gene_mean[hold_idx], excl_block.shape)
            excl_block = np.where(np.isfinite(excl_block), excl_block, fallback)
            excl_block = np.where(np.isfinite(excl_block), excl_block, 0.0)
            ref[:, hold_idx] = excl_block

        ref = np.where(np.isfinite(ref), ref, global_ct_means)
        ref = np.where(np.isfinite(ref), ref, np.broadcast_to(base_gene_mean, ref.shape))
        ref = np.where(np.isfinite(ref), ref, np.broadcast_to(gene_mean_global, ref.shape))
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        avgexp_df_by_slide[slide_id] = pd.DataFrame(ref, index=classes, columns=gene_names)

    return avgexp_df_by_slide


def _build_train_region_avgexp_df_by_slide(
    src_list,
    train_regions,
    fold_id: int,
    gene_names,
    classes,
    expr_scale: float,
    *,
    fallback_df_by_slide=None,
    holdout_mask_by_slide=None,
    domain_specific: bool = False,
):
    """
    Build per-slide avgexp priors using only cells that fall inside the effective
    training region for each slide. This is used for validation to avoid leaking
    val-region cells into the prior that predicts the val region. If slide-level
    holdout genes are configured, preserve the existing non-leaky behavior for
    those genes by using leave-one-slide-out global statistics instead of the
    same-slide prior.
    """
    if not src_list or not gene_names or not classes:
        return {}

    n_classes_local = len(classes)
    n_genes_local = len(gene_names)
    slide_ct_sums_map = {}
    slide_ct_counts_map = {}
    global_ct_sums = np.zeros((n_classes_local, n_genes_local), dtype=np.float64)
    global_ct_counts = np.zeros((n_classes_local, n_genes_local), dtype=np.int64)
    domain_ct_sums_map = {}
    domain_ct_counts_map = {}

    divisions_fold = _resolve_divisions_fold(train_regions, fold_id)

    for src in src_list:
        slide_id = int(getattr(src, "slide_idx", -1))
        domain_id = _source_domain_id(src)
        fp_expr = getattr(src, "fp_expr", None)
        if fp_expr is None or not os.path.isfile(fp_expr):
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: missing fp_expr",
                slide_id,
            )
            continue

        ct_series_tmp = _load_ct_series_for_classes(getattr(src, "fp_cell_type", None), classes)
        if ct_series_tmp is None:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: missing fp_cell_type",
                slide_id,
            )
            continue

        coord_map = _load_histology_coord_map_from_source(src)
        if not coord_map:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: missing coord map",
                slide_id,
            )
            continue

        try:
            df_expr_raw = pd.read_csv(fp_expr, index_col=0).reindex(columns=gene_names)
        except Exception as exc:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: failed to read expr (%s)",
                slide_id,
                exc,
            )
            continue

        try:
            df_expr_raw.index = df_expr_raw.index.astype(int)
        except Exception:
            pass
        try:
            ct_series_tmp.index = ct_series_tmp.index.astype(int)
        except Exception:
            pass

        common_ids = [int(cid) for cid in df_expr_raw.index.intersection(ct_series_tmp.index)]
        common_ids = [cid for cid in common_ids if cid in coord_map]
        if not common_ids:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: no overlapping cells with coords",
                slide_id,
            )
            continue

        whole_h, _ = _read_image_hw(getattr(src, "fp_hist"))
        y_coords = np.asarray([float(coord_map[cid][0]) for cid in common_ids], dtype=np.float64)
        keep_train = _select_region_rows(y_coords, whole_h, divisions_fold, mode="train")
        train_ids = np.asarray(common_ids, dtype=np.int64)[keep_train]
        if train_ids.size == 0:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: empty train region",
                slide_id,
            )
            continue

        expr_arr = df_expr_raw.loc[train_ids].to_numpy(dtype=np.float64)
        expr_arr = np.log1p(np.clip(expr_arr, 0.0, None)) * float(expr_scale)
        ct_arr = ct_series_tmp.loc[train_ids].to_numpy(dtype=np.int64)

        sums = np.zeros((n_classes_local, n_genes_local), dtype=np.float64)
        counts = np.zeros((n_classes_local, n_genes_local), dtype=np.int64)
        valid_mask = np.isfinite(expr_arr)
        for ct_val in np.unique(ct_arr):
            if ct_val < 0 or ct_val >= n_classes_local:
                continue
            rows = ct_arr == ct_val
            if not rows.any():
                continue
            expr_rows = expr_arr[rows]
            valid_rows = valid_mask[rows]
            sums[ct_val] += np.nansum(np.where(valid_rows, expr_rows, 0.0), axis=0)
            counts[ct_val] += valid_rows.sum(axis=0)

        slide_ct_sums_map[slide_id] = sums
        slide_ct_counts_map[slide_id] = counts
        global_ct_sums += sums
        global_ct_counts += counts
        if domain_specific:
            if domain_id not in domain_ct_sums_map:
                domain_ct_sums_map[domain_id] = np.zeros_like(global_ct_sums)
                domain_ct_counts_map[domain_id] = np.zeros_like(global_ct_counts)
            domain_ct_sums_map[domain_id] += sums
            domain_ct_counts_map[domain_id] += counts

    if not slide_ct_sums_map:
        return {}

    with np.errstate(divide="ignore", invalid="ignore"):
        global_ct_means = np.full_like(global_ct_sums, np.nan, dtype=np.float64)
        np.divide(
            global_ct_sums,
            global_ct_counts,
            out=global_ct_means,
            where=global_ct_counts > 0,
        )

    gene_sums_global = global_ct_sums.sum(axis=0)
    gene_counts_global = global_ct_counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_mean_global = np.full_like(gene_sums_global, np.nan, dtype=np.float64)
        np.divide(
            gene_sums_global,
            gene_counts_global,
            out=gene_mean_global,
            where=gene_counts_global > 0,
        )
    gene_mean_global = np.where(np.isfinite(gene_mean_global), gene_mean_global, 0.0)

    domain_ct_means_map = {}
    domain_gene_mean_map = {}
    if domain_specific:
        for domain_id, domain_sums in domain_ct_sums_map.items():
            domain_counts = domain_ct_counts_map[domain_id]
            with np.errstate(divide="ignore", invalid="ignore"):
                domain_means = np.full_like(domain_sums, np.nan, dtype=np.float64)
                np.divide(
                    domain_sums,
                    domain_counts,
                    out=domain_means,
                    where=domain_counts > 0,
                )
            domain_gene_sums = domain_sums.sum(axis=0)
            domain_gene_counts = domain_counts.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                domain_gene_mean = np.full_like(domain_gene_sums, np.nan, dtype=np.float64)
                np.divide(
                    domain_gene_sums,
                    domain_gene_counts,
                    out=domain_gene_mean,
                    where=domain_gene_counts > 0,
                )
            domain_ct_means_map[domain_id] = domain_means
            domain_gene_mean_map[domain_id] = domain_gene_mean

    avgexp_df_by_slide = {}
    fallback_df_by_slide = fallback_df_by_slide or {}
    holdout_mask_by_slide = holdout_mask_by_slide or {}
    for src in src_list:
        slide_id = int(getattr(src, "slide_idx", -1))
        domain_id = _source_domain_id(src)
        sums = slide_ct_sums_map.get(slide_id)
        counts = slide_ct_counts_map.get(slide_id)
        if sums is None or counts is None:
            if slide_id in fallback_df_by_slide:
                avgexp_df_by_slide[slide_id] = fallback_df_by_slide[slide_id]
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            slide_means = np.full_like(sums, np.nan, dtype=np.float64)
            np.divide(sums, counts, out=slide_means, where=counts > 0)

        base_ct_sums = global_ct_sums
        base_ct_counts = global_ct_counts
        base_ct_means = global_ct_means
        base_gene_mean = gene_mean_global
        if domain_specific and domain_id in domain_ct_means_map:
            base_ct_sums = domain_ct_sums_map[domain_id]
            base_ct_counts = domain_ct_counts_map[domain_id]
            base_ct_means = domain_ct_means_map[domain_id]
            base_gene_mean = domain_gene_mean_map[domain_id]

        ref = base_ct_means.copy()
        present_mask = counts.sum(axis=0) > 0
        holdout_mask = holdout_mask_by_slide.get(slide_id)
        if holdout_mask is None:
            holdout_mask_bool = np.zeros(n_genes_local, dtype=bool)
        else:
            holdout_mask_bool = np.asarray(holdout_mask, dtype=bool)

        use_slide_mask = present_mask & (~holdout_mask_bool)
        if use_slide_mask.any():
            ref[:, use_slide_mask] = slide_means[:, use_slide_mask]

        if holdout_mask_bool.any():
            excl_sums = base_ct_sums - sums
            excl_counts = base_ct_counts - counts
            with np.errstate(divide="ignore", invalid="ignore"):
                excl_means = np.full_like(excl_sums, np.nan, dtype=np.float64)
                np.divide(
                    excl_sums,
                    excl_counts,
                    out=excl_means,
                    where=excl_counts > 0,
                )
            excl_gene_sums = excl_sums.sum(axis=0)
            excl_gene_counts = excl_counts.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                excl_gene_mean = np.full_like(excl_gene_sums, np.nan, dtype=np.float64)
                np.divide(
                    excl_gene_sums,
                    excl_gene_counts,
                    out=excl_gene_mean,
                    where=excl_gene_counts > 0,
                )

            hold_idx = np.where(holdout_mask_bool)[0]
            excl_block = excl_means[:, hold_idx]
            fallback = np.broadcast_to(excl_gene_mean[hold_idx], excl_block.shape)
            excl_block = np.where(np.isfinite(excl_block), excl_block, fallback)
            excl_block = np.where(np.isfinite(excl_block), excl_block, 0.0)
            ref[:, hold_idx] = excl_block

        ref = np.where(np.isfinite(ref), ref, global_ct_means)
        ref = np.where(np.isfinite(ref), ref, np.broadcast_to(base_gene_mean, ref.shape))
        ref = np.where(np.isfinite(ref), ref, np.broadcast_to(gene_mean_global, ref.shape))
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        avgexp_df_by_slide[slide_id] = pd.DataFrame(ref, index=classes, columns=gene_names)

    return avgexp_df_by_slide


def _punch_cache_path(base_dir: str, slide_idx: int) -> str:
    return os.path.join(base_dir, f"punch_slide{int(slide_idx)}.pt")


def _as_float_attr(obj, names, default):
    for name in names:
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return float(default)


def preselect_tma_punch_with_vq(
    model,
    src,
    opts,
    train_regions,
    fold_id: int,
    experiment_path: str,
    device,
    expr_ref_torch,
    expr_ref_torch_map,
    gene_names,
    classes,
    *,
    graph_k: int = 6,
):
    """
    Select one training ROI/punch per slide using the old VQ ROI logic.

    This is only used by train_tma_select.py. It scans the unfiltered training
    patches, scores candidate windows by VQ-code balance/coverage/size, then
    optionally re-ranks top candidates with cell-type and molecular diversity.
    """
    enabled = bool(
        getattr(opts.data, "punch_select_enabled", True)
        or getattr(opts.data, "tma_select_enabled", True)
    )
    if not enabled:
        return

    src = _to_namespace(src)
    slide_idx = int(getattr(src, "slide_idx", -1))
    cache_path = _punch_cache_path(experiment_path, slide_idx)
    force = bool(getattr(opts.data, "punch_force_reselect", False))
    if os.path.isfile(cache_path) and not force:
        logging.info("[punch] Using cached punch selection for slide=%s: %s", slide_idx, cache_path)
        return

    window_um = _as_float_attr(opts.data, ("broadcast_window_um", "roi_size_um"), 1000.0)
    pixel_um = _as_float_attr(opts.data, ("pixel_size_um",), _as_float_attr(opts.model, ("pixel_size_um",), 0.2125))
    if window_um <= 0 or pixel_um <= 0:
        logging.warning("[punch] Invalid window_um=%.3f pixel_um=%.5f; skipping slide=%s", window_um, pixel_um, slide_idx)
        return
    window_px = window_um / pixel_um

    opts_stain_norm = opts.stain_norm
    if hasattr(opts_stain_norm, "fp_norm_ref") and isinstance(opts_stain_norm.fp_norm_ref, (list, tuple)):
        opts_stain_norm = SimpleNamespace(**vars(opts_stain_norm))
        opts_stain_norm.fp_norm_ref = opts_stain_norm.fp_norm_ref[0]

    logging.info("[punch] Preselecting TMA punch via VQ for slide=%s", slide_idx)
    ds = DataProcessingBase(
        src,
        opts.data,
        train_regions,
        opts.comps,
        opts_stain_norm,
        classes,
        gene_names,
        device,
        experiment_path,
        False,
        fold_id,
        mode="train",
        immune_sampler_boost=1.0,
        immune_class_multipliers=None,
        return_coords=True,
        force_no_punch_filter=True,
    )
    if getattr(ds, "tfs_test", None) is not None:
        ds.tfs = ds.tfs_test

    num_workers = int(getattr(opts.data, "punch_num_workers", 0))
    dl = DataLoader(
        dataset=ds,
        batch_size=max(1, int(getattr(opts.training, "batch_size", 1))),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=getattr(opts.data, "pin_memory", False),
    )

    coords_all = []
    vq_err_all = []
    vq_idx_all = []
    n_cells_all = []
    expr_sum_all = []
    ct_counts_all = []
    expr_mean_all = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for (
            batch_nuclei,
            batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            patch_ids,
            patch_coords,
            patch_slide_idx,
        ) in dl:
            batch_nuclei = batch_nuclei.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)
            patch_ids = patch_ids.to(device)
            patch_coords = patch_coords.to(device)
            patch_slide_idx = patch_slide_idx.to(device)

            graph = build_cell_graph(
                batch_nuclei,
                patch_ids,
                k_neighbors=max(int(graph_k), 2),
            )
            expr_ref_batch = (
                expr_ref_torch_map.get(slide_idx, expr_ref_torch)
                if isinstance(expr_ref_torch_map, dict)
                else expr_ref_torch
            )
            model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                expr_ref_batch,
                batch_ct,
                batch_expr,
                patch_ids=patch_ids,
                coords_cells=graph.coords,
                cell_edge_index=graph.edge_index,
                cell_patch_ids=graph.patch_index,
            )

            aux = getattr(model, "last_aux_losses", {}) or {}
            vq_err = aux.get("vq_patch_err")
            vq_idx = aux.get("vq_patch_idx")

            coords_np = patch_coords.detach().cpu().numpy().astype(np.float32)
            slides_np = patch_slide_idx.detach().cpu().numpy().astype(np.int64)
            n_cells_np = batch_n_cells.detach().cpu().numpy().reshape(-1).astype(np.int64)
            expr_sum_np = batch_expr.detach().sum(dim=(1, 2)).cpu().numpy().astype(np.float32)

            keep = slides_np == slide_idx
            if not np.any(keep):
                continue
            coords_all.append(coords_np[keep])
            n_cells_all.append(n_cells_np[keep])
            expr_sum_all.append(expr_sum_np[keep])
            if vq_err is not None and isinstance(vq_err, torch.Tensor) and vq_err.numel() > 0:
                vq_err_all.append(vq_err.detach().cpu().numpy()[keep])
            if vq_idx is not None and isinstance(vq_idx, torch.Tensor) and vq_idx.numel() > 0:
                vq_idx_all.append(vq_idx.detach().cpu().numpy()[keep])

            n_classes = int(len(classes)) if classes is not None else 0
            if n_classes > 0 and batch_ct is not None and batch_expr is not None:
                max_cells = int(batch_ct.shape[1])
                n_cells_vec = batch_n_cells.view(-1).long().to(device)
                mask_cells = torch.arange(max_cells, device=device).unsqueeze(0) < n_cells_vec.unsqueeze(1)
                ct_onehot = F.one_hot(batch_ct.clamp_min(0), num_classes=n_classes).float()
                ct_onehot = ct_onehot * mask_cells.unsqueeze(-1).float()
                ct_counts_all.append(ct_onehot.sum(dim=1).detach().cpu().numpy()[keep])

                mask_float = mask_cells.unsqueeze(-1).to(batch_expr.dtype)
                expr_sum_genes = (batch_expr * mask_float).sum(dim=1)
                denom = n_cells_vec.clamp_min(1).to(batch_expr.dtype).unsqueeze(1)
                expr_mean_all.append((expr_sum_genes / denom).detach().cpu().numpy()[keep])

    if was_training:
        model.train()

    if not coords_all:
        logging.warning("[punch] No patches found for slide=%s; keeping full slide.", slide_idx)
        return

    coords_all = np.concatenate(coords_all, axis=0).astype(np.float32)
    n_cells_all = np.concatenate(n_cells_all, axis=0).astype(np.int64)
    expr_sum_all = np.concatenate(expr_sum_all, axis=0).astype(np.float32)
    vq_err_all = np.concatenate(vq_err_all, axis=0).astype(np.float32) if vq_err_all else None
    vq_idx_all = np.concatenate(vq_idx_all, axis=0).astype(np.int64) if vq_idx_all else None
    ct_counts_all = np.concatenate(ct_counts_all, axis=0).astype(np.int64) if ct_counts_all else None
    expr_mean_all = np.concatenate(expr_mean_all, axis=0).astype(np.float32) if expr_mean_all else None

    qc_min_cells = int(getattr(opts.data, "punch_qc_min_cells", getattr(opts.data, "broadcast_min_cells", 1)))
    qc_min_expr_sum = float(getattr(opts.data, "punch_qc_min_expr_sum", 0.0))
    qc_mask = (n_cells_all >= qc_min_cells) & (expr_sum_all > qc_min_expr_sum)
    n_qc_total = int(qc_mask.sum())
    if n_qc_total <= 0:
        logging.warning("[punch] No QC patches for slide=%s; falling back to minimum VQ error.", slide_idx)
        if vq_err_all is None or vq_err_all.size == 0:
            return
        best_coord = coords_all[int(np.argmin(vq_err_all))]
        best_meta = {"fallback": "min_vq_err_no_qc"}
    elif vq_idx_all is None or vq_idx_all.size == 0:
        logging.warning("[punch] VQ indices unavailable for slide=%s; falling back to minimum VQ error.", slide_idx)
        if vq_err_all is None or vq_err_all.size == 0:
            return
        idx = np.where(qc_mask)[0]
        best_coord = coords_all[idx[int(np.argmin(vq_err_all[idx]))]]
        best_meta = {"fallback": "min_vq_err_no_idx"}
    else:
        vq_cfg = getattr(opts.model, "vq_patch", None)
        k_clusters = int(getattr(vq_cfg, "n_codes", 0)) if vq_cfg is not None else 0
        k_clusters = max(k_clusters, int(vq_idx_all.max()) + 1, 2)
        roi_balance_target = str(getattr(opts.data, "punch_roi_balance_target", "uniform")).strip().lower()
        if roi_balance_target == "slide":
            counts_slide = np.bincount(vq_idx_all[qc_mask], minlength=k_clusters).astype(np.float32)
            target = counts_slide / counts_slide.sum() if counts_slide.sum() > 0 else np.full((k_clusters,), 1.0 / k_clusters, dtype=np.float32)
        else:
            target = np.full((k_clusters,), 1.0 / k_clusters, dtype=np.float32)

        half = 0.5 * float(window_px)
        pool_coords = coords_all[qc_mask]
        roi_area = float(window_px * window_px)
        wsi_area = float(
            (coords_all[:, 0].max() - coords_all[:, 0].min() + 1.0)
            * (coords_all[:, 1].max() - coords_all[:, 1].min() + 1.0)
        )
        sampling_factor = float(getattr(opts.data, "punch_sampling_factor", 500.0))
        num_samples = int(sampling_factor * wsi_area / max(roi_area, 1.0))
        num_samples = int(getattr(opts.data, "punch_num_samples", num_samples))
        num_samples = max(int(getattr(opts.data, "punch_min_samples", 200)), num_samples)
        num_samples = min(int(getattr(opts.data, "punch_max_samples", 5000)), num_samples, int(pool_coords.shape[0]))
        num_samples = max(num_samples, 1)
        seed = int(getattr(opts.training, "seed", getattr(opts.training, "batch_sampler_seed", 0)))
        rng = np.random.default_rng(seed + 10007 * max(slide_idx, 0))
        cand_idx = rng.choice(pool_coords.shape[0], size=num_samples, replace=pool_coords.shape[0] < num_samples)
        cand_centers = pool_coords[cand_idx]

        roi_w_balance = float(getattr(opts.data, "punch_roi_w_balance", 1.0))
        roi_w_coverage = float(getattr(opts.data, "punch_roi_w_coverage", 1.0))
        roi_w_size = float(getattr(opts.data, "punch_roi_w_size", 1.0))
        roi_min_qc = int(getattr(opts.data, "punch_roi_min_qc", 1))
        candidates = []
        for center in cand_centers:
            in_mask = (np.abs(coords_all[:, 0] - center[0]) <= half) & (np.abs(coords_all[:, 1] - center[1]) <= half)
            n_total = int(in_mask.sum())
            if n_total <= 0:
                continue
            qc_in = in_mask & qc_mask
            n_qc = int(qc_in.sum())
            if n_qc < roi_min_qc:
                continue
            coverage = float(np.sqrt(n_qc / max(n_total, 1)))
            size_score = float(1.0 / (1.0 + np.exp(-2.0 * (n_qc / max(n_qc_total, 1)))))
            counts = np.bincount(vq_idx_all[qc_in], minlength=k_clusters).astype(np.float32)
            if counts.sum() <= 0:
                balance = 0.0
            else:
                p = counts / counts.sum()
                balance = float(np.dot(p, target) / (np.linalg.norm(p) * np.linalg.norm(target) + 1e-8))
            score = (balance ** roi_w_balance) * (coverage ** roi_w_coverage) * (size_score ** roi_w_size)
            candidates.append(
                {
                    "center": center,
                    "stage1_score": float(score),
                    "balance": float(balance),
                    "coverage": float(coverage),
                    "size": float(size_score),
                    "n_qc_patches": int(n_qc),
                    "n_total_patches": int(n_total),
                }
            )
        if not candidates:
            logging.warning("[punch] ROI candidate sampling produced no valid windows for slide=%s.", slide_idx)
            return

        candidates.sort(key=lambda d: d["stage1_score"], reverse=True)
        best = candidates[0]
        best_coord = best["center"]
        best_meta = dict(best)
        best_meta.pop("center", None)

        if ct_counts_all is not None and expr_mean_all is not None:
            stage2_topk = max(1, int(getattr(opts.data, "punch_stage2_topk", 25)))
            stage2_min_ratio = float(np.clip(float(getattr(opts.data, "punch_stage2_min_stage1_ratio", 0.98)), 0.0, 1.0))
            subset = [c for c in candidates if c["stage1_score"] >= best["stage1_score"] * stage2_min_ratio]
            subset = subset[:stage2_topk] if len(subset) >= stage2_topk else candidates[:stage2_topk]

            n_classes = int(ct_counts_all.shape[1])
            keep_ct = np.ones((n_classes,), dtype=bool)
            if bool(getattr(opts.data, "punch_stage2_exclude_unassigned", True)) and classes:
                for idx_cls, name in enumerate(classes):
                    if str(name).strip().lower() == "unassigned":
                        keep_ct[idx_cls] = False
                        break
            slide_ct = ct_counts_all[qc_mask].sum(axis=0).astype(np.float32)[keep_ct]
            p_slide = slide_ct / slide_ct.sum() if slide_ct.sum() > 0 else np.full((keep_ct.sum(),), 1.0 / max(int(keep_ct.sum()), 1), dtype=np.float32)
            p_uniform = np.full_like(p_slide, 1.0 / max(p_slide.size, 1))
            alpha = float(np.clip(float(getattr(opts.data, "punch_stage2_ct_blend_alpha", 0.5)), 0.0, 1.0))
            p_target = (1.0 - alpha) * p_slide + alpha * p_uniform
            slide_cells_total = max(int(n_cells_all[qc_mask].sum()), 1)

            def _cosine(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

            def _mol_rank(X, w):
                min_patches = int(getattr(opts.data, "punch_stage2_min_patches_for_mol", 8))
                if X.shape[0] < min_patches:
                    return 0.0
                w = w.astype(np.float64)
                w_sum = float(w.sum())
                if w_sum <= 0:
                    return 0.0
                X = X.astype(np.float64)
                mu = (X * w[:, None]).sum(axis=0) / w_sum
                Xc = X - mu
                cov = (Xc * w[:, None]).T @ Xc / w_sum
                cov.flat[:: cov.shape[0] + 1] += float(getattr(opts.data, "punch_stage2_mol_ridge", 1e-4))
                eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
                s = float(eig.sum())
                if s <= 0:
                    return 0.0
                p = eig / s
                return float(np.clip(np.exp(-np.sum(p * np.log(p + 1e-12))) / max(X.shape[1], 1), 0.0, 1.0))

            stage2_w_ct = float(getattr(opts.data, "punch_stage2_w_ct", 1.0))
            stage2_w_cells = float(getattr(opts.data, "punch_stage2_w_cells", 1.0))
            stage2_w_mol = float(getattr(opts.data, "punch_stage2_w_mol", 1.0))
            best_stage2 = None
            best_stage2_score = -1.0
            for cand in subset:
                center = cand["center"]
                in_mask = (np.abs(coords_all[:, 0] - center[0]) <= half) & (np.abs(coords_all[:, 1] - center[1]) <= half)
                qc_in = in_mask & qc_mask
                if not np.any(qc_in):
                    continue
                cells_roi = int(n_cells_all[qc_in].sum())
                if cells_roi < int(getattr(opts.data, "punch_stage2_min_cells", 1)):
                    continue
                ct_roi = ct_counts_all[qc_in].sum(axis=0).astype(np.float32)[keep_ct]
                s_ct = _cosine(ct_roi / ct_roi.sum(), p_target) if ct_roi.sum() > 0 else 0.0
                s_ct = float(np.clip(s_ct, 0.0, 1.0))
                s_cells = float(1.0 / (1.0 + np.exp(-2.0 * (cells_roi / float(slide_cells_total)))))
                s_mol = _mol_rank(expr_mean_all[qc_in], n_cells_all[qc_in].astype(np.float32))
                score = (s_ct ** stage2_w_ct) * (s_cells ** stage2_w_cells) * (s_mol ** stage2_w_mol)
                cand.update(
                    {
                        "stage2_score": float(score),
                        "stage2_ct": float(s_ct),
                        "stage2_cells": float(s_cells),
                        "stage2_mol": float(s_mol),
                        "stage2_cells_roi": int(cells_roi),
                    }
                )
                if score > best_stage2_score:
                    best_stage2_score = float(score)
                    best_stage2 = cand
            if best_stage2 is not None:
                best_coord = best_stage2["center"]
                best_meta = dict(best_stage2)
                best_meta.pop("center", None)

    meta = {
        "punch_center": [float(best_coord[0]), float(best_coord[1])],
        "window_px": float(window_px),
        "punch_select_method": "vq_roi_score_twostage",
        "slide_idx": int(slide_idx),
        **best_meta,
    }
    torch.save(meta, cache_path)
    logging.info("[punch] Selected slide=%s center=%s window_px=%.1f -> %s", slide_idx, meta["punch_center"], window_px, cache_path)


class PanelCompletionHead(nn.Module):
    """
    Gene-conditioned imputation head ("panel completion").

    Inputs (per cell):
      - delta_obs: (expr_true - expr_ref_base) on observed genes only
      - mask_obs:  0/1 mask for observed genes
      - delta_morph (optional): (out_expr - expr_ref_base) as morphology residual

    Output:
      - delta_hat: predicted residual for all genes (to add on top of expr_ref_base)
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_morph: bool = True,
        morph_gate_init: float = -2.0,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.use_morph = bool(use_morph)

        in_dim = self.n_genes * 2  # delta_obs + mask_obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.n_genes),
        )
        if self.use_morph:
            self.morph_gate = nn.Parameter(
                torch.full((self.n_genes,), float(morph_gate_init))
            )
        else:
            self.register_parameter("morph_gate", None)

    def forward(
        self,
        delta_obs: torch.Tensor,
        mask_obs: torch.Tensor,
        delta_morph=None,
    ) -> torch.Tensor:
        if delta_obs.shape[-1] != self.n_genes or mask_obs.shape[-1] != self.n_genes:
            raise ValueError("PanelCompletionHead: gene dimension mismatch.")
        x = torch.cat([delta_obs, mask_obs], dim=1)
        delta_hat = self.net(x)
        if self.use_morph and delta_morph is not None:
            gate = torch.sigmoid(self.morph_gate).view(1, -1)
            delta_hat = delta_hat + gate * delta_morph
        return delta_hat


def pearson_loss(pred, target, eps=1e-6):
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    num = (pred * target).mean(dim=0)
    denom = (
        pred.std(dim=0, unbiased=False) * target.std(dim=0, unbiased=False)
    ).clamp_min(eps)
    corr = num / denom
    return (1.0 - corr).mean()


def masked_pearson(pred, target, mask=None, eps=1e-6):
    """
    Pearson distance (1 - corr) with an optional per-gene mask.
    """
    if mask is None:
        return pearson_loss(pred, target, eps=eps)
    mask = mask.float()
    valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_center = (pred * mask - (pred * mask).sum(dim=1, keepdim=True) / valid)
    targ_center = (target * mask - (target * mask).sum(dim=1, keepdim=True) / valid)
    num = (pred_center * targ_center * mask).sum(dim=1)
    denom = (
        ((pred_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
        * ((targ_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
    ).clamp_min(eps)
    corr = num / denom
    return (1.0 - corr).mean()


def evaluate_validation(
    model,
    dataloader,
    expr_ref_torch,
    device,
    n_classes,
    graph_k=None,
    graph_cross_patch=False,
    graph_cross_patch_k=None,
    slide_coord_map_by_slide=None,
    expr_ref_torch_map=None,
    holdout_mask_by_slide=None,
    gene_names=None,
    epoch=None,
    per_gene_dir=None,
    svg_rank_gene_indices_by_slide=None,
    svg_topk=(20, 50),
):
    # exposed for external quick evals
    def masked_mse(pred, target, mask):
        if mask is None:
            return F.mse_loss(pred, target, reduction="mean")
        mask = mask.float()
        denom = mask.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.sum((pred - target) ** 2 * mask) / denom

    def masked_pearson(pred, target, mask, eps=1e-6):
        if mask is None:
            return pearson_loss(pred, target, eps=eps)
        mask = mask.float()
        valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred_center = (pred * mask - (pred * mask).sum(dim=1, keepdim=True) / valid)
        targ_center = (target * mask - (target * mask).sum(dim=1, keepdim=True) / valid)
        num = (pred_center * targ_center * mask).sum(dim=1)
        denom = (
            ((pred_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
            * ((targ_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
        ).clamp_min(eps)
        corr = num / denom
        return (1.0 - corr).mean()
    def masked_mse(pred, target, mask):
        if mask is None:
            return F.mse_loss(pred, target, reduction="mean")
        mask = mask.float()
        denom = mask.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.sum((pred - target) ** 2 * mask) / denom

    def masked_pearson(pred, target, mask, eps=1e-6):
        if mask is None:
            return pearson_loss(pred, target, eps=eps)
        mask = mask.float()
        valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred_center = (pred * mask - (pred * mask).sum(dim=1, keepdim=True) / valid)
        targ_center = (target * mask - (target * mask).sum(dim=1, keepdim=True) / valid)
        num = (pred_center * targ_center * mask).sum(dim=1)
        denom = (
            ((pred_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
            * ((targ_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
        ).clamp_min(eps)
        corr = num / denom
        return (1.0 - corr).mean()
    model.eval()
    holdout_sse = 0.0
    holdout_sae = 0.0
    holdout_n = 0.0
    ct_counts = np.zeros(n_classes, dtype=np.int64)
    pred_counts = np.zeros(n_classes, dtype=np.int64)
    correct_counts = np.zeros(n_classes, dtype=np.int64)
    # per-slide class counts
    ct_counts_per = {}
    pred_counts_per = {}
    correct_counts_per = {}
    total_cells = 0

    # Ensure we always have a mapping to avoid NameError when per-slide refs are missing
    expr_ref_torch_map = expr_ref_torch_map or {}
    svg_rank_gene_indices_by_slide = svg_rank_gene_indices_by_slide or {}
    svg_topk = tuple(int(k) for k in (svg_topk or (20, 50)) if int(k) > 0)

    # Debug: log expression variance/pearson once per validation run
    log_debug_once = True

    compute_per_gene = gene_names is not None and len(gene_names) > 0
    per_gene_stats = {}
    holdout_gene_stats = {}

    def _get_per_gene_stats(slide_id):
        if slide_id not in per_gene_stats:
            n = len(gene_names)
            per_gene_stats[slide_id] = {
                "count": np.zeros(n, dtype=np.float64),
                "sum_pred": np.zeros(n, dtype=np.float64),
                "sum_targ": np.zeros(n, dtype=np.float64),
                "sum_pred2": np.zeros(n, dtype=np.float64),
                "sum_targ2": np.zeros(n, dtype=np.float64),
                "sum_xy": np.zeros(n, dtype=np.float64),
            }
        return per_gene_stats[slide_id]

    def _get_holdout_gene_stats(slide_id):
        if slide_id not in holdout_gene_stats:
            n = len(gene_names)
            holdout_gene_stats[slide_id] = {
                "count": np.zeros(n, dtype=np.float64),
                "sum_pred": np.zeros(n, dtype=np.float64),
                "sum_targ": np.zeros(n, dtype=np.float64),
                "sum_pred2": np.zeros(n, dtype=np.float64),
                "sum_targ2": np.zeros(n, dtype=np.float64),
                "sum_xy": np.zeros(n, dtype=np.float64),
            }
        return holdout_gene_stats[slide_id]

    try:
        import inspect

        fwd_params = inspect.signature(model.forward).parameters
        supports_cell_graph = all(
            k in fwd_params for k in ("coords_cells", "cell_edge_index", "cell_patch_ids")
        )
    except Exception:
        supports_cell_graph = False

    with torch.no_grad():
        for (
            batch_nuclei,
            batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            patch_ids,
            batch_expr_mask,
            batch_slide_id,
        ) in dataloader:
            batch_nuclei = batch_nuclei.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_expr_mask = batch_expr_mask.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)
            patch_ids = patch_ids.to(device)
            slide_ids_unique = torch.unique(batch_slide_id)
            if slide_ids_unique.numel() != 1:
                raise RuntimeError("Mixed slides in batch; set batch_size=1 for per-slide avgexp.")
            slide_id_val = int(slide_ids_unique.item())
            expr_ref_batch = expr_ref_torch_map.get(slide_id_val, expr_ref_torch)
            model_extra_kwargs = {}
            if supports_cell_graph:
                coord_map_slide = None
                if isinstance(slide_coord_map_by_slide, dict):
                    coord_map_slide = slide_coord_map_by_slide.get(slide_id_val)
                graph = build_cell_graph(
                    batch_nuclei,
                    patch_ids,
                    k_neighbors=graph_k or 6,
                    coords_batch=None,
                    cell_coord_map=coord_map_slide,
                    cross_patch=bool(graph_cross_patch),
                    cross_patch_k=graph_cross_patch_k,
                )
                model_extra_kwargs = {
                    "coords_cells": graph.coords,
                    "cell_edge_index": graph.edge_index,
                    "cell_patch_ids": graph.patch_index,
                }

            # Prevent leakage in validation: for slide-level held-out genes, replace GT expr
            # with the (non-leaky) ref baseline before calling the model.
            batch_expr_for_model = batch_expr
            holdout_mask_vec = None
            if holdout_mask_by_slide is not None:
                holdout_mask_vec = holdout_mask_by_slide.get(slide_id_val)
            if holdout_mask_vec is not None and np.any(np.asarray(holdout_mask_vec) > 0):
                holdout_idx = torch.from_numpy(
                    np.where(np.asarray(holdout_mask_vec) > 0.5)[0].astype(np.int64)
                ).to(device)
                if holdout_idx.numel() > 0:
                    batch_expr_for_model = batch_expr.clone()
                    n_ref_local = expr_ref_batch.shape[0] if expr_ref_batch is not None else 0
                    for b in range(batch_expr_for_model.shape[0]):
                        n_valid = int(batch_n_cells[b].item())
                        if n_valid <= 0:
                            continue
                        if (
                            n_ref_local > 0
                            and batch_ct is not None
                            and batch_ct.numel() > 0
                        ):
                            ct_b = batch_ct[b, :n_valid].long().clamp(min=0).clamp(max=n_ref_local - 1)
                            baseline_all = expr_ref_batch[ct_b]
                            batch_expr_for_model[b, :n_valid, holdout_idx] = baseline_all[:, holdout_idx]
                        else:
                            batch_expr_for_model[b, :n_valid, holdout_idx] = 0.0

            (
                out_cell_type,
                out_map,
                batch_ct_pc,
                out_expr,
                out_expr_immune,
                out_expr_invasive,
                out_cell_type_expr,
                fv_cell_type_expr,
                out_cell_type_gt_expr,
                fv_cell_type_gt_expr,
                batch_expr_pc,
                comp_estimated,
                _,
                _,
            ) = model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                expr_ref_batch,
                batch_ct,
                batch_expr_for_model,
                patch_ids=patch_ids,
                **model_extra_kwargs,
            )
            batch_expr_mask_pc = flatten_expr_mask(batch_expr_mask, batch_n_cells)

            if batch_ct_pc.shape[0] == 0:
                continue

            if log_debug_once:
                # Keep one debug line without any per-cell Pearson metrics (too noisy for logs).
                aux = getattr(model, "last_aux_losses", {}) if hasattr(model, "last_aux_losses") else {}
                ref_base = aux.get("expr_ref_base")
                ref_gate = aux.get("expr_ref_gate")
                ref_stats = ""
                if ref_base is not None and isinstance(ref_base, torch.Tensor) and ref_base.numel() > 0:
                    ref_stats = (
                        " | ref_mean=%.4f ref_std=%.4f ref_min=%.4f ref_max=%.4f"
                        % (
                            ref_base.mean().item(),
                            ref_base.std(unbiased=False).item(),
                            ref_base.min().item(),
                            ref_base.max().item(),
                        )
                    )
                gate_stats = ""
                if ref_gate is not None and isinstance(ref_gate, torch.Tensor) and ref_gate.numel() > 0:
                    gate_stats = " | expr_ref_gate=%.4f" % ref_gate.detach().float().mean().item()
                zero_frac = (
                    (out_expr <= 0).float().mean().item() if out_expr.numel() > 0 else 0.0
                )
                logging.info(
                    "Validation diagnostics: zero_frac=%.4f%s%s",
                    zero_frac,
                    ref_stats,
                    gate_stats,
                )
                log_debug_once = False

            # Use the raw (unmodified) GT expression for metrics.
            expr_true_pc = flatten_expr(batch_expr, batch_n_cells)
            if expr_true_pc is None or expr_true_pc.shape != out_expr.shape:
                expr_true_pc = batch_expr_pc

            pred = out_expr.detach().cpu().numpy()
            target = expr_true_pc.detach().cpu().numpy()
            mask_np = (
                batch_expr_mask_pc.detach().cpu().numpy()
                if batch_expr_mask_pc is not None
                else None
            )

            # Panel completion prediction (gene-conditioned imputation head), if present.
            pred_holdout = pred
            completion_head = getattr(model, "completion_head", None)
            if completion_head is not None and batch_expr_mask_pc is not None:
                try:
                    aux = getattr(model, "last_aux_losses", {}) if hasattr(model, "last_aux_losses") else {}
                    ref_base = aux.get("expr_ref_base")
                    ref_base_pc = (
                        ref_base
                        if ref_base is not None and isinstance(ref_base, torch.Tensor) and ref_base.shape == out_expr.shape
                        else torch.zeros_like(out_expr)
                    )
                    mask_obs_f = (batch_expr_mask_pc > 0.5).float()
                    delta_obs = (expr_true_pc - ref_base_pc) * mask_obs_f
                    delta_morph = out_expr - ref_base_pc
                    delta_hat = completion_head(delta_obs, mask_obs_f, delta_morph)
                    pred_completed = F.relu(ref_base_pc + delta_hat)
                    # At inference we keep measured genes as-is; this doesn't affect holdout metrics.
                    pred_completed = mask_obs_f * expr_true_pc + (1.0 - mask_obs_f) * pred_completed
                    pred_holdout = pred_completed.detach().cpu().numpy()
                except Exception as exc:
                    logging.warning("Panel completion head failed in validation (using morph-only preds): %s", exc)
                    pred_holdout = pred

            # Holdout-only metrics (imputation): evaluate on held-out genes only
            hold_mask_vec = None
            if holdout_mask_by_slide is not None:
                hold_mask_vec = holdout_mask_by_slide.get(slide_id_val)
            if hold_mask_vec is not None:
                hold_mask_vec = np.asarray(hold_mask_vec, dtype=np.float64)
                if hold_mask_vec.ndim == 1 and hold_mask_vec.size == pred.shape[1] and hold_mask_vec.sum() > 0:
                    mask_hold = np.broadcast_to(hold_mask_vec, pred_holdout.shape)
                    diff = pred_holdout - target
                    holdout_sse += float(((diff ** 2) * mask_hold).sum())
                    holdout_sae += float((np.abs(diff) * mask_hold).sum())
                    holdout_n += float(mask_hold.sum())

                    if compute_per_gene:
                        stats_h = _get_holdout_gene_stats(slide_id_val)
                        c = mask_hold.sum(axis=0)
                        if np.any(c > 0):
                            stats_h["count"] += c
                            stats_h["sum_pred"] += (pred_holdout * mask_hold).sum(axis=0)
                            stats_h["sum_targ"] += (target * mask_hold).sum(axis=0)
                            stats_h["sum_pred2"] += ((pred_holdout**2) * mask_hold).sum(axis=0)
                            stats_h["sum_targ2"] += ((target**2) * mask_hold).sum(axis=0)
                            stats_h["sum_xy"] += ((pred_holdout * target) * mask_hold).sum(axis=0)

            if compute_per_gene:
                mask_gene = mask_np if mask_np is not None else np.ones_like(pred, dtype=np.float64)
                stats = _get_per_gene_stats(slide_id_val)
                c = mask_gene.sum(axis=0)
                if np.any(c > 0):
                    stats["count"] += c
                    stats["sum_pred"] += (pred * mask_gene).sum(axis=0)
                    stats["sum_targ"] += (target * mask_gene).sum(axis=0)
                    stats["sum_pred2"] += ((pred**2) * mask_gene).sum(axis=0)
                    stats["sum_targ2"] += ((target**2) * mask_gene).sum(axis=0)
                    stats["sum_xy"] += ((pred * target) * mask_gene).sum(axis=0)

            ct_np = batch_ct_pc.detach().cpu().numpy().astype(int)
            if ct_np.size > 0:
                ct_counts += np.bincount(ct_np, minlength=n_classes)
                total_cells += ct_np.size

                preds_np = (
                    out_cell_type.detach().cpu().argmax(dim=1).numpy().astype(int)
                )
                pred_counts += np.bincount(preds_np, minlength=n_classes)
                matches = preds_np == ct_np
                if matches.any():
                    correct_counts += np.bincount(
                        ct_np[matches], minlength=n_classes
                    )
                # per-slide tallies
                if slide_id_val not in ct_counts_per:
                    ct_counts_per[slide_id_val] = np.zeros(n_classes, dtype=np.int64)
                    pred_counts_per[slide_id_val] = np.zeros(n_classes, dtype=np.int64)
                    correct_counts_per[slide_id_val] = np.zeros(n_classes, dtype=np.int64)
                ct_counts_per[slide_id_val] += np.bincount(ct_np, minlength=n_classes)
                pred_counts_per[slide_id_val] += np.bincount(preds_np, minlength=n_classes)
                if matches.any():
                    correct_counts_per[slide_id_val] += np.bincount(
                        ct_np[matches], minlength=n_classes
                    )

    def _pooled_gene_corr_metrics(stats_by_slide):
        if not stats_by_slide:
            return {
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "n_genes": 0,
            }
        stats_iter = list(stats_by_slide.values())
        count = np.sum([s["count"] for s in stats_iter], axis=0)
        sum_pred = np.sum([s["sum_pred"] for s in stats_iter], axis=0)
        sum_targ = np.sum([s["sum_targ"] for s in stats_iter], axis=0)
        sum_pred2 = np.sum([s["sum_pred2"] for s in stats_iter], axis=0)
        sum_targ2 = np.sum([s["sum_targ2"] for s in stats_iter], axis=0)
        sum_xy = np.sum([s["sum_xy"] for s in stats_iter], axis=0)
        denom_x = sum_pred2 - (sum_pred ** 2) / np.maximum(count, 1e-8)
        denom_y = sum_targ2 - (sum_targ ** 2) / np.maximum(count, 1e-8)
        num = sum_xy - (sum_pred * sum_targ) / np.maximum(count, 1e-8)
        denom = np.sqrt(np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0))
        corr = np.full_like(num, np.nan, dtype=np.float64)
        valid = count > 1
        corr[valid] = num[valid] / np.maximum(denom[valid], 1e-8)
        corr_vals = corr[np.isfinite(corr)]
        if corr_vals.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "n_genes": 0,
            }
        return {
            "mean": float(np.mean(corr_vals)),
            "median": float(np.median(corr_vals)),
            "max": float(np.max(corr_vals)),
            "p95": float(np.percentile(corr_vals, 95)),
            "n_genes": int(corr_vals.size),
        }

    metrics = {}
    if holdout_n > 0:
        metrics["holdout_mse"] = float(holdout_sse / holdout_n)
        metrics["holdout_mae"] = float(holdout_sae / holdout_n)
    else:
        metrics["holdout_mse"] = 0.0
        metrics["holdout_mae"] = 0.0

    if compute_per_gene and holdout_gene_stats:
        corrs_all = []
        corrs_per_slide = {}
        holdout_per_gene = {}
        holdout_per_gene_files = {}
        for sid, stats in holdout_gene_stats.items():
            c = stats["count"]
            denom_x = stats["sum_pred2"] - (stats["sum_pred"] ** 2) / np.maximum(c, 1e-8)
            denom_y = stats["sum_targ2"] - (stats["sum_targ"] ** 2) / np.maximum(c, 1e-8)
            num = stats["sum_xy"] - (stats["sum_pred"] * stats["sum_targ"]) / np.maximum(c, 1e-8)
            denom = np.sqrt(np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0))
            corr = np.full_like(num, np.nan, dtype=np.float64)
            valid = c > 1
            corr[valid] = num[valid] / np.maximum(denom[valid], 1e-8)
            # only genes that were actually held out in this slide have c>0
            corr_vals = corr[np.isfinite(corr)]
            if corr_vals.size > 0:
                corrs_per_slide[int(sid)] = float(np.nanmean(corr_vals))
                corrs_all.extend(corr_vals.tolist())
            else:
                corrs_per_slide[int(sid)] = float("nan")

            # Per-gene holdout Pearson (only held-out genes for this slide)
            hold_mask_vec = None
            if holdout_mask_by_slide is not None:
                hold_mask_vec = holdout_mask_by_slide.get(int(sid))
            if hold_mask_vec is not None:
                hold_mask_vec = np.asarray(hold_mask_vec, dtype=np.float64)
                hold_idx = np.where(hold_mask_vec > 0.0)[0]
            else:
                hold_idx = np.where(c > 0)[0]

            if hold_idx.size > 0:
                # keep deterministic ordering
                hold_idx = np.sort(hold_idx)
                holdout_per_gene[int(sid)] = {
                    str(gene_names[i]): (float(corr[i]) if np.isfinite(corr[i]) else float("nan"))
                    for i in hold_idx
                }
                if per_gene_dir and epoch is not None:
                    os.makedirs(per_gene_dir, exist_ok=True)
                    fp = os.path.join(
                        per_gene_dir,
                        f"slide{sid}_epoch{epoch}_holdout_per_gene_pearson.csv",
                    )
                    df_corr_h = pd.DataFrame(
                        {
                            "gene": [gene_names[i] for i in hold_idx],
                            "pearson": corr[hold_idx],
                            "n_cells": c[hold_idx],
                        }
                    )
                    df_corr_h.to_csv(fp, index=False)
                    holdout_per_gene_files[int(sid)] = fp
                    logging.info(
                        "Saved holdout per-gene Pearson for slide %s (epoch %s) to %s",
                        sid,
                        epoch,
                        fp,
                    )
        metrics["holdout_gene_pearson_mean"] = float(np.nanmean(corrs_all)) if corrs_all else 0.0
        metrics["holdout_gene_pearson_per_slide_mean"] = corrs_per_slide
        holdout_pooled = _pooled_gene_corr_metrics(holdout_gene_stats)
        metrics["holdout_gene_pooled_mean"] = holdout_pooled["mean"]
        metrics["holdout_gene_pooled_median"] = holdout_pooled["median"]
        metrics["holdout_gene_pooled_max"] = holdout_pooled["max"]
        metrics["holdout_gene_pooled_p95"] = holdout_pooled["p95"]
        metrics["holdout_gene_pooled_n_genes"] = holdout_pooled["n_genes"]
        if holdout_per_gene:
            metrics["holdout_pearson_per_gene"] = holdout_per_gene
        if holdout_per_gene_files:
            metrics["holdout_pearson_per_gene_files"] = holdout_per_gene_files

    per_gene_files = {}
    gene_pcc_distribution_per_slide = {}
    if compute_per_gene and per_gene_stats:
        if per_gene_dir:
            os.makedirs(per_gene_dir, exist_ok=True)
        per_gene_summary = {}
        for sid, stats in per_gene_stats.items():
            c = stats["count"]
            denom_x = stats["sum_pred2"] - (stats["sum_pred"] ** 2) / np.maximum(c, 1e-8)
            denom_y = stats["sum_targ2"] - (stats["sum_targ"] ** 2) / np.maximum(c, 1e-8)
            num = stats["sum_xy"] - (stats["sum_pred"] * stats["sum_targ"]) / np.maximum(c, 1e-8)
            denom = np.sqrt(np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0))
            corr = np.zeros_like(num)
            valid = c > 1
            corr[valid] = num[valid] / np.maximum(denom[valid], 1e-8)
            corr[~valid] = np.nan

            corr_vals = corr[np.isfinite(corr)]
            if corr_vals.size > 0:
                per_gene_summary[int(sid)] = {
                    "mean": float(np.mean(corr_vals)),
                    "median": float(np.median(corr_vals)),
                    "max": float(np.max(corr_vals)),
                    "p95": float(np.percentile(corr_vals, 95)),
                    "n_genes": int(corr_vals.size),
                }
            else:
                per_gene_summary[int(sid)] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "max": float("nan"),
                    "p95": float("nan"),
                    "n_genes": 0,
                }

            # Per-slide gene-PCC distribution summaries:
            # all genes + top-k SVG subsets ranked by fixed Giotto scores.
            dist_entry = {"all": _summarize_gene_pcc_distribution(corr)}
            rank_order = svg_rank_gene_indices_by_slide.get(int(sid))
            if rank_order is not None:
                rank_order = np.asarray(rank_order, dtype=np.int64).reshape(-1)
            for k_svg in svg_topk:
                key = f"svg{k_svg}"
                if rank_order is None or rank_order.size == 0:
                    dist_entry[key] = _summarize_gene_pcc_distribution(np.array([], dtype=np.float64))
                    continue
                idx_top = rank_order[: min(int(k_svg), int(rank_order.size))]
                idx_top = idx_top[(idx_top >= 0) & (idx_top < corr.shape[0])]
                if idx_top.size == 0:
                    dist_entry[key] = _summarize_gene_pcc_distribution(np.array([], dtype=np.float64))
                else:
                    dist_entry[key] = _summarize_gene_pcc_distribution(corr[idx_top])
            gene_pcc_distribution_per_slide[int(sid)] = dist_entry

            if per_gene_dir and epoch is not None:
                fp = os.path.join(
                    per_gene_dir, f"slide{sid}_epoch{epoch}_per_gene_pearson.csv"
                )
                df_corr = pd.DataFrame(
                    {"gene": gene_names, "pearson": corr, "n_cells": c}
                )
                df_corr.to_csv(fp, index=False)
                per_gene_files[int(sid)] = fp
                logging.info(
                    "Saved per-gene Pearson for slide %s (epoch %s) to %s",
                    sid,
                    epoch,
                    fp,
                )
        if per_gene_files:
            metrics["pearson_per_gene_files"] = per_gene_files
        if per_gene_summary:
            metrics["pearson_per_gene_summary_per_slide"] = per_gene_summary
        if gene_pcc_distribution_per_slide:
            metrics["gene_pcc_distribution_per_slide"] = gene_pcc_distribution_per_slide
        pooled_gene = _pooled_gene_corr_metrics(per_gene_stats)
        metrics["pearson_gene_pooled_mean"] = pooled_gene["mean"]
        metrics["pearson_gene_pooled_median"] = pooled_gene["median"]
        metrics["pearson_gene_pooled_max"] = pooled_gene["max"]
        metrics["pearson_gene_pooled_p95"] = pooled_gene["p95"]
        metrics["pearson_gene_pooled_n_genes"] = pooled_gene["n_genes"]

    if total_cells > 0:
        metrics["ct_prop_gt"] = (ct_counts / total_cells).tolist()
        metrics["ct_prop_pred"] = (pred_counts / total_cells).tolist()
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_per_class = np.divide(
                correct_counts,
                np.maximum(ct_counts, 1),
            )
        metrics["ct_accuracy_per_class"] = acc_per_class.tolist()
        supported = ct_counts > 0
        metrics["ct_accuracy_micro"] = float(correct_counts.sum() / total_cells)
        metrics["ct_accuracy_macro"] = (
            float(np.mean(acc_per_class[supported])) if supported.any() else 0.0
        )
    else:
        zero_list = [0.0 for _ in range(n_classes)]
        metrics["ct_prop_gt"] = zero_list
        metrics["ct_prop_pred"] = zero_list
        metrics["ct_accuracy_per_class"] = zero_list
        metrics["ct_accuracy_micro"] = 0.0
        metrics["ct_accuracy_macro"] = 0.0
    # per-slide CT metrics
    ct_per_slide = {}
    for sid, counts in ct_counts_per.items():
        preds = pred_counts_per.get(sid, np.zeros_like(counts))
        correct = correct_counts_per.get(sid, np.zeros_like(counts))
        total = counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_per_class_slide = np.divide(correct, np.maximum(counts, 1))
        supported = counts > 0
        ct_per_slide[int(sid)] = {
            "prop_gt": ((counts / max(total, 1)).tolist() if total > 0 else [0.0 for _ in range(n_classes)]),
            "prop_pred": ((preds / max(preds.sum(), 1)).tolist() if preds.sum() > 0 else [0.0 for _ in range(n_classes)]),
            "acc_per_class": acc_per_class_slide.tolist(),
            "acc_micro": float(correct.sum() / max(total, 1)),
            "acc_macro": float(np.mean(acc_per_class_slide[supported])) if supported.any() else 0.0,
        }
    metrics["ct_per_slide"] = ct_per_slide

    model.train()
    return metrics




def main(config):
    opts = _to_namespace(json_file_to_pyobj(config.config_file))
    torch.autograd.set_detect_anomaly(False)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    device = get_device(config.gpu_id)

    eval_cfg = _to_namespace(getattr(opts, "evaluation", None)) or SimpleNamespace()
    if hasattr(opts, "data") and opts.data is not None:
        if not hasattr(opts.data, "punch_select_enabled"):
            opts.data.punch_select_enabled = True
        if not hasattr(opts.data, "punch_filter_splits"):
            opts.data.punch_filter_splits = "train"
        if not hasattr(opts.data, "roi_size_um") and not hasattr(opts.data, "broadcast_window_um"):
            opts.data.roi_size_um = 1000.0
        if not hasattr(opts.data, "pixel_size_um"):
            opts.data.pixel_size_um = 0.2125

    if not hasattr(opts, "model") or opts.model is None:
        opts.model = SimpleNamespace()
    if not hasattr(opts.model, "ecrm") or opts.model.ecrm is None:
        opts.model.ecrm = SimpleNamespace()

    if getattr(opts, "strict_method", None) is not None:
        logging.warning(
            "Config field 'strict_method' is deprecated and ignored; leak guards are always enabled."
        )

    # Leak guards are unconditional: fit statistics on train/val only and never
    # use GT cell types in reference weighting during validation/test-style paths.
    trainval_only_stats = True
    opts.model.use_gt_ct_ref_weights = False
    opts.model.ecrm.use_gt_ct = False
    if not getattr(opts.model, "refiner_type", None):
        opts.model.refiner_type = "mta"
    if not hasattr(opts.model.ecrm, "depth"):
        opts.model.ecrm.depth = 2
    if not hasattr(opts.model.ecrm, "cross_patch"):
        opts.model.ecrm.cross_patch = True
    opts.model.ecrm.cross_patch_k = int(
        getattr(opts.model.ecrm, "cross_patch_k", getattr(opts.model.ecrm, "k_target", 12))
    )
    opts.model.ecrm.graph_k = int(
        getattr(opts.model.ecrm, "graph_k", getattr(opts.model.ecrm, "k_target", 12))
    )
    opts.model.ecrm.edge_dropout = float(getattr(opts.model.ecrm, "edge_dropout", 0.05))
    opts.model.ecrm.message_dropout = float(getattr(opts.model.ecrm, "message_dropout", 0.05))
    opts.model.ecrm.residual_gate_init = float(
        getattr(opts.model.ecrm, "residual_gate_init", -1.4)
    )

    this_dir = os.path.abspath(os.path.dirname(__file__))
    nature_root = this_dir
    output_root_override = os.environ.get("OUTPUT_ROOT")
    if output_root_override:
        output_root = os.path.abspath(os.path.expanduser(output_root_override))
    else:
        output_root = nature_root
    default_results_dir = os.path.join(output_root, "results")
    default_metrics_dir = os.path.join(output_root, "metrics")
    os.makedirs(default_results_dir, exist_ok=True)
    os.makedirs(default_metrics_dir, exist_ok=True)

    if not hasattr(opts, "experiment_dirs") or opts.experiment_dirs is None:
        opts.experiment_dirs = SimpleNamespace()
    cfg_load_dir = getattr(opts.experiment_dirs, "load_dir", None)
    cfg_metrics_dir = getattr(opts.experiment_dirs, "metrics_dir", None)
    if not getattr(opts.experiment_dirs, "load_dir", None):
        opts.experiment_dirs.load_dir = default_results_dir
    if not getattr(opts.experiment_dirs, "model_dir", None):
        opts.experiment_dirs.model_dir = "models"
    if cfg_load_dir and os.path.abspath(cfg_load_dir) != default_results_dir:
        logging.warning(
            "Ignoring experiment_dirs.load_dir=%s; outputs are pinned to %s",
            cfg_load_dir,
            default_results_dir,
        )
    if cfg_metrics_dir and os.path.abspath(cfg_metrics_dir) != default_metrics_dir:
        logging.warning(
            "Ignoring experiment_dirs.metrics_dir=%s; metrics are pinned to %s",
            cfg_metrics_dir,
            default_metrics_dir,
        )
    opts.experiment_dirs.load_dir = default_results_dir
    opts.experiment_dirs.metrics_dir = default_metrics_dir
    metrics_dir = default_metrics_dir
    os.makedirs(metrics_dir, exist_ok=True)

    cache_root_override = os.environ.get("CACHE_ROOT")
    if cache_root_override:
        cache_root = os.path.abspath(os.path.expanduser(cache_root_override))
    else:
        cache_root = default_results_dir
    os.environ["CACHE_ROOT"] = cache_root
    logging.info(
        "Path policy: forcing outputs under %s and caches under %s",
        output_root,
        cache_root,
    )
    os.makedirs(cache_root, exist_ok=True)

    # Create experiment directories
    if config.resume_epoch != 0:
        make_new = False
    else:
        make_new = True

    timestamp = get_experiment_id(make_new, opts.experiment_dirs.load_dir, config.fold_id)
    timestamp_override = os.environ.get("RUN_ID")
    if timestamp_override:
        if os.path.isabs(timestamp_override):
            timestamp = os.path.basename(os.path.normpath(timestamp_override))
            logging.warning(
                "Ignoring absolute RUN_ID=%s; using run name %s under %s",
                timestamp_override,
                timestamp,
                default_results_dir,
            )
        else:
            timestamp = timestamp_override

    if os.path.isabs(timestamp):
        experiment_path_abs = os.path.abspath(timestamp)
        try:
            inside_results = (
                os.path.commonpath([experiment_path_abs, default_results_dir]) == default_results_dir
            )
        except ValueError:
            inside_results = False
        if inside_results:
            experiment_path = experiment_path_abs
        else:
            experiment_path = os.path.join(
                default_results_dir, os.path.basename(os.path.normpath(experiment_path_abs))
            )
            logging.warning(
                "Ignoring absolute experiment path outside %s; using %s",
                default_results_dir,
                experiment_path,
            )
    else:
        experiment_path = os.path.join(default_results_dir, timestamp)

    per_gene_dir = os.path.join(experiment_path, "per_gene_pearson")
    os.makedirs(experiment_path + "/" + opts.experiment_dirs.model_dir, exist_ok=True)
    os.makedirs(per_gene_dir, exist_ok=True)

    # Save copy of current config file
    shutil.copyfile(
        config.config_file, experiment_path + "/" + os.path.basename(config.config_file)
    )

    run_meta = {
        "config_file": os.path.abspath(config.config_file),
        "fold_id": int(config.fold_id),
        "gpu_id": int(config.gpu_id),
        "resume_epoch": int(config.resume_epoch),
        "experiment_path": os.path.abspath(experiment_path),
        "metrics_dir": metrics_dir,
        "data_policy": {
            "stats_fit_sources": "trainval_only",
            "gene_union_sources": "all_sources",
            "use_gt_ct_ref_weights": False,
            "ecrm_use_gt_ct": False,
        },
        "evaluation": _to_serialisable(eval_cfg),
    }
    _write_json(os.path.join(metrics_dir, "run_meta.json"), run_meta)

    # Set up the model
    logging.info("Initialising model")

    use_avgexp = opts.comps.avgexp
    use_celltype = opts.comps.celltype
    use_neighb = opts.comps.neighb if use_celltype else False
    avgexp_domain_specific = bool(
        getattr(getattr(opts, "model", SimpleNamespace()), "avgexp_domain_specific", False)
    )

    immune_class_indices = []
    immune_label_whitelist = {
        "b",
        "t",
        "plasma",
        "macrophage",
        "myeloid (excluding macrophage)",
        "myeloid",
    }
    if use_celltype:
        classes = opts.data.cell_types
        n_classes = len(classes)
        class_weights_np = None
        immune_class_indices = [
            idx
            for idx, name in enumerate(classes)
            if str(name).strip().lower() in immune_label_whitelist
        ]
        print(classes)
        print(f"Num cell types {n_classes}")
    else:
        n_classes = 0
        classes = []
        class_weights_np = None

    # Build union gene list from all train/val (and test) sources and impute missing genes.
    def _ensure_list(sources):
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        return [_to_namespace(json_file_to_pyobj(src)) if isinstance(src, str) else _to_namespace(src) for src in sources]

    sources_trainval = _ensure_list(getattr(opts, "data_sources_train_val", []))
    sources_test = _ensure_list(getattr(opts, "data_sources_test", []))
    all_sources = sources_trainval + sources_test
    stats_sources = sources_trainval
    logging.info(
        "Source policy: fit_sources=trainval_only (fit_sources=%d, impute_targets=%d, test_sources=%d)",
        len(stats_sources),
        len(all_sources),
        len(sources_test),
    )

    # Build the union panel from all known sources, but fit all statistics on
    # train/val sources only to avoid leaking test expression values.
    gene_union = set()
    expr_per_source = {}
    for src in all_sources:
        df_expr_tmp = pd.read_csv(src.fp_expr, index_col=0)
        gene_union.update(df_expr_tmp.columns.tolist())
    for src in stats_sources:
        df_expr_tmp = pd.read_csv(src.fp_expr, index_col=0)
        expr_per_source[src.fp_expr] = df_expr_tmp
    gene_names = natsort.natsorted(gene_union)
    excluded_paths = {str(getattr(src, "fp_expr", "")) for src in sources_test}
    overlap = excluded_paths.intersection(set(expr_per_source.keys()))
    assert len(overlap) == 0, "Leakage guard failed: test source used for stats fitting."
    logging.info(
        "Leakage guard active: excluded %d test expression source(s) from union/statistics fit.",
        len(excluded_paths),
    )

    holdout_n_genes = int(getattr(opts.training, "holdout_n_genes", 20))
    holdout_seed = int(getattr(opts.training, "holdout_seed", 0))
    if holdout_n_genes < 0:
        holdout_n_genes = 0

    holdout_n_genes_eval = 0
    logging.info("Holdout eval disabled; training uses all measured genes.")

    panel_hide_frac = float(
        getattr(
            opts.training,
            "panel_hide_frac",
            0.30 if holdout_n_genes > 0 else 0.0,
        )
    )
    panel_use_natural_missing = bool(
        getattr(opts.training, "panel_use_natural_missing", False)
    )
    panel_completion_enabled = bool(
        (holdout_n_genes > 0)
        or (panel_hide_frac > 0.0)
        or panel_use_natural_missing
    )
    panel_completion_loss_weight = float(
        getattr(
            opts.training,
            "panel_completion_loss_weight",
            1.0 if panel_completion_enabled else 0.0,
        )
    )
    panel_hidden_dim = 256
    panel_dropout = 0.0
    panel_use_morph = True
    panel_detach_morph = False
    panel_copy_observed = True
    panel_train_on_holdout = False
    panel_hide_in_forward = False
    panel_morph_gate_init = -2.0
    logging.info(
        "Panel completion: enabled=%s natural_missing=%s loss_w=%.3f hide_frac=%.2f hidden=%d use_morph=%s",
        panel_completion_enabled,
        panel_use_natural_missing,
        panel_completion_loss_weight,
        panel_hide_frac,
        panel_hidden_dim,
        panel_use_morph,
    )

    # Per-gene global means from available values
    gene_means = {}
    for g in gene_names:
        vals = []
        for df_expr_tmp in expr_per_source.values():
            if g in df_expr_tmp.columns:
                v = df_expr_tmp[g].to_numpy()
                if v.size > 0:
                    vals.append(v)
        gene_means[g] = float(np.mean(np.concatenate(vals))) if vals else 0.0
    gene_means_series = pd.Series(gene_means)
    gene_means_vec = gene_means_series.to_numpy()
    use_expr_baseline = bool(getattr(opts.training, "use_expr_baseline", False))
    gene_weight_max = float(getattr(opts.training, "gene_weight_max", 5.0))
    gene_weight_min = float(getattr(opts.training, "gene_weight_min", 0.5))
    # Per-gene variance weights to emphasize informative genes
    try:
        vals_all = []
        for df_expr_tmp in expr_per_source.values():
            vals_all.append(df_expr_tmp.reindex(columns=gene_names).to_numpy(dtype=np.float32))
        if vals_all:
            expr_concat = np.concatenate(vals_all, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                gene_var = np.nanvar(expr_concat, axis=0)
            finite_var_mask = np.isfinite(gene_var)
            if finite_var_mask.any():
                mean_var = float(np.mean(gene_var[finite_var_mask]))
                if not np.isfinite(mean_var) or mean_var <= 0.0:
                    mean_var = 1.0
                gene_var = gene_var / max(mean_var, 1e-8)
            else:
                gene_var = np.ones(len(gene_names), dtype=np.float32)
            neutral_fill_count = int((~np.isfinite(gene_var)).sum())
            gene_var = np.where(np.isfinite(gene_var), gene_var, 1.0)
            gene_var = np.clip(gene_var, gene_weight_min, gene_weight_max)
            gene_weights_torch = torch.from_numpy(gene_var.astype(np.float32)).to(device)
            logging.info(
                "Using per-gene variance weights; min %.3f max %.3f mean %.3f neutral_filled=%d/%d",
                float(gene_var.min()),
                float(gene_var.max()),
                float(gene_var.mean()),
                neutral_fill_count,
                len(gene_names),
            )
        else:
            gene_weights_torch = None
    except Exception as exc:
        logging.warning("Failed to compute gene variance weights: %s", exc)
        gene_weights_torch = None
    if use_expr_baseline:
        baseline_torch = torch.from_numpy(gene_means_vec.astype(np.float32)).float().to(device)
        logging.info("Using per-gene baseline for delta training (breast_all)")
    else:
        baseline_torch = None

    holdout_genes_by_slide = {}
    holdout_mask_by_slide = {}

    # Cell-type specific means for imputing missing genes (fallback to global means)
    ct_means = None
    ct_means_fallback = None
    ct_series_map = {}
    if use_celltype:
        n_classes_local = len(classes)
        ct_sums = np.zeros((n_classes_local, len(gene_names)), dtype=np.float64)
        ct_counts_arr = np.zeros((n_classes_local, len(gene_names)), dtype=np.int64)

        def _load_ct_series(fp_ct):
            if fp_ct is None or not os.path.isfile(fp_ct):
                return None
            df_ct = pd.read_csv(fp_ct, index_col="c_id")
            ct_numeric = pd.to_numeric(df_ct["ct"], errors="coerce")
            is_all_numbers = ct_numeric.notna().all()
            unassigned_idx = (
                classes.index("Unassigned")
                if any(str(c).strip().lower() == "unassigned" for c in classes)
                else (len(classes) - 1)
            )
            if not is_all_numbers:
                ct_dict = {name: idx for idx, name in enumerate(classes)}
                mapped = (
                    df_ct["ct"]
                    .astype(str)
                    .str.strip()
                    .map(ct_dict)
                    .fillna(unassigned_idx)
                    .astype(int)
                )
                return mapped
            # numeric: accept either 0..K-1 or 1..K coding
            ct_vals = ct_numeric.astype(int)
            if ct_vals.min() >= 1 and ct_vals.max() <= len(classes):
                ct_vals = ct_vals - 1
            ct_vals = ct_vals.clip(lower=0, upper=len(classes) - 1)
            return ct_vals

        # Decide which genes to hold out per train/val slide (measured genes only)
        # NOTE: This is only used for explicit holdout-vs-GT evaluation.
        if holdout_n_genes_eval > 0:
            def _svg_scores_for_slide(src, df_expr_local, present_genes):
                """
                Approximate Giotto-style SVG ranking via Moran's I on a kNN graph.

                Uses a downsampled nuclei segmentation label map if available
                (he_image_nuclei_seg_microns.tif in the same folder as fp_nuc_seg).
                Returns Moran scores aligned to present_genes, or None if unavailable.
                """
                try:
                    fp_seg = getattr(src, "fp_nuc_seg", None)
                    if not fp_seg:
                        return None
                    fp_microns = os.path.join(
                        os.path.dirname(fp_seg), "he_image_nuclei_seg_microns.tif"
                    )
                    if not os.path.isfile(fp_microns):
                        return None

                    cell_ids_all = df_expr_local.index.to_numpy(dtype=np.int64)
                    if cell_ids_all.size == 0:
                        return None

                    # Deterministic sampling for speed.
                    rng = np.random.default_rng(
                        holdout_seed
                        + int(getattr(src, "slide_idx", 0)) * 10007
                        + int(getattr(src, "domain_id", 0)) * 1009
                    )
                    sample_cap = min(3000, cell_ids_all.size)
                    sample_ids = (
                        rng.choice(cell_ids_all, size=sample_cap, replace=False)
                        if sample_cap < cell_ids_all.size
                        else cell_ids_all
                    )
                    kept_ids, coords_yx = _centroids_from_label_image(
                        fp_microns, sample_ids, chunk_rows=256
                    )
                    if kept_ids.size < 100:
                        return None

                    expr_counts = df_expr_local.loc[kept_ids, present_genes].to_numpy(
                        dtype=np.float32, copy=False
                    )
                    expr_model = np.log1p(np.clip(expr_counts, 0.0, None)) * float(
                        opts.data.expr_scale
                    )
                    return _morans_many(expr_model, coords_yx, k=8)
                except Exception as exc:
                    logging.warning(
                        "SVG Moran ranking failed for slide %s: %s",
                        getattr(src, "slide_idx", "na"),
                        exc,
                    )
                    return None

            for src in sources_trainval:
                df_expr_tmp = expr_per_source[src.fp_expr]
                present = [g for g in df_expr_tmp.columns.tolist() if g in gene_names]
                if not present:
                    continue
                # Choose genes with real variation on this slide so evaluation is meaningful.
                # Use the same expression scale as training targets: expr_scale * log1p(counts).
                ct_series_tmp = _load_ct_series(getattr(src, "fp_cell_type", None))
                idx = (
                    df_expr_tmp.index.intersection(ct_series_tmp.index)
                    if ct_series_tmp is not None
                    else df_expr_tmp.index
                )
                if idx.empty:
                    continue
                expr_arr = df_expr_tmp.loc[idx, present].to_numpy(dtype=np.float64)
                expr_target = np.log1p(np.clip(expr_arr, 0.0, None)) * float(
                    opts.data.expr_scale
                )
                var = np.var(expr_target, axis=0)
                nonzero = (expr_arr > 0).sum(axis=0)

                # Prefer "top SVG" holdout genes but avoid extremely sparse genes.
                # We target a reasonable detection rate first (>=5% nonzero), then relax if needed.
                n_cells = int(expr_arr.shape[0])
                min_nonzero = max(5, int(0.05 * n_cells))
                cand_mask = (var > 0.0) & (nonzero >= min_nonzero)
                if int(cand_mask.sum()) < holdout_n_genes_eval:
                    min_nonzero = max(5, int(0.01 * n_cells))
                    cand_mask = (var > 0.0) & (nonzero >= min_nonzero)
                if int(cand_mask.sum()) < holdout_n_genes_eval:
                    cand_mask = (var > 0.0) & (nonzero > 0)
                    min_nonzero = int(nonzero[cand_mask].min()) if int(cand_mask.sum()) else 0

                svg_scores = _svg_scores_for_slide(src, df_expr_tmp, present)
                chosen = []
                if svg_scores is not None and len(svg_scores) == len(present):
                    order = np.argsort(-svg_scores)
                    for j in order:
                        if cand_mask[j]:
                            chosen.append(present[int(j)])
                            if len(chosen) >= holdout_n_genes_eval:
                                break
                    # If still short, fill remaining from SVG order without the detection filter.
                    if len(chosen) < holdout_n_genes_eval:
                        for j in order:
                            g = present[int(j)]
                            if g not in chosen:
                                chosen.append(g)
                                if len(chosen) >= holdout_n_genes_eval:
                                    break
                else:
                    # Fallback: pick high-variance genes with detection filtering.
                    order = np.argsort(-var)
                    for j in order:
                        if cand_mask[j]:
                            chosen.append(present[int(j)])
                            if len(chosen) >= holdout_n_genes_eval:
                                break
                    if len(chosen) < holdout_n_genes_eval:
                        for j in order:
                            g = present[int(j)]
                            if g not in chosen:
                                chosen.append(g)
                                if len(chosen) >= holdout_n_genes_eval:
                                    break

                chosen = [str(g) for g in chosen[:holdout_n_genes_eval]]
                chosen = sorted(set(chosen), key=lambda x: gene_names.index(x))
                slide_id = int(getattr(src, "slide_idx", -1))
                holdout_genes_by_slide[slide_id] = chosen
                m = np.zeros(len(gene_names), dtype=np.float32)
                for g in chosen:
                    m[gene_names.index(g)] = 1.0
                holdout_mask_by_slide[slide_id] = m
                logging.info(
                    "Holdout slide %s: %d genes (top_SVG=True min_nonzero=%d seed=%d)",
                    slide_id,
                    len(chosen),
                    min_nonzero,
                    holdout_seed,
                )
                logging.info(
                    "Holdout genes slide %s: %s",
                    slide_id,
                    ", ".join(chosen),
                )

        # Cache cell-type series for all sources; statistics fit must use stats_sources only.
        for src in all_sources:
            ct_series_tmp = _load_ct_series(getattr(src, "fp_cell_type", None))
            if ct_series_tmp is None:
                continue
            ct_series_map[getattr(src, "fp_expr", "")] = ct_series_tmp
            if src not in stats_sources:
                continue
            df_expr_tmp = expr_per_source[src.fp_expr].reindex(columns=gene_names)
            idx = df_expr_tmp.index.intersection(ct_series_tmp.index)
            if idx.empty:
                continue
            expr_arr = df_expr_tmp.loc[idx].to_numpy(dtype=np.float64)
            ct_arr = ct_series_tmp.loc[idx].to_numpy(dtype=np.int64)
            valid_mask = np.isfinite(expr_arr)
            for ct_val in np.unique(ct_arr):
                if ct_val < 0 or ct_val >= n_classes_local:
                    continue
                rows = ct_arr == ct_val
                if not rows.any():
                    continue
                expr_rows = expr_arr[rows]
                valid_rows = valid_mask[rows]
                ct_sums[ct_val] += np.nansum(np.where(valid_rows, expr_rows, 0.0), axis=0)
                ct_counts_arr[ct_val] += valid_rows.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ct_means = np.divide(ct_sums, ct_counts_arr, where=ct_counts_arr > 0)
        # fallback to global means where a class never had a value for a gene
        ct_means_fallback = np.where(
            ct_counts_arr > 0,
            ct_means,
            np.broadcast_to(gene_means_vec, ct_means.shape),
        )

    # Build avgexp priors in target scale (expr_scale * log1p(counts)).
    avgexp_df_by_slide = {}
    if use_avgexp and use_celltype and classes:
        avgexp_df_by_slide = build_avgexp_df_by_slide(
            all_sources,
            stats_sources,
            gene_names,
            classes,
            float(opts.data.expr_scale),
            holdout_mask_by_slide=holdout_mask_by_slide,
            expr_per_source=expr_per_source,
            domain_specific=avgexp_domain_specific,
        )

        logging.info(
            "Built %savgexp priors for %d slide(s) (shape %dx%d)",
            "domain-specific " if avgexp_domain_specific else "global ",
            len(avgexp_df_by_slide),
            len(classes),
            len(gene_names),
        )

    gene_union_hash = hashlib.md5(",".join(gene_names).encode("utf-8")).hexdigest()[:8]
    impute_dir = os.path.join(cache_root, f"imputed_{gene_union_hash}")
    os.makedirs(impute_dir, exist_ok=True)
    force_reimpute = str(os.environ.get("FORCE_REIMPUTE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    def _impute_and_save(src_obj, kind="trainval"):
        if isinstance(src_obj, SimpleNamespace):
            src = SimpleNamespace(**src_obj.__dict__)
        elif isinstance(src_obj, dict):
            src = SimpleNamespace(**src_obj)
        else:
            # fall back to attribute lookup
            src = SimpleNamespace(
                slide_idx=getattr(src_obj, "slide_idx", -1),
                domain_id=getattr(src_obj, "domain_id", 0),
                fp_avgexp=getattr(src_obj, "fp_avgexp", None),
                fp_expr=getattr(src_obj, "fp_expr", None),
                fp_cell_type=getattr(src_obj, "fp_cell_type", None),
                fp_nuc_seg=getattr(src_obj, "fp_nuc_seg", None),
                fp_hist=getattr(src_obj, "fp_hist", None),
                fp_nuc_sizes=getattr(src_obj, "fp_nuc_sizes", None),
            )
        if not hasattr(src, "domain_id"):
            src.domain_id = 0
        slide_id_local = int(getattr(src, "slide_idx", -1))

        expr_out = os.path.join(
            impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_expr.csv"
        )
        mask_out = os.path.join(
            impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_mask.npy"
        )
        # Fast restart path: reuse existing imputed files to keep dataset-cache mtimes stable.
        can_reuse = (not force_reimpute) and os.path.isfile(expr_out) and os.path.isfile(mask_out)
        if can_reuse:
            src.fp_expr = expr_out
            src.fp_mask = mask_out
            df_ref_cached = (
                avgexp_df_by_slide.get(slide_id_local)
                if use_avgexp and use_celltype
                else None
            )
            logging.info(
                "Reusing imputed cache for slide=%s kind=%s: %s",
                slide_id_local,
                str(kind),
                expr_out,
            )
            return src, df_ref_cached

        # expr
        src_expr_key = src.fp_expr
        df_expr = pd.read_csv(src_expr_key, index_col=0)
        missing_expr = [g for g in gene_names if g not in df_expr.columns]
        mask_vec = np.ones(len(gene_names), dtype=np.float32)
        for g in missing_expr:
            mask_vec[gene_names.index(g)] = 0.0
        # Hold out measured genes (evaluate imputation on them)
        if kind == "trainval":
            for g in holdout_genes_by_slide.get(slide_id_local, []):
                if g in gene_names:
                    mask_vec[gene_names.index(g)] = 0.0
        # Reindex to union genes
        df_expr = df_expr.reindex(columns=gene_names)
        # Fill missing with CT-specific means where available, else global means
        if use_celltype and ct_means_fallback is not None:
            ct_series = ct_series_map.get(src_expr_key)
            df_expr_np = df_expr.to_numpy(dtype=np.float32)
            missing_mask_np = ~np.isfinite(df_expr_np)
            if ct_series is not None:
                ct_aligned = ct_series.reindex(df_expr.index)
                ct_arr = ct_aligned.to_numpy(dtype=np.float32)
                fill_vals = np.broadcast_to(gene_means_vec, df_expr_np.shape).copy()
                if ct_means_fallback is not None:
                    for ct_val in np.unique(ct_arr[np.isfinite(ct_arr)]):
                        ct_int = int(ct_val)
                        if 0 <= ct_int < ct_means_fallback.shape[0]:
                            rows = ct_arr == ct_int
                            if rows.any():
                                fill_vals[rows] = ct_means_fallback[ct_int]
                df_expr_np[missing_mask_np] = fill_vals[missing_mask_np]
                df_expr = pd.DataFrame(df_expr_np, index=df_expr.index, columns=df_expr.columns)
            else:
                df_expr = df_expr.fillna(gene_means_series).copy()
        else:
            df_expr = df_expr.fillna(gene_means_series).copy()
        # Track how many genes were actually present
        present_frac = 1.0 - (mask_vec == 0).mean()
        logging.debug(
            "Slide %s kind %s: %d/%d genes present (%.2f%%)",
            getattr(src, "slide_idx", "na"),
            kind,
            int(mask_vec.sum()),
            len(mask_vec),
            present_frac * 100,
        )
        # ensure indices stay numeric to match nuclei IDs
        try:
            df_expr.index = df_expr.index.astype(int)
        except Exception:
            pass
        df_expr.to_csv(expr_out)
        src.fp_expr = expr_out
        np.save(mask_out, mask_vec)
        src.fp_mask = mask_out

        if kind == "trainval":
            holdout_genes = holdout_genes_by_slide.get(slide_id_local)
            if holdout_genes:
                fp_hold = os.path.join(
                    impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_holdout_genes.txt"
                )
                with open(fp_hold, "w") as handle:
                    for g in holdout_genes:
                        handle.write(f"{g}\n")

        # avgexp
        df_ref = None
        if use_avgexp and use_celltype:
            df_ref = avgexp_df_by_slide.get(slide_id_local)
        return src, df_ref

    # Impute and rewrite sources
    imputed_trainval = []
    imputed_refs = []
    expr_ref_map = {}
    for src in sources_trainval:
        src_out, df_ref = _impute_and_save(src, kind="trainval")
        imputed_trainval.append(src_out)
        if df_ref is not None:
            imputed_refs.append(df_ref)
            expr_ref_map[src_out.slide_idx] = df_ref

    imputed_test = []
    for src in sources_test:
        src_out, df_ref = _impute_and_save(src, kind="test")
        imputed_test.append(src_out)
        if df_ref is not None:
            expr_ref_map[src_out.slide_idx] = df_ref

    # Use local imputed source lists
    train_sources = imputed_trainval
    test_sources = imputed_test

    # Optional global cell-coordinate maps per slide for cross-patch graph construction.
    slide_coord_map_by_slide = {}
    for src in (train_sources + test_sources):
        sid = int(getattr(src, "slide_idx", -1))
        if sid in slide_coord_map_by_slide:
            continue
        cmap = _load_histology_coord_map_from_source(src)
        if cmap:
            slide_coord_map_by_slide[sid] = cmap
    logging.info(
        "Loaded global cell-coordinate maps for %d slide(s).",
        len(slide_coord_map_by_slide),
    )
    if bool(getattr(opts.model.ecrm, "cross_patch", False)) and len(slide_coord_map_by_slide) == 0:
        logging.warning(
            "ECRM cross-patch requested but no global coordinate maps were found; "
            "graph will fall back to within-patch connectivity."
        )

    # Build per-slide avgexp references
    expr_ref_torch_map = {}
    if use_avgexp and imputed_refs:
        ref_counts = []
        ref_stack = []
        for slide_id, df_ref_tmp in expr_ref_map.items():
            df_aligned = df_ref_tmp.reindex(columns=gene_names)
            ref_counts.append(df_aligned.shape[0])
            # df_ref_tmp is already in target scale (expr_scale * log1p(counts))
            ref_np = df_aligned.to_numpy(dtype=np.float32)
            expr_ref_torch_map[slide_id] = torch.from_numpy(ref_np).float().to(device)
            ref_stack.append(ref_np)

        unique_counts = set(ref_counts)
        if len(unique_counts) != 1:
            raise ValueError(
                f"Avgexp references per slide differ: {unique_counts}. "
                "All slides must have the same number of refs for a shared model."
            )
        n_ref = ref_counts[0]
        if n_ref <= 0:
            raise ValueError("Avgexp references found but none valid (n_ref <= 0).")

        # Default ref: mean over slides, preserve ref dimension (n_ref, n_genes)
        ref_stack_arr = np.stack(ref_stack, axis=0)  # (n_slides, n_ref, n_genes)
        expr_ref_mean = np.nanmean(ref_stack_arr, axis=0)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        logging.info("Using avgexp with %d reference(s) per slide", n_ref)
    elif use_avgexp:
        # Fallback: no avgexp loaded, use zeros to avoid crash
        expr_ref_mean = np.zeros((1, len(gene_names)), dtype=np.float32)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        n_ref = 1
        logging.warning("Avgexp enabled but no references loaded; falling back to zeros.")
    else:
        n_ref = None
        expr_ref_torch = None

    n_genes = len(gene_names)
    print(f"{n_genes} genes (union)")

    fp_out = os.path.join(experiment_path, "genes.txt")
    with open(fp_out, "w") as f:
        for line in gene_names:
            f.write(f"{line}\n")

    # Best imputation: allow model to predict residual over avgexp prior
    if use_avgexp and hasattr(opts, "model"):
        try:
            if float(getattr(opts.model, "avgexp_residual_scale", 0.0)) <= 0.0:
                setattr(opts.model, "avgexp_residual_scale", 0.1)
                logging.info("Overriding model.avgexp_residual_scale=0.1 for imputation")
        except Exception:
            pass

    framework_name = f"{Framework.__module__}.{Framework.__name__}"
    try:
        model = Framework(
            n_classes,
            n_genes,
            opts.model.emb_dim,
            device,
            n_ref,
            use_avgexp,
            use_celltype,
            use_neighb,
            model_cfg=opts.model,
        )
        logging.info("Using %s (with model_cfg)", framework_name)
    except TypeError:
        model = Framework(
            n_classes,
            n_genes,
            opts.model.emb_dim,
            device,
            n_ref,
            use_avgexp,
            use_celltype,
            use_neighb,
        )
        logging.info("Using %s (no model_cfg)", framework_name)

    if panel_completion_enabled:
        model.completion_head = PanelCompletionHead(
            n_genes,
            hidden_dim=panel_hidden_dim,
            dropout=panel_dropout,
            use_morph=panel_use_morph,
            morph_gate_init=panel_morph_gate_init,
        )

    # Some frameworks accept cell-graph kwargs; detect once and avoid per-batch try/except.
    try:
        fwd_params = inspect.signature(model.forward).parameters
        supports_cell_graph = all(
            k in fwd_params for k in ("coords_cells", "cell_edge_index", "cell_patch_ids")
        )
    except Exception:
        supports_cell_graph = False
    logging.info("Cell-graph support: %s", supports_cell_graph)
    model.to(device)

    # Dataloader
    logging.info("Preparing data")

    regions_train = getattr(opts, "regions_train", None)
    train_regions = regions_train if regions_train is not None else getattr(opts, "regions_val", None)
    if train_regions is None:
        raise ValueError("No regions specified for training (expected regions_train or regions_val in config).")

    punch_enabled = bool(
        getattr(opts.data, "punch_select_enabled", False)
        or getattr(opts.data, "tma_select_enabled", False)
    )
    if punch_enabled:
        ecrm_cfg_punch = getattr(opts.model, "ecrm", None)
        punch_graph_k = (
            int(getattr(ecrm_cfg_punch, "graph_k", getattr(ecrm_cfg_punch, "k_target", 8)))
            if ecrm_cfg_punch is not None
            else 8
        )
        for src in train_sources:
            try:
                preselect_tma_punch_with_vq(
                    model,
                    src,
                    opts,
                    train_regions,
                    config.fold_id,
                    experiment_path,
                    device,
                    expr_ref_torch,
                    expr_ref_torch_map,
                    gene_names,
                    classes,
                    graph_k=punch_graph_k,
                )
            except Exception as exc:
                logging.warning(
                    "[punch] VQ TMA preselection failed for slide=%s: %s",
                    getattr(src, "slide_idx", "unknown"),
                    exc,
                )

    expr_ref_torch_val = expr_ref_torch
    expr_ref_torch_val_map = expr_ref_torch_map
    if use_avgexp and use_celltype and classes:
        fallback_val_df_by_slide = {}
        for src in sources_trainval:
            slide_id = int(getattr(src, "slide_idx", -1))
            df_ref = avgexp_df_by_slide.get(slide_id)
            if df_ref is not None:
                fallback_val_df_by_slide[slide_id] = df_ref
        avgexp_val_df_by_slide = _build_train_region_avgexp_df_by_slide(
            sources_trainval,
            train_regions,
            config.fold_id,
            gene_names,
            classes,
            float(opts.data.expr_scale),
            fallback_df_by_slide=fallback_val_df_by_slide,
            holdout_mask_by_slide=holdout_mask_by_slide,
            domain_specific=avgexp_domain_specific,
        )
        if avgexp_val_df_by_slide:
            ref_stack_val = []
            ref_counts_val = []
            expr_ref_torch_val_map = dict(expr_ref_torch_map)
            for slide_id, df_ref_tmp in avgexp_val_df_by_slide.items():
                df_aligned = df_ref_tmp.reindex(columns=gene_names)
                ref_counts_val.append(df_aligned.shape[0])
                ref_np = df_aligned.to_numpy(dtype=np.float32)
                expr_ref_torch_val_map[int(slide_id)] = torch.from_numpy(ref_np).float().to(device)
                ref_stack_val.append(ref_np)

            unique_val_counts = set(ref_counts_val)
            if len(unique_val_counts) != 1:
                logging.warning(
                    "Validation train-region avgexp refs differ in count %s; falling back to standard refs.",
                    sorted(unique_val_counts),
                )
                expr_ref_torch_val = expr_ref_torch
                expr_ref_torch_val_map = expr_ref_torch_map
            elif ref_stack_val:
                expr_ref_torch_val = torch.from_numpy(np.nanmean(np.stack(ref_stack_val, axis=0), axis=0)).float().to(device)
                logging.info(
                    "Validation avgexp refs use train-region-only statistics for %d train slide(s).",
                    len(ref_stack_val),
                )
        else:
            logging.warning(
                "Validation train-region avgexp refs could not be built; using standard refs.",
            )

    def _slide_batch_sampler(datasets, batch_size, interleave=False):
        sampler_seed = int(getattr(opts.training, "batch_sampler_seed", 0))
        shuffle_within_slide = bool(
            getattr(opts.training, "shuffle_within_slide_batches", True)
        )
        weighted_interleave = bool(
            getattr(opts.training, "weighted_interleave_slide_batches", True)
        )
        weight_cap = float(getattr(opts.training, "sampler_weight_cap", 3.0))
        offsets = []
        lengths = []
        acc = 0
        for ds in datasets:
            offsets.append(acc)
            l = len(ds)
            lengths.append(l)
            acc += l
        if interleave:
            logging.info(
                "Using interleaved slide batch sampler: seed=%d shuffle_within_slide=%s weighted=%s",
                sampler_seed,
                str(shuffle_within_slide),
                str(weighted_interleave),
            )
            slide_weights = None
            if weighted_interleave:
                slide_weights = []
                for ds in datasets:
                    slide_weights.append(getattr(ds, "patch_weights", None))
            return _InterleavedSlideBatchSampler(
                datasets,
                batch_size,
                seed=sampler_seed,
                shuffle_within_slide=shuffle_within_slide,
                shuffle_slide_order=True,
                slide_weights=slide_weights,
                weight_cap=weight_cap,
            )
        batches = []
        for off, l in zip(offsets, lengths):
            idxs = list(range(off, off + l))
            for i in range(0, l, batch_size):
                batches.append(idxs[i : i + batch_size])
        return batches

    immune_sampler_boost = float(getattr(opts.training, "immune_sampler_boost", 1.0))
    # Derive a boost automatically if none provided (>1 only when immune classes are rare)
    if immune_sampler_boost <= 1.0 and use_celltype:
        try:
            counts = np.zeros(n_classes, dtype=np.int64)
            ct_to_idx = {name: idx for idx, name in enumerate(classes)}
            for src in train_sources:
                df_ct_counts = pd.read_csv(
                    src.fp_cell_type, index_col="c_id"
                )["ct"].astype(str)
                ct_indices = df_ct_counts.map(lambda x: ct_to_idx.get(x, None))
                counts += np.bincount(
                    np.array([c for c in ct_indices if c is not None], dtype=int),
                    minlength=n_classes,
                )
            immune_counts = (
                counts[immune_class_indices] if immune_class_indices else np.array([])
            )
            if immune_counts.size > 0 and immune_counts.max() > 0:
                max_boost = float(getattr(opts.training, "sampler_weight_cap", 3.0))
                rare_ratio = immune_counts.max() / max(immune_counts.min(), 1)
                # bias toward modest bump; cap to keep sampling stable
                immune_sampler_boost = min(max_boost, max(1.0, rare_ratio))
                logging.info(
                    "Auto immune_sampler_boost=%.2f (rare_ratio=%.2f, cap=%.2f)",
                    immune_sampler_boost,
                    rare_ratio,
                    max_boost,
                )
        except Exception as exc:
            logging.warning("Failed to derive immune_sampler_boost automatically: %s", exc)

    train_datasets = []
    for src in train_sources:
        src_ns = src if isinstance(src, SimpleNamespace) else SimpleNamespace(**src)
        ds = DataProcessing(
            src_ns,
            opts.data,
            train_regions,
            opts.comps,
            opts.stain_norm,
            classes,
            gene_names,
            device,
            experiment_path,
            opts.training.stain_aug,
            config.fold_id,
            mode="train",
            immune_sampler_boost=immune_sampler_boost,
            immune_class_multipliers=None,
        )
        train_datasets.append(ds)

    # Apply adaptive immune balancing per slide before concatenation so each
    # slide's patch weights reflect its own class rarity profile.
    def _auto_immune_multipliers(ds, immune_idx, classes_all):
        if not immune_idx or not hasattr(ds, "df_ct") or "ct" not in ds.df_ct.columns:
            return {}
        try:
            ct_series = ds.df_ct["ct"].astype(int) - 1
            counts = np.bincount(
                ct_series.clip(lower=0), minlength=len(classes_all)
            ).astype(float)
            immune_counts = counts[immune_idx]
            if immune_counts.sum() <= 0:
                return {}
            props = immune_counts / immune_counts.sum()
            target = np.ones_like(props) / len(props)
            beta = float(getattr(opts.training, "sampler_multiplier_beta", 0.5))
            m_min = float(getattr(opts.training, "sampler_multiplier_min", 0.7))
            m_max = float(getattr(opts.training, "sampler_multiplier_max", 1.5))
            ratio = target / np.maximum(props, 1e-8)
            mult = np.power(ratio, beta)
            mult = np.clip(mult, m_min, m_max)
            keys = [idx + 1 for idx in immune_idx]
            multipliers = {k: float(v) for k, v in zip(keys, mult)}
            logging.info(
                "Auto immune multipliers for slide=%s (beta=%.2f, min=%.2f, max=%.2f): %s",
                getattr(ds, "slide_idx", "unknown"),
                beta,
                m_min,
                m_max,
                multipliers,
            )
            return multipliers
        except Exception as exc:
            logging.warning(
                "Failed to derive immune multipliers for slide=%s: %s",
                getattr(ds, "slide_idx", "unknown"),
                exc,
            )
            return {}

    for ds in train_datasets:
        immune_multipliers = _auto_immune_multipliers(
            ds, immune_class_indices, classes
        )
        if immune_multipliers:
            ds.set_immune_sampling_multipliers(immune_multipliers)
        ds.refresh_patch_sampling_weights(immune_sampler_boost)

    if len(train_datasets) == 0:
        raise ValueError("No training slides could be loaded (all skipped).")
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        use_batch_sampler = False
        batches = None
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        batches = _slide_batch_sampler(
            train_datasets,
            opts.training.batch_size,
            interleave=bool(getattr(opts.training, "interleave_slide_batches", True)),
        )
        use_batch_sampler = True

    sampler = None
    patch_weights_all = None
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        pw_list = []
        for ds in train_dataset.datasets:
            if getattr(ds, "patch_weights", None) is not None:
                pw_list.extend(ds.patch_weights)
        if pw_list:
            patch_weights_all = pw_list
    else:
        if getattr(train_dataset, "patch_weights", None) is not None:
            patch_weights_all = train_dataset.patch_weights

    if patch_weights_all is not None and not use_batch_sampler:
        try:
            weights = torch.as_tensor(patch_weights_all, dtype=torch.double)
            weight_cap = float(getattr(opts.training, "sampler_weight_cap", 3.0))
            if weight_cap > 0:
                weights = torch.clamp(weights, max=weight_cap)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )
            logging.info(
                "Using WeightedRandomSampler with cap %.2f (min %.4f, max %.4f)",
                weight_cap,
                float(weights.min()),
                float(weights.max()),
            )
        except Exception as exc:
            logging.warning("Falling back to shuffle dataloader (sampler init failed): %s", exc)
            sampler = None

    if use_batch_sampler:
        # batches is already a list of index lists; feed directly to DataLoader
        train_loader_kwargs = {
            "batch_sampler": batches,
            "num_workers": opts.data.num_workers,
            "drop_last": False,
            "pin_memory": getattr(opts.data, "pin_memory", False),
        }
    else:
        train_loader_kwargs = {
            "batch_size": opts.training.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "num_workers": opts.data.num_workers,
            "drop_last": True,
            "pin_memory": getattr(opts.data, "pin_memory", False),
        }
    # Keep workers alive and prefetch when using multiple workers
    if train_loader_kwargs["num_workers"] and train_loader_kwargs["num_workers"] > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = getattr(opts.data, "prefetch_factor", 2)

    dataloader = DataLoader(
        dataset=train_dataset,
        **train_loader_kwargs,
    )

    n_train_examples = len(dataloader)
    logging.info("Total number of training batches: %d" % n_train_examples)

    if use_celltype and class_weights_np is None:
        try:
            counts_total = np.zeros(n_classes, dtype=np.int64)
            datasets_iter = (
                train_dataset.datasets
                if isinstance(train_dataset, torch.utils.data.ConcatDataset)
                else [train_dataset]
            )
            for ds in datasets_iter:
                if hasattr(ds, "df_ct") and "ct" in ds.df_ct.columns:
                    ct_series = ds.df_ct["ct"].astype(int) - 1
                    counts = np.bincount(ct_series.clip(lower=0), minlength=n_classes)
                    counts_total += counts
            counts_total = np.maximum(counts_total, 1)
            inv = 1.0 / counts_total
            inv = inv / inv.mean()
            weight_cap = float(getattr(opts.training, "class_weight_cap", 5.0))
            inv = np.clip(inv, 0.0, weight_cap)
            class_weights_np = inv
            logging.info(
                "CT class weights (cap %.2f): %s",
                weight_cap,
                class_weights_np.tolist(),
            )
        except Exception as exc:
            logging.warning("Failed to compute class weights: %s", exc)

    ecrm_cfg = getattr(opts.model, "ecrm", None)
    if ecrm_cfg is None:
        graph_k = 8
        graph_cross_patch = False
        graph_cross_patch_k = 0
    else:
        graph_k = int(getattr(ecrm_cfg, "graph_k", getattr(ecrm_cfg, "k_target", 8)))
        graph_cross_patch = bool(getattr(ecrm_cfg, "cross_patch", False))
        graph_cross_patch_k = int(
            getattr(ecrm_cfg, "cross_patch_k", getattr(ecrm_cfg, "graph_k", graph_k))
        )
    graph_k = max(graph_k, 2)
    graph_cross_patch_k = max(graph_cross_patch_k, 1)

    slide_comp_target = None
    if use_celltype:
        try:
            if hasattr(train_dataset, "df_ct") and "ct" in train_dataset.df_ct.columns:
                slide_series = train_dataset.df_ct["ct"].astype(int) - 1
                counts = np.bincount(slide_series, minlength=n_classes)
                counts = np.maximum(counts, 0)
                total = counts.sum()
                if total > 0:
                    comp = counts / total
                    slide_comp_target = (
                        torch.from_numpy(comp).float().to(device)
                    )
                    logging.info(
                        "Slide GT composition: %s",
                        ", ".join(
                            f"{classes[idx]}:{comp[idx]:.4f}"
                            for idx in range(n_classes)
                    ),
                )
        except Exception as exc:
            logging.warning("Failed to read slide composition: %s", exc)

    def _make_eval_loader(src_list, regions, mode_name):
        datasets = []
        for src in src_list:
            src_ns = src if isinstance(src, SimpleNamespace) else SimpleNamespace(**src)
            ds = DataProcessing(
                src_ns,
                opts.data,
                regions,
                opts.comps,
                opts.stain_norm,
                classes,
                gene_names,
                device,
                experiment_path,
                False,
                config.fold_id,
                mode=mode_name,
                immune_sampler_boost=1.0,
                immune_class_multipliers=None,
            )
            datasets.append(ds)
        if not datasets:
            return None
        if len(datasets) == 1:
            dataset = datasets[0]
            kwargs = {
                "dataset": dataset,
                "batch_size": opts.training.batch_size,
                "shuffle": False,
                "num_workers": opts.data.num_workers,
                "drop_last": False,
                "pin_memory": getattr(opts.data, "pin_memory", False),
            }
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)
            kwargs = {
                "dataset": dataset,
                "num_workers": opts.data.num_workers,
                "batch_sampler": _slide_batch_sampler(
                    datasets,
                    opts.training.batch_size,
                    interleave=False,
                ),
                "drop_last": False,
                "pin_memory": getattr(opts.data, "pin_memory", False),
            }
        if opts.data.num_workers and opts.data.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = getattr(opts.data, "prefetch_factor", 2)
        return DataLoader(**kwargs)

    try:
        # Validation is computed on train/val sources.
        val_dataloader = _make_eval_loader(train_sources, opts.regions_val, mode_name="val")
    except Exception as exc:
        logging.warning("Validation loader creation failed: %s", exc)
        val_dataloader = None

    ext_regions = getattr(opts, "regions_test", None) or getattr(opts, "regions_val", None)
    try:
        external_dataloader = _make_eval_loader(test_sources, ext_regions, mode_name="val")
    except Exception as exc:
        logging.warning("External loader creation failed: %s", exc)
        external_dataloader = None

    # Fixed SVG ranks (Giotto-ranked) per slide for log-time top-k reporting.
    svg_topk = (20, 50)
    svg_knn_k = 8
    svg_sample_cap = 3000
    val_svg_rank_indices_by_slide = _compute_svg_rank_gene_indices_by_slide(
        sources_trainval,
        opts.regions_val,
        config.fold_id,
        mode_name="val",
        gene_names=gene_names,
        k_neighbors=svg_knn_k,
        sample_cap=svg_sample_cap,
    )
    ext_svg_rank_indices_by_slide = _compute_svg_rank_gene_indices_by_slide(
        sources_test,
        ext_regions,
        config.fold_id,
        mode_name="val",
        gene_names=gene_names,
        k_neighbors=svg_knn_k,
        sample_cap=svg_sample_cap,
    )
    logging.info(
        "Precomputed Giotto SVG ranks: val_slides=%d ext_slides=%d topk=%s kNN=%d sample_cap=%d",
        len(val_svg_rank_indices_by_slide),
        len(ext_svg_rank_indices_by_slide),
        list(svg_topk),
        svg_knn_k,
        svg_sample_cap,
    )

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training.learning_rate,
        betas=(opts.training.beta1, opts.training.beta2),
        weight_decay=opts.training.weight_decay,
        eps=opts.training.eps,
    )

    global_step = 0

    # Starting epoch
    if config.resume_epoch != 0:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch != 0:
        logging.info("Resume training")

        load_path = (
            experiment_path
            + "/"
            + opts.experiment_dirs.model_dir
            + "/epoch_%d_model.pth" % (config.resume_epoch)
        )
        checkpoint = torch.load(load_path)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            logging.warning("Strict state_dict load failed (%s); retrying with strict=False", exc)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        epoch = checkpoint["epoch"]
        print("Loaded " + load_path)

        model.to(device)

        try:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/epoch_%d_optim.pth" % (config.resume_epoch)
            )
            checkpoint = torch.load(load_path)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded " + load_path)
        except:
            print("Optimizer state dict not found")

    else:
        model.to(device)

    logging.info("Begin training")

    if class_weights_np is not None:
        class_weights_torch = torch.from_numpy(class_weights_np).float().to(device)
        logging.info("Using class weights (immune boost applied): %s", class_weights_np.tolist())
    else:
        class_weights_torch = None

    loss_map = nn.CrossEntropyLoss(reduction="mean")
    loss_ct_hist = nn.CrossEntropyLoss(
        weight=class_weights_torch, reduction="mean"
    )
    loss_expr_ct = nn.CrossEntropyLoss(
        weight=class_weights_torch, reduction="mean"
    )
    loss_expr_ct_embed = nn.CosineEmbeddingLoss(reduction="mean")
    loss_expr = nn.MSELoss(reduction="mean")
    loss_expr_immune = nn.MSELoss(reduction="mean")
    loss_expr_invasive = nn.MSELoss(reduction="mean")
    loss_logits = nn.MSELoss(reduction="mean")
    loss_comp_est = nn.KLDivLoss(reduction="batchmean")
    loss_comp_gt = nn.KLDivLoss(reduction="batchmean")

    zero_weight = float(getattr(opts.training, "zero_weight", 0.1))
    zero_threshold = float(getattr(opts.training, "zero_threshold", 0.0))
    # Correlation is carried by the explicit Pearson term below; keep the
    # reconstruction loss purely MSE-like here to avoid double-counting.
    corr_loss_weight = 0.0
    pearson_loss_weight = float(getattr(opts.training, "pearson_loss_weight", 1.0))
    expr_ct_embed_loss_weight = float(
        getattr(opts.training, "expr_ct_embed_loss_weight", 1.0)
    )
    logits_loss_weight = float(getattr(opts.training, "logits_loss_weight", 1.0))
    bulk_loss_weight = float(getattr(opts.training, "bulk_loss_weight", 0.0))
    logging.info(
        "Loss setup: corr=%.3f pearson=%.3f expr_ct_embed_w=%.3f logits_w=%.3f bulk=%.3f expr_ct_embed_internal=100 interleave_slide_batches=%s",
        corr_loss_weight,
        pearson_loss_weight,
        expr_ct_embed_loss_weight,
        logits_loss_weight,
        bulk_loss_weight,
        str(bool(getattr(opts.training, "interleave_slide_batches", True))),
    )
    gene_weights_torch = None

    def expr_loss_weighted(pred, target, mask=None):
        """
        Expression loss = zero-aware weighted MSE + correlation (shape) loss.
        - Down-weights near-zero targets so the model is not rewarded for predicting zeros.
        - Adds a correlation term to focus on the shape of the gene vector.
        Preserves existing masking for missing genes.
        """
        w_zero = pred.new_tensor(zero_weight)
        w_one = pred.new_tensor(1.0)
        w = torch.where(target > zero_threshold, w_one, w_zero)
        if mask is not None:
            w = w * mask
        if gene_weights_torch is not None and gene_weights_torch.numel() == pred.shape[1]:
            w = w * gene_weights_torch.view(1, -1)
        mse_num = ((pred - target) ** 2 * w).sum()
        mse_den = w.sum().clamp_min(1e-8)
        loss_mse_val = mse_num / mse_den

        loss_corr_val = masked_pearson(pred, target, mask)
        return loss_mse_val + corr_loss_weight * loss_corr_val

    def masked_mse(pred, target, mask):
        # expr_loss_weighted already normalises by sum of weights and handles masks.
        return expr_loss_weighted(pred, target, mask)

    def pseudo_bulk_mse(pred, target, mask, group_ids):
        """
        Compute MSE between group-level means (pseudo-bulk). Groups can be slide
        or slide+celltype to respect biological structure.
        """
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        uniq = torch.unique(group_ids)
        losses = []
        for gid in uniq:
            idx = group_ids == gid
            if mask is None:
                if not idx.any():
                    continue
                pred_mean = pred[idx].mean(dim=0)
                target_mean = target[idx].mean(dim=0)
                if gene_weights_torch is not None and gene_weights_torch.numel() == pred_mean.shape[0]:
                    gw = gene_weights_torch
                    mse_num = ((pred_mean - target_mean) ** 2 * gw).sum()
                    mse_den = gw.sum().clamp_min(1e-8)
                    loss_g = mse_num / mse_den
                else:
                    loss_g = F.mse_loss(pred_mean, target_mean, reduction="mean")
            else:
                gmask = mask[idx].float()
                valid = gmask.sum(dim=0)
                valid_mask = valid > 0
                if not valid_mask.any():
                    continue
                pred_mean = (pred[idx] * gmask).sum(dim=0) / valid.clamp_min(1.0)
                target_mean = (target[idx] * gmask).sum(dim=0) / valid.clamp_min(1.0)
                if gene_weights_torch is not None and gene_weights_torch.numel() == pred_mean.shape[0]:
                    gw = gene_weights_torch
                    mse_num = ((pred_mean - target_mean) ** 2 * gw).sum()
                    mse_den = gw.sum().clamp_min(1e-8)
                    loss_g = mse_num / mse_den
                else:
                    loss_g = F.mse_loss(
                        pred_mean[valid_mask], target_mean[valid_mask], reduction="mean"
                    )
            losses.append(loss_g)

        if not losses:
            return torch.tensor(0.0, device=pred.device)

        return torch.stack(losses).mean()

    def masked_var(x, mask):
        if mask is None:
            return torch.var(x, unbiased=False)
        m = mask.bool()
        if not m.any():
            return torch.tensor(0.0, device=x.device)
        vals = x[m]
        return torch.var(vals, unbiased=False)

    def masked_pearson(pred, target, mask, eps=1e-6):
        if mask is None:
            return pearson_loss(pred, target, eps=eps)
        mask = mask.float()
        valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred_center = (pred * mask - (pred * mask).sum(dim=1, keepdim=True) / valid)
        targ_center = (target * mask - (target * mask).sum(dim=1, keepdim=True) / valid)
        num = (pred_center * targ_center * mask).sum(dim=1)
        denom = (
            ((pred_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
            * ((targ_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
        ).clamp_min(eps)
        corr = num / denom
        return (1.0 - corr).mean()

    # losses_names = [
    #     "loss_epoch_expr",
    #     "loss_epoch_ct_hist",
    #     "loss_epoch_map",
    #     "loss_epoch_expr_ct",
    #     "loss_epoch_expr_immune",
    #     "loss_epoch_expr_invasive",
    #     "loss_epoch_expr_ct_embed",
    #     "loss_epoch_logits",
    #     "loss_epoch_comp_est",
    #     "loss_epoch_comp_gt"
    # ]
    # df_losses = pd.DataFrame(
    #     0.0, index=list(range(opts.training.total_epochs)), columns=losses_names
    # )

    total_epochs = opts.training.total_epochs
    ext_eval_every_epochs = max(1, int(getattr(eval_cfg, "external_every_epochs", 5)))
    ext_eval_final_epoch = bool(getattr(eval_cfg, "external_final_epoch", True))
    grad_clip_norm = float(getattr(opts.training, "grad_clip_norm", 1.0))
    var_ratio_limit = float(getattr(opts.training, "expr_var_ratio_limit", 30.0))
    parity_tolerance_abs = float(getattr(eval_cfg, "parity_tolerance_abs", 0.001))
    best_metric_eps = float(getattr(eval_cfg, "best_metric_eps", 1e-8))
    external_source_tag = str(getattr(eval_cfg, "external_source", "data_sources_test"))

    epoch_records = []
    best_val_gene_pooled = -float("inf")
    best_val_ct_macro = -float("inf")
    best_epoch = None
    best_ckpt_path = None
    kpi_csv_path = os.path.join(default_results_dir, "main_kpi_summary.csv")
    epoch_jsonl_path = os.path.join(metrics_dir, "epoch_metrics.jsonl")
    if initial_epoch == 0:
        for fp in (kpi_csv_path, epoch_jsonl_path):
            if os.path.isfile(fp):
                os.remove(fp)

    for epoch in range(initial_epoch, total_epochs):
        print(f"Epoch: {epoch+1}")
        # Encoder stays frozen for the entire run (no unfreeze step)
        model.train()
        if hasattr(model, "set_epoch_progress"):
            total_eps = max(total_epochs - 1, 1)
            model.set_epoch_progress(epoch / total_eps)

        optimizer.param_groups[0]["lr"] = opts.training.learning_rate * (
            1 - epoch / total_epochs
        )

        loss_epoch = 0
        loss_epoch_map = 0
        loss_epoch_ct_hist = 0
        loss_epoch_expr_ct = 0
        loss_epoch_expr_ct_embed = 0
        loss_epoch_expr = 0
        loss_epoch_expr_immune = 0
        loss_epoch_expr_invasive = 0
        loss_epoch_expr_bulk = 0
        loss_epoch_logits = 0
        loss_epoch_comp_est = 0
        loss_epoch_comp_gt = 0
        loss_epoch_pearson = 0
        loss_epoch_vq = 0
        loss_epoch_panel_completion = 0
        if use_celltype:
            running_pred_counts = torch.zeros(n_classes, device=device)
            running_gt_counts = torch.zeros(n_classes, device=device)

        epoch_class_counts = (
            np.zeros(n_classes, dtype=np.int64) if n_classes > 0 else None
        )
        pbar = tqdm(
            dataloader,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc=f"Train epoch {epoch+1}/{total_epochs}",
            dynamic_ncols=True,
        )
        loss_total = None

        for (
            batch_nuclei,
            batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            patch_ids,
            batch_expr_mask,
            batch_slide_id,
        ) in pbar:
            optimizer.zero_grad()

            batch_nuclei = batch_nuclei.to(device)
            batch_type_patch = batch_type_patch.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_expr_mask = batch_expr_mask.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)
            patch_ids = patch_ids.to(device)
            slide_ids_unique = torch.unique(batch_slide_id)
            if slide_ids_unique.numel() != 1:
                raise RuntimeError("Mixed slides in batch; set batch_size=1 for per-slide avgexp.")
            slide_id_val = int(slide_ids_unique.item())
            expr_ref_batch = expr_ref_torch_map.get(slide_id_val, expr_ref_torch)
            model_extra_kwargs = {}
            if supports_cell_graph:
                coord_map_slide = slide_coord_map_by_slide.get(slide_id_val)
                graph = build_cell_graph(
                    batch_nuclei,
                    patch_ids,
                    k_neighbors=graph_k,
                    coords_batch=None,
                    cell_coord_map=coord_map_slide,
                    cross_patch=graph_cross_patch,
                    cross_patch_k=graph_cross_patch_k,
                )
                model_extra_kwargs = {
                    "coords_cells": graph.coords,
                    "cell_edge_index": graph.edge_index,
                    "cell_patch_ids": graph.patch_index,
                }

            batch_expr_mask_pc = flatten_expr_mask(batch_expr_mask, batch_n_cells)

            # Optional: random hiding mask for panel-completion training (defined in per-cell space).
            mask_panel_pc = (
                batch_expr_mask_pc > 0.5
                if batch_expr_mask_pc is not None and batch_expr_mask_pc.numel() > 0
                else None
            )
            natural_missing_pc = (
                (~mask_panel_pc)
                if panel_completion_enabled
                and panel_use_natural_missing
                and mask_panel_pc is not None
                else None
            )
            mask_hide_pc = None
            if (
                panel_completion_enabled
                and panel_completion_loss_weight > 0
                and mask_panel_pc is not None
                and panel_hide_frac > 0
            ):
                mask_hide_pc = (torch.rand_like(batch_expr_mask_pc) < panel_hide_frac) & mask_panel_pc

            # Prevent leakage: for held-out genes (and optionally hidden genes), replace GT expr
            # with the (non-leaky) ref baseline before calling the model.
            batch_expr_for_model = batch_expr
            holdout_mask_vec = holdout_mask_by_slide.get(slide_id_val)
            need_clone = False
            if holdout_mask_vec is not None and np.any(holdout_mask_vec > 0):
                need_clone = True
            if natural_missing_pc is not None and bool(natural_missing_pc.any()):
                need_clone = True
            if mask_hide_pc is not None and panel_hide_in_forward:
                need_clone = True
            if need_clone:
                batch_expr_for_model = batch_expr.clone()

            if holdout_mask_vec is not None and np.any(holdout_mask_vec > 0):
                holdout_idx = torch.from_numpy(
                    np.where(holdout_mask_vec > 0.5)[0].astype(np.int64)
                ).to(device)
                if holdout_idx.numel() > 0:
                    n_ref_local = expr_ref_batch.shape[0] if expr_ref_batch is not None else 0
                    pc_off = 0
                    for b in range(batch_expr_for_model.shape[0]):
                        n_valid = int(batch_n_cells[b].item())
                        if n_valid <= 0:
                            continue
                        ct_b = batch_ct[b, :n_valid].long().clamp(min=0)
                        if n_ref_local > 0:
                            ct_b = ct_b.clamp(max=n_ref_local - 1)
                            baseline_all = expr_ref_batch[ct_b]
                            batch_expr_for_model[b, :n_valid, holdout_idx] = baseline_all[:, holdout_idx]
                            if mask_hide_pc is not None and panel_hide_in_forward:
                                mask_hide_b = mask_hide_pc[pc_off : pc_off + n_valid]
                                if mask_hide_b.any():
                                    expr_b = batch_expr_for_model[b, :n_valid, :]
                                    expr_b[mask_hide_b] = baseline_all[mask_hide_b]
                        pc_off += n_valid
            else:
                n_ref_local = expr_ref_batch.shape[0] if expr_ref_batch is not None else 0
                pc_off = 0
                for b in range(batch_expr_for_model.shape[0]):
                    n_valid = int(batch_n_cells[b].item())
                    if n_valid <= 0:
                        continue
                    mask_forward_b = None
                    if natural_missing_pc is not None:
                        mask_missing_b = natural_missing_pc[pc_off : pc_off + n_valid]
                        if mask_missing_b.any():
                            mask_forward_b = mask_missing_b
                    if mask_hide_pc is not None and panel_hide_in_forward:
                        mask_hide_b = mask_hide_pc[pc_off : pc_off + n_valid]
                        if mask_hide_b.any():
                            mask_forward_b = (
                                mask_hide_b
                                if mask_forward_b is None
                                else (mask_forward_b | mask_hide_b)
                            )
                    if mask_forward_b is None or not mask_forward_b.any():
                        pc_off += n_valid
                        continue
                    if n_ref_local > 0:
                        ct_b = batch_ct[b, :n_valid].long().clamp(min=0).clamp(max=n_ref_local - 1)
                        baseline_all = expr_ref_batch[ct_b]
                        expr_b = batch_expr_for_model[b, :n_valid, :]
                        expr_b[mask_forward_b] = baseline_all[mask_forward_b]
                    pc_off += n_valid

            (
                out_cell_type,
                out_map,
                batch_ct_pc,
                out_expr,
                out_expr_immune,
                out_expr_invasive,
                out_cell_type_expr,
                fv_cell_type_expr,
                out_cell_type_gt_expr,
                fv_cell_type_gt_expr,
                batch_expr_pc,
                comp_estimated,
                _,
                _,
            ) = model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                expr_ref_batch,
                batch_ct,
                batch_expr_for_model,
                patch_ids=patch_ids,
                **model_extra_kwargs,
            )

            if batch_ct_pc.shape[0] == 0:
                continue

            current_immune_frac = 0.0
            if (
                epoch_class_counts is not None
                and immune_class_indices
                and batch_ct_pc.numel() > 0
            ):
                class_counts_batch = (
                    torch.bincount(
                        batch_ct_pc.detach().cpu(), minlength=n_classes
                    )
                    .to(torch.int64)
                    .cpu()
                    .numpy()
                )
                epoch_class_counts += class_counts_batch
                immune_count = class_counts_batch[immune_class_indices].sum()
                total_cells_batch = class_counts_batch.sum()
                if total_cells_batch > 0:
                    current_immune_frac = float(immune_count) / float(
                        total_cells_batch
                    )
                pass

            aux_main = getattr(model, "last_aux_losses", {}) or {}
            ref_base_main = aux_main.get("expr_ref_base")
            # Train the main expression path on full expression rather than
            # residual-over-reference expression; keep ref_base_main available
            # for panel completion and other auxiliary terms below.
            pred_expr_for_loss = out_expr
            target_expr_for_loss = batch_expr_pc
            if use_expr_baseline and baseline_torch is not None:
                pred_expr_for_loss = pred_expr_for_loss - baseline_torch
                target_expr_for_loss = target_expr_for_loss - baseline_torch

            loss_expr_val = masked_mse(pred_expr_for_loss, target_expr_for_loss, batch_expr_mask_pc)
            loss_map_val = loss_map(out_map, batch_type_patch)

            # Panel completion loss: predict held-out (and optionally randomly hidden) genes
            # using the measured genes in the same cell as context.
            loss_panel_completion_val = torch.tensor(0.0, device=device)
            if (
                panel_completion_enabled
                and panel_completion_loss_weight > 0
                and hasattr(model, "completion_head")
                and model.completion_head is not None
            ):
                expr_true_pc = flatten_expr(batch_expr, batch_n_cells)
                if expr_true_pc is not None and expr_true_pc.shape == out_expr.shape:
                    ref_base_pc = (
                        ref_base_main
                        if ref_base_main is not None and ref_base_main.shape == out_expr.shape
                        else torch.zeros_like(out_expr)
                    )

                    mask_panel = batch_expr_mask_pc.float() if batch_expr_mask_pc is not None else None
                    if mask_panel is not None and mask_panel.shape == out_expr.shape:
                        mask_hide = (
                            mask_hide_pc
                            if mask_hide_pc is not None
                            else torch.zeros_like(mask_panel, dtype=torch.bool)
                        )
                        mask_obs = (mask_panel > 0.5) & (~mask_hide)
                        mask_obs_f = mask_obs.float()

                        # Targets: natural missing genes and/or explicitly hidden genes.
                        mask_target = mask_hide.float()
                        if panel_use_natural_missing:
                            mask_target = torch.maximum(mask_target, (1.0 - mask_panel))
                        if panel_train_on_holdout:
                            hold_mask_vec = None
                            if holdout_mask_by_slide is not None:
                                hold_mask_vec = holdout_mask_by_slide.get(slide_id_val)
                            if hold_mask_vec is not None:
                                hold_mask_t = torch.from_numpy(
                                    (np.asarray(hold_mask_vec) > 0.5).astype(np.float32)
                                ).to(device)
                                if hold_mask_t.numel() == mask_target.shape[1]:
                                    mask_target = torch.maximum(mask_target, hold_mask_t.view(1, -1))

                        if float(mask_target.sum().item()) > 0:
                            delta_obs = (expr_true_pc - ref_base_pc) * mask_obs_f
                            delta_morph = out_expr - ref_base_pc
                            if panel_detach_morph:
                                delta_morph = delta_morph.detach()

                            delta_hat = model.completion_head(
                                delta_obs,
                                mask_obs_f,
                                delta_morph if panel_use_morph else None,
                            )
                            pred_completed = F.relu(ref_base_pc + delta_hat)
                            if panel_copy_observed:
                                pred_completed = mask_obs_f * expr_true_pc + (1.0 - mask_obs_f) * pred_completed

                            pred_comp_for_loss = pred_completed - ref_base_pc
                            targ_comp_for_loss = expr_true_pc - ref_base_pc
                            if use_expr_baseline and baseline_torch is not None:
                                pred_comp_for_loss = pred_comp_for_loss - baseline_torch
                                targ_comp_for_loss = targ_comp_for_loss - baseline_torch

                            loss_panel_completion_val = masked_mse(
                                pred_comp_for_loss, targ_comp_for_loss, mask_target
                            )

            if use_celltype:
                loss_ct_hist_val = loss_ct_hist(out_cell_type.clone(), batch_ct_pc)

                loss_expr_ct_val = loss_expr_ct(out_cell_type_expr.clone(), batch_ct_pc)

                loss_expr_ct_embed_val = 100 * loss_expr_ct_embed(
                    fv_cell_type_expr,
                    fv_cell_type_gt_expr,
                    target=torch.ones(batch_ct_pc.size(0)).to(device),
                )

                loss_logits_val = loss_logits(
                    out_cell_type_expr.clone(), out_cell_type_gt_expr.clone()
                )
            else:
                loss_ct_hist_val = torch.tensor(0.0).to(device)
                loss_expr_ct_val = torch.tensor(0.0).to(device)
                loss_expr_ct_embed_val = torch.tensor(0.0).to(device)
                loss_logits_val = torch.tensor(0.0).to(device)

            if use_neighb:

                def _find_class_idx(candidates):
                    candidates = {c.strip().lower() for c in candidates}
                    for idx, name in enumerate(classes):
                        lname = name.strip().lower()
                        if lname in candidates:
                            return idx
                    raise ValueError(f"No class found matching {candidates}")

                inv_ct_idx = _find_class_idx(["malignant"])

                if immune_class_indices:
                    imm_mask = torch.isin(
                        batch_ct_pc,
                        torch.tensor(immune_class_indices, device=device),
                    )
                    imm_idx = torch.where(imm_mask)[0]
                else:
                    imm_idx = torch.tensor([], device=device, dtype=torch.long)

                if imm_idx.shape[0] > 0:
                    imm_mask_expr = (
                        batch_expr_mask_pc[imm_idx, :] if batch_expr_mask_pc is not None else None
                    )
                    ref_base_immune = aux_main.get("expr_ref_base_immune")
                    pred_imm = out_expr_immune[imm_idx, :]
                    targ_imm = batch_expr_pc[imm_idx, :]
                    if ref_base_immune is not None and ref_base_immune.shape == out_expr_immune.shape:
                        pred_imm = pred_imm - ref_base_immune[imm_idx, :]
                        targ_imm = targ_imm - ref_base_immune[imm_idx, :]
                    if use_expr_baseline and baseline_torch is not None:
                        pred_imm = pred_imm - baseline_torch
                        targ_imm = targ_imm - baseline_torch
                    loss_expr_immune_val = (1 / n_classes) * masked_mse(
                        pred_imm, targ_imm, imm_mask_expr
                    )
                else:
                    loss_expr_immune_val = torch.tensor(0.0).to(device)

                inv_idx = torch.where(batch_ct_pc == inv_ct_idx)[0]
                if inv_idx.shape[0] > 0:
                    inv_mask_expr = (
                        batch_expr_mask_pc[inv_idx, :] if batch_expr_mask_pc is not None else None
                    )
                    ref_base_inv = aux_main.get("expr_ref_base_invasive")
                    pred_inv = out_expr_invasive[inv_idx, :]
                    targ_inv = batch_expr_pc[inv_idx, :]
                    if ref_base_inv is not None and ref_base_inv.shape == out_expr_invasive.shape:
                        pred_inv = pred_inv - ref_base_inv[inv_idx, :]
                        targ_inv = targ_inv - ref_base_inv[inv_idx, :]
                    if use_expr_baseline and baseline_torch is not None:
                        pred_inv = pred_inv - baseline_torch
                        targ_inv = targ_inv - baseline_torch
                    loss_expr_invasive_val = (1 / n_classes) * masked_mse(
                        pred_inv, targ_inv, inv_mask_expr
                    )
                else:
                    loss_expr_invasive_val = torch.tensor(0.0).to(device)

                comp_cells = aux_main.get("comp_cells")
                comp_source = (
                    comp_estimated
                    if comp_estimated is not None and comp_estimated.shape[0] > 0
                    else comp_cells
                )
                comp_losses_ready = comp_source is not None and comp_source.shape[0] > 0
                if comp_losses_ready:
                    comp_estimated_vals = comp_source.clone()
                    n_cells = batch_n_cells.squeeze(-1).float()
                    valid_mask = n_cells > 0
                    if valid_mask.any():
                        weights = n_cells[valid_mask]
                        weights = weights / weights.sum()
                        comp_estimated_sum = torch.sum(
                            comp_estimated_vals[valid_mask] * weights.unsqueeze(1), dim=0
                        )
                    else:
                        comp_estimated_sum = comp_estimated_vals.mean(dim=0)

                    comp_gt = F.one_hot(batch_ct_pc, num_classes=n_classes).float()
                    comp_gt = torch.mean(comp_gt, dim=0)
                    comp_gt = comp_gt.clamp_min(1e-8)
                    comp_gt = comp_gt / comp_gt.sum()

                    kl_eps = 1e-8
                    comp_estimated_log = torch.log(
                        comp_estimated_sum.clamp_min(kl_eps)
                    )

                    comp_logits = out_cell_type.clone()
                    logits_logsum = torch.logsumexp(comp_logits, dim=1, keepdim=True)
                    cell_log_probs = comp_logits - logits_logsum
                    comp_out = torch.exp(cell_log_probs).mean(dim=0)
                    comp_out = comp_out.clamp_min(kl_eps)
                    comp_out = comp_out / comp_out.sum()
                    comp_out_log = torch.log(comp_out.clamp_min(kl_eps))

                    kl_est_vec = F.kl_div(
                        comp_estimated_log, comp_gt, reduction="none"
                    )
                    kl_out_vec = F.kl_div(
                        comp_out_log, comp_gt, reduction="none"
                    )

                    pred_probs_cells = torch.softmax(out_cell_type.detach(), dim=1)
                    running_pred_counts += pred_probs_cells.sum(dim=0)
                    gt_onehot = F.one_hot(batch_ct_pc, num_classes=n_classes).float()
                    running_gt_counts += gt_onehot.sum(dim=0)

                    if slide_comp_target is not None:
                        target_dist = slide_comp_target
                    else:
                        target_dist = running_gt_counts / running_gt_counts.sum().clamp_min(1.0)

                    pred_dist = running_pred_counts / running_pred_counts.sum().clamp_min(1.0)
                    class_error = torch.abs(pred_dist - target_dist).detach()
                    class_weights = class_error + 1e-6
                    class_weights = class_weights / class_weights.sum().clamp_min(1e-6)

                    loss_comp_est_val = torch.sum(class_weights * kl_est_vec)
                    loss_comp_gt_val = torch.sum(class_weights * kl_out_vec)
                else:
                    loss_comp_est_val = torch.tensor(0.0, device=device)
                    loss_comp_gt_val = torch.tensor(0.0, device=device)
            else:
                loss_comp_est_val = torch.tensor(0.0).to(device)
                loss_comp_gt_val = torch.tensor(0.0).to(device)
                loss_expr_immune_val = torch.tensor(0.0).to(device)
                loss_expr_invasive_val = torch.tensor(0.0).to(device)

            loss_expr_bulk_val = torch.tensor(0.0, device=device)
            if bulk_loss_weight > 0:
                if use_celltype and batch_ct_pc.numel() > 0:
                    # Combine slide_id and cell type to keep pseudo-bulk biologically coherent.
                    bulk_groups = batch_ct_pc + (slide_id_val + 1) * (n_classes + 1)
                else:
                    # Fall back to slide-level pseudo-bulk.
                    bulk_groups = batch_ct_pc.new_full(
                        (batch_ct_pc.shape[0],), slide_id_val, dtype=torch.long
                    )
                loss_expr_bulk_val = pseudo_bulk_mse(
                    pred_expr_for_loss, target_expr_for_loss, batch_expr_mask_pc, bulk_groups
                )

            # The explicit Pearson term is the only active correlation penalty
            # in train.py; corr_loss_weight is intentionally kept off here.
            pearson_weight_mult = pearson_loss_weight
            if pearson_weight_mult > 0:
                loss_pearson_val = masked_pearson(
                    pred_expr_for_loss,
                    target_expr_for_loss,
                    batch_expr_mask_pc,
                )
            else:
                loss_pearson_val = torch.tensor(0.0, device=device)

            # Always define these for the variance-ratio guard below.
            var_pred = masked_var(out_expr, batch_expr_mask_pc)
            var_label = masked_var(batch_expr_pc, batch_expr_mask_pc)
            var_w_cfg = float(getattr(opts.training, "expr_var_penalty_weight", 0.0))
            if var_w_cfg > 0:
                loss_var_val = var_w_cfg * torch.abs(var_pred - var_label)
            else:
                loss_var_val = torch.tensor(0.0, device=device)

            # Entropy regulariser to discourage reference weight collapse
            entropy_w = float(getattr(opts.training, "ref_entropy_weight", 0.0))
            if entropy_w > 0:
                aux = getattr(model, "last_aux_losses", {})
                ent_base = aux.get("ref_weight_entropy")
                ent_imm = aux.get("ref_weight_entropy_immune")
                ent_inv = aux.get("ref_weight_entropy_invasive")
                ent_terms = [
                    x for x in (ent_base, ent_imm, ent_inv)
                    if x is not None and torch.isfinite(x)
                ]
                if ent_terms:
                    entropy_gain = torch.stack(ent_terms).mean()
                    # subtract because we want to maximise entropy
                    loss_entropy_val = -entropy_w * entropy_gain
                else:
                    loss_entropy_val = torch.tensor(0.0, device=device)
            else:
                loss_entropy_val = torch.tensor(0.0, device=device)

            aux_losses = getattr(model, "last_aux_losses", {})
            loss_vq_val = aux_losses.get(
                "vq_patch", torch.tensor(0.0, device=device)
            )

            # sum all losses
            loss = (
                loss_map_val
                + loss_ct_hist_val
                + loss_expr_ct_val
                + loss_expr_val
                + panel_completion_loss_weight * loss_panel_completion_val
                + loss_expr_immune_val
                + loss_expr_invasive_val
                + expr_ct_embed_loss_weight * loss_expr_ct_embed_val
                + logits_loss_weight * loss_logits_val
                + loss_comp_est_val
                + loss_comp_gt_val
                + bulk_loss_weight * loss_expr_bulk_val
                + pearson_weight_mult * loss_pearson_val
                + loss_var_val
                + loss_entropy_val
                + loss_vq_val
            )

            loss_total = loss.detach()

            # Hard guard: skip update on non-finite loss or extreme variance ratio
            var_ratio = float(
                (var_pred.detach() / var_label.detach().clamp_min(1e-6)).item()
            )
            if not torch.isfinite(loss_total):
                logging.warning(
                    "Skip batch (non-finite loss) slide_id=%s var_pred=%.4f var_label=%.4f",
                    slide_id_val,
                    float(var_pred.detach().item()),
                    float(var_label.detach().item()),
                )
                continue
            if torch.isfinite(var_pred) and torch.isfinite(var_label) and var_ratio > var_ratio_limit:
                logging.warning(
                    "Skip batch (expr_var/label_var=%.2f > %.1f) slide_id=%s",
                    var_ratio,
                    var_ratio_limit,
                    slide_id_val,
                )
                continue

            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            loss_epoch += loss.mean().item()

            loss_epoch_map += loss_map_val.mean().item()
            loss_epoch_ct_hist += loss_ct_hist_val.mean().item()
            loss_epoch_expr_ct += loss_expr_ct_val.item()
            loss_epoch_expr += loss_expr_val.item()
            loss_epoch_expr_immune += loss_expr_immune_val.item()
            loss_epoch_expr_invasive += loss_expr_invasive_val.item()
            loss_epoch_expr_bulk += loss_expr_bulk_val.item()
            loss_epoch_expr_ct_embed += loss_expr_ct_embed_val.item()
            loss_epoch_logits += loss_logits_val.item()
            loss_epoch_comp_est += loss_comp_est_val.item()
            loss_epoch_comp_gt += loss_comp_gt_val.item()
            loss_epoch_pearson += loss_pearson_val.item()
            loss_epoch_vq += loss_vq_val.item()
            loss_epoch_panel_completion += loss_panel_completion_val.item()

            # Only update description if tqdm is being used
            if hasattr(pbar, "set_description"):
                pbar.set_description(f"loss: {loss_total:.4f}")

            optimizer.step()

        print(
            "Epoch[{}/{}], Loss:{:.4f}".format(
                epoch + 1, opts.training.total_epochs, loss_epoch
            )
        )
        if use_celltype:
            pred_dist_epoch = running_pred_counts / running_pred_counts.sum().clamp_min(1.0)
            gt_dist_epoch = running_gt_counts / running_gt_counts.sum().clamp_min(1.0)
            logging.info(
                "Epoch %d running composition pred %s | gt %s",
                epoch + 1,
                ", ".join(
                    f"{classes[idx]}:{pred_dist_epoch[idx]:.4f}"
                    for idx in range(n_classes)
                ),
                ", ".join(
                    f"{classes[idx]}:{gt_dist_epoch[idx]:.4f}"
                    for idx in range(n_classes)
                ),
            )
        # Save model
        ckpt_model_path = None
        if (epoch % opts.save_freqs.model_freq) == 0:
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            logging.info("Model saved: %s" % save_path)
            ckpt_model_path = save_path
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_optim.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Optimiser saved: %s" % save_path)

        val_metrics = None
        if val_dataloader is not None:
            val_metrics = evaluate_validation(
                model,
                val_dataloader,
                expr_ref_torch_val,
                device,
                n_classes,
                graph_k=graph_k,
                graph_cross_patch=graph_cross_patch,
                graph_cross_patch_k=graph_cross_patch_k,
                slide_coord_map_by_slide=slide_coord_map_by_slide,
                expr_ref_torch_map=expr_ref_torch_val_map,
                holdout_mask_by_slide=None,
                gene_names=gene_names,
                epoch=epoch + 1,
                per_gene_dir=None,
                svg_rank_gene_indices_by_slide=val_svg_rank_indices_by_slide,
                svg_topk=svg_topk,
            )
            _log_gene_pcc_epoch(
                val_metrics,
                split_tag="VAL",
                epoch=epoch + 1,
                svg_topk=svg_topk,
            )

        ext_metrics = None
        run_ext_eval = False
        if external_dataloader is not None:
            run_ext_eval = ((epoch + 1) % ext_eval_every_epochs == 0) or (
                ext_eval_final_epoch and (epoch + 1 == total_epochs)
            )
        if run_ext_eval and external_dataloader is not None:
            ext_metrics = evaluate_validation(
                model,
                external_dataloader,
                expr_ref_torch,
                device,
                n_classes,
                graph_k=graph_k,
                graph_cross_patch=graph_cross_patch,
                graph_cross_patch_k=graph_cross_patch_k,
                slide_coord_map_by_slide=slide_coord_map_by_slide,
                expr_ref_torch_map=expr_ref_torch_map,
                holdout_mask_by_slide=None,
                gene_names=gene_names,
                epoch=epoch + 1,
                per_gene_dir=None,
                svg_rank_gene_indices_by_slide=ext_svg_rank_indices_by_slide,
                svg_topk=svg_topk,
            )
            _log_gene_pcc_epoch(
                ext_metrics,
                split_tag="EXT",
                epoch=epoch + 1,
                svg_topk=svg_topk,
            )
        elif external_dataloader is not None:
            logging.info(
                "EXT evaluation skipped at epoch %d (every %d epochs)",
                epoch + 1,
                ext_eval_every_epochs,
            )

        val_gene_pooled_mean = (
            float(val_metrics.get("pearson_gene_pooled_mean", 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        val_ct_macro = (
            float(val_metrics.get("ct_accuracy_macro", 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        if isinstance(ext_metrics, dict):
            ext_gene_pooled_record = float(
                ext_metrics.get("pearson_gene_pooled_mean", 0.0)
            )
        else:
            ext_gene_pooled_record = None
        record = {
            "epoch": int(epoch + 1),
            "checkpoint": ckpt_model_path,
            "train_loss_total": float(loss_epoch),
            "val_pearson_gene_pooled_mean": val_gene_pooled_mean,
            "val_ct_accuracy_macro": val_ct_macro,
            "ext_pearson_gene_pooled_mean": ext_gene_pooled_record,
            "external_source": external_source_tag,
            "parity_tolerance_abs": parity_tolerance_abs,
        }
        epoch_records.append(record)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(epoch_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(record) + "\n")
        df_epochs = pd.DataFrame(epoch_records)
        df_epochs.to_csv(kpi_csv_path, index=False)
        df_epochs.to_csv(os.path.join(metrics_dir, "epoch_metrics.csv"), index=False)

        if val_metrics is not None:
            metrics_payload = {
                "epoch": int(epoch + 1),
                "checkpoint": ckpt_model_path,
                "val": val_metrics,
                "external": ext_metrics if ext_metrics is not None else {},
            }
            _write_json(
                os.path.join(metrics_dir, f"epoch_{epoch + 1:03d}_metrics.json"),
                metrics_payload,
            )

        primary_improved = val_gene_pooled_mean > (best_val_gene_pooled + best_metric_eps)
        primary_tied = abs(val_gene_pooled_mean - best_val_gene_pooled) <= best_metric_eps
        ct_constraint_ok = (
            best_epoch is None
            or val_ct_macro >= (best_val_ct_macro - best_metric_eps)
        )
        if (primary_improved and ct_constraint_ok) or (
            primary_tied and val_ct_macro > (best_val_ct_macro + best_metric_eps)
        ):
            best_val_gene_pooled = val_gene_pooled_mean
            best_val_ct_macro = val_ct_macro
            best_epoch = int(epoch + 1)
            best_ckpt_path = ckpt_model_path
        elif primary_improved and not ct_constraint_ok:
            logging.info(
                "Best-checkpoint update rejected at epoch %d: pooled_gene_pearson %.6f > %.6f but ct_macro %.6f < %.6f",
                epoch + 1,
                val_gene_pooled_mean,
                best_val_gene_pooled,
                val_ct_macro,
                best_val_ct_macro,
            )

        # Legacy loader: no immune feedback loop

        global_step += 1

    # df_losses.to_csv(f"{experiment_path}/losses.csv")

    strict_best = {
        "selection_metric": "pearson_gene_pooled_mean",
        "selection_constraint": "non_decreasing_ct_accuracy_macro",
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_pearson_gene_pooled_mean": (
            float(best_val_gene_pooled) if np.isfinite(best_val_gene_pooled) else None
        ),
        "best_val_ct_accuracy_macro": (
            float(best_val_ct_macro) if np.isfinite(best_val_ct_macro) else None
        ),
        "best_checkpoint": best_ckpt_path,
        "parity_tolerance_abs": parity_tolerance_abs,
        "external_source": external_source_tag,
    }
    _write_json(os.path.join(metrics_dir, "strict_best.json"), strict_best)
    _write_json(os.path.join(experiment_path, "strict_best.json"), strict_best)
    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=0,
        type=int,
        help="resume training from this epoch, set to 0 for new training",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    config = parser.parse_args()
    main(config)

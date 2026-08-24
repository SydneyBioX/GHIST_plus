"""Original TMA selector/filter behaviour on the canonical GHIST+ dataset.

Only ``CoordinateDatasetView`` is new interface glue: it exposes the original
coordinate-bearing selector sample from an already-built canonical union
dataset. Selection formulas, defaults, cache policy, fallbacks, and filtering
match ``HEAD:train_tma_select.py`` and its dataset fork.
"""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import model.graph as graph_utils


TMA_SELECTION_METHOD = "vq_roi_score_twostage"


def punch_cache_path(base_dir, slide_idx: int) -> str:
    return os.path.join(os.fspath(base_dir), f"punch_slide{int(slide_idx)}.pt")


def _as_float_attr(obj, names, default):
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(default)


class CoordinateDatasetView(Dataset):
    """Adapt canonical 9-field samples to the original selector contract."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._read_dataset = copy.copy(dataset)
        self._read_dataset.stain_aug = False
        if getattr(self._read_dataset, "tfs_test", None) is not None:
            self._read_dataset.tfs = self._read_dataset.tfs_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self._read_dataset[index]
        (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
            _expr_mask,
            _slide_id,
        ) = sample
        hs, ws = self.dataset.coords_starts[index]
        patch_coord = torch.tensor(
            [
                float(hs) + 0.5 * float(self.dataset.hsize),
                float(ws) + 0.5 * float(self.dataset.wsize),
            ],
            dtype=torch.float32,
        )
        patch_slide_idx = torch.tensor(
            int(getattr(self.dataset, "slide_idx", -1)), dtype=torch.long
        )
        return (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
            patch_coord,
            patch_slide_idx,
        )


class PunchSubset(Dataset):
    """Canonical adapter carrying the original filtered coordinate state."""

    def __init__(self, dataset, indices, metadata, immune_sampler_boost=1.0):
        self.source_dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.selection_metadata = metadata
        self.dataset = copy.copy(dataset)
        self.dataset.coords_starts_unfiltered = list(dataset.coords_starts)
        self.dataset.coords_starts = [
            dataset.coords_starts[index] for index in self.indices
        ]
        self.dataset.n_patches = len(self.dataset.coords_starts)
        compute_weights = getattr(
            self.dataset, "_compute_patch_sampling_weights", None
        )
        if compute_weights is not None:
            self.dataset.patch_weights = compute_weights(immune_sampler_boost)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "dataset"), name)


def _filter_indices(dataset, metadata):
    center = metadata.get("punch_center", None)
    window_px = float(metadata.get("window_px", 0.0))
    if center is None or len(center) < 2 or window_px <= 0:
        return None
    center_y, center_x = float(center[0]), float(center[1])
    half = 0.5 * window_px
    indices = []
    for index, (hs, ws) in enumerate(dataset.coords_starts):
        patch_y = float(hs) + 0.5 * float(dataset.hsize)
        patch_x = float(ws) + 0.5 * float(dataset.wsize)
        if abs(patch_y - center_y) <= half and abs(patch_x - center_x) <= half:
            indices.append(index)
    return indices


def apply_cached_punch_filter(
    dataset,
    opts_data,
    cache_path,
    *,
    immune_sampler_boost=1.0,
    force_no_punch_filter=False,
):
    """Apply the original dataset-fork filter, including fail-open semantics."""

    if force_no_punch_filter:
        return dataset
    enabled = bool(
        getattr(opts_data, "punch_select_enabled", False)
        or getattr(opts_data, "tma_select_enabled", False)
    )
    if not enabled:
        return dataset
    split_cfg = str(getattr(opts_data, "punch_filter_splits", "train")).lower()
    allowed = {
        value.strip()
        for value in split_cfg.replace(",", " ").split()
        if value.strip()
    }
    if allowed and dataset.mode not in allowed:
        return dataset
    cache_path = os.fspath(cache_path)
    if not os.path.isfile(cache_path):
        print(
            f"[punch] No cached punch for slide={dataset.slide_idx}; "
            "using full patch set."
        )
        return dataset
    try:
        metadata = torch.load(cache_path, weights_only=False)
        indices = _filter_indices(dataset, metadata)
        if indices is None:
            print(f"[punch] Invalid punch cache {cache_path}; using full patch set.")
            return dataset
        if not indices:
            print(
                f"[punch] Punch filter empty for slide={dataset.slide_idx}; "
                "using full patch set."
            )
            return dataset
    except Exception as exc:
        print(
            f"[punch] Failed loading punch cache {cache_path}: {exc}; "
            "using full patch set."
        )
        return dataset
    print(
        f"[punch] Applied punch filter slide={dataset.slide_idx}: "
        f"{len(indices)}/{len(dataset.coords_starts)} patches"
    )
    return PunchSubset(dataset, indices, metadata, immune_sampler_boost)


def _collect_patch_statistics(
    model,
    dataset,
    opts,
    device,
    expr_ref_torch,
    expr_ref_torch_map,
    classes,
    graph_k,
):
    loader = DataLoader(
        dataset=CoordinateDatasetView(dataset),
        batch_size=max(1, int(getattr(opts.training, "batch_size", 1))),
        shuffle=False,
        num_workers=int(getattr(opts.data, "punch_num_workers", 0)),
        drop_last=False,
        pin_memory=getattr(opts.data, "pin_memory", False),
    )
    slide_idx = int(getattr(dataset, "slide_idx", -1))
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
            _batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            patch_ids,
            patch_coords,
            patch_slide_idx,
        ) in loader:
            batch_nuclei = batch_nuclei.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)
            patch_ids = patch_ids.to(device)
            patch_coords = patch_coords.to(device)
            patch_slide_idx = patch_slide_idx.to(device)

            model_extra_kwargs = {}
            if getattr(model, "use_ecrm", False):
                graph = graph_utils.build_cell_graph(
                    batch_nuclei,
                    patch_ids,
                    k_neighbors=max(int(graph_k), 2),
                )
                model_extra_kwargs = {
                    "coords_cells": graph.coords,
                    "cell_edge_index": graph.edge_index,
                    "cell_patch_ids": graph.patch_index,
                }
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
                **model_extra_kwargs,
            )

            aux = getattr(model, "last_aux_losses", {}) or {}
            vq_err = aux.get("vq_patch_err")
            vq_idx = aux.get("vq_patch_idx")
            coords_np = patch_coords.detach().cpu().numpy().astype(np.float32)
            slides_np = patch_slide_idx.detach().cpu().numpy().astype(np.int64)
            n_cells_np = (
                batch_n_cells.detach().cpu().numpy().reshape(-1).astype(np.int64)
            )
            expr_sum_np = (
                batch_expr.detach()
                .sum(dim=(1, 2))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            keep = slides_np == slide_idx
            if not np.any(keep):
                continue
            coords_all.append(coords_np[keep])
            n_cells_all.append(n_cells_np[keep])
            expr_sum_all.append(expr_sum_np[keep])
            if (
                vq_err is not None
                and isinstance(vq_err, torch.Tensor)
                and vq_err.numel() > 0
            ):
                vq_err_all.append(vq_err.detach().cpu().numpy()[keep])
            if (
                vq_idx is not None
                and isinstance(vq_idx, torch.Tensor)
                and vq_idx.numel() > 0
            ):
                vq_idx_all.append(vq_idx.detach().cpu().numpy()[keep])

            n_classes = int(len(classes)) if classes is not None else 0
            if n_classes > 0 and batch_ct is not None and batch_expr is not None:
                max_cells = int(batch_ct.shape[1])
                n_cells_vec = batch_n_cells.view(-1).long().to(device)
                mask_cells = (
                    torch.arange(max_cells, device=device).unsqueeze(0)
                    < n_cells_vec.unsqueeze(1)
                )
                ct_onehot = F.one_hot(
                    batch_ct.clamp_min(0), num_classes=n_classes
                ).float()
                ct_onehot = ct_onehot * mask_cells.unsqueeze(-1).float()
                ct_counts_all.append(
                    ct_onehot.sum(dim=1).detach().cpu().numpy()[keep]
                )
                mask_float = mask_cells.unsqueeze(-1).to(batch_expr.dtype)
                expr_sum_genes = (batch_expr * mask_float).sum(dim=1)
                denominator = (
                    n_cells_vec.clamp_min(1).to(batch_expr.dtype).unsqueeze(1)
                )
                expr_mean_all.append(
                    (expr_sum_genes / denominator).detach().cpu().numpy()[keep]
                )

    if was_training:
        model.train()
    if not coords_all:
        return None
    return {
        "coords": np.concatenate(coords_all, axis=0).astype(np.float32),
        "n_cells": np.concatenate(n_cells_all, axis=0).astype(np.int64),
        "expr_sum": np.concatenate(expr_sum_all, axis=0).astype(np.float32),
        "vq_err": (
            np.concatenate(vq_err_all, axis=0).astype(np.float32)
            if vq_err_all
            else None
        ),
        "vq_idx": (
            np.concatenate(vq_idx_all, axis=0).astype(np.int64)
            if vq_idx_all
            else None
        ),
        "ct_counts": (
            np.concatenate(ct_counts_all, axis=0).astype(np.int64)
            if ct_counts_all
            else None
        ),
        "expr_mean": (
            np.concatenate(expr_mean_all, axis=0).astype(np.float32)
            if expr_mean_all
            else None
        ),
    }


def _choose_punch(statistics, opts, classes, slide_idx, window_px):
    coords_all = statistics["coords"]
    n_cells_all = statistics["n_cells"]
    expr_sum_all = statistics["expr_sum"]
    vq_err_all = statistics["vq_err"]
    vq_idx_all = statistics["vq_idx"]
    ct_counts_all = statistics["ct_counts"]
    expr_mean_all = statistics["expr_mean"]

    qc_min_cells = int(
        getattr(
            opts.data,
            "punch_qc_min_cells",
            getattr(opts.data, "broadcast_min_cells", 1),
        )
    )
    qc_min_expr_sum = float(getattr(opts.data, "punch_qc_min_expr_sum", 0.0))
    qc_mask = (n_cells_all >= qc_min_cells) & (expr_sum_all > qc_min_expr_sum)
    n_qc_total = int(qc_mask.sum())
    if n_qc_total <= 0:
        logging.warning(
            "[punch] No QC patches for slide=%s; falling back to minimum VQ error.",
            slide_idx,
        )
        if vq_err_all is None or vq_err_all.size == 0:
            return None
        return coords_all[int(np.argmin(vq_err_all))], {
            "fallback": "min_vq_err_no_qc"
        }
    if vq_idx_all is None or vq_idx_all.size == 0:
        logging.warning(
            "[punch] VQ indices unavailable for slide=%s; "
            "falling back to minimum VQ error.",
            slide_idx,
        )
        if vq_err_all is None or vq_err_all.size == 0:
            return None
        indices = np.where(qc_mask)[0]
        best = indices[int(np.argmin(vq_err_all[indices]))]
        return coords_all[best], {"fallback": "min_vq_err_no_idx"}

    vq_cfg = getattr(opts.model, "vq_patch", None)
    k_clusters = int(getattr(vq_cfg, "n_codes", 0)) if vq_cfg is not None else 0
    k_clusters = max(k_clusters, int(vq_idx_all.max()) + 1, 2)
    roi_balance_target = str(
        getattr(opts.data, "punch_roi_balance_target", "uniform")
    ).strip().lower()
    if roi_balance_target == "slide":
        counts_slide = np.bincount(
            vq_idx_all[qc_mask], minlength=k_clusters
        ).astype(np.float32)
        target = (
            counts_slide / counts_slide.sum()
            if counts_slide.sum() > 0
            else np.full((k_clusters,), 1.0 / k_clusters, dtype=np.float32)
        )
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
    num_samples = max(
        int(getattr(opts.data, "punch_min_samples", 200)), num_samples
    )
    num_samples = min(
        int(getattr(opts.data, "punch_max_samples", 5000)),
        num_samples,
        int(pool_coords.shape[0]),
    )
    num_samples = max(num_samples, 1)
    seed = int(
        getattr(
            opts.training,
            "seed",
            getattr(opts.training, "batch_sampler_seed", 0),
        )
    )
    rng = np.random.default_rng(seed + 10007 * max(slide_idx, 0))
    candidate_indices = rng.choice(
        pool_coords.shape[0],
        size=num_samples,
        replace=pool_coords.shape[0] < num_samples,
    )
    candidate_centers = pool_coords[candidate_indices]

    roi_w_balance = float(getattr(opts.data, "punch_roi_w_balance", 1.0))
    roi_w_coverage = float(getattr(opts.data, "punch_roi_w_coverage", 1.0))
    roi_w_size = float(getattr(opts.data, "punch_roi_w_size", 1.0))
    roi_min_qc = int(getattr(opts.data, "punch_roi_min_qc", 1))
    candidates = []
    for center in candidate_centers:
        in_mask = (np.abs(coords_all[:, 0] - center[0]) <= half) & (
            np.abs(coords_all[:, 1] - center[1]) <= half
        )
        n_total = int(in_mask.sum())
        if n_total <= 0:
            continue
        qc_in = in_mask & qc_mask
        n_qc = int(qc_in.sum())
        if n_qc < roi_min_qc:
            continue
        coverage = float(np.sqrt(n_qc / max(n_total, 1)))
        size_score = float(
            1.0 / (1.0 + np.exp(-2.0 * (n_qc / max(n_qc_total, 1))))
        )
        counts = np.bincount(
            vq_idx_all[qc_in], minlength=k_clusters
        ).astype(np.float32)
        if counts.sum() <= 0:
            balance = 0.0
        else:
            proportions = counts / counts.sum()
            balance = float(
                np.dot(proportions, target)
                / (
                    np.linalg.norm(proportions) * np.linalg.norm(target)
                    + 1e-8
                )
            )
        score = (
            (balance**roi_w_balance)
            * (coverage**roi_w_coverage)
            * (size_score**roi_w_size)
        )
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
        logging.warning(
            "[punch] ROI candidate sampling produced no valid windows for slide=%s.",
            slide_idx,
        )
        return None

    candidates.sort(key=lambda value: value["stage1_score"], reverse=True)
    best = candidates[0]
    best_coord = best["center"]
    best_meta = dict(best)
    best_meta.pop("center", None)

    if ct_counts_all is not None and expr_mean_all is not None:
        stage2_topk = max(1, int(getattr(opts.data, "punch_stage2_topk", 25)))
        stage2_min_ratio = float(
            np.clip(
                float(
                    getattr(
                        opts.data, "punch_stage2_min_stage1_ratio", 0.98
                    )
                ),
                0.0,
                1.0,
            )
        )
        subset = [
            candidate
            for candidate in candidates
            if candidate["stage1_score"]
            >= best["stage1_score"] * stage2_min_ratio
        ]
        subset = (
            subset[:stage2_topk]
            if len(subset) >= stage2_topk
            else candidates[:stage2_topk]
        )

        n_classes = int(ct_counts_all.shape[1])
        keep_ct = np.ones((n_classes,), dtype=bool)
        if bool(
            getattr(opts.data, "punch_stage2_exclude_unassigned", True)
        ) and classes:
            for class_index, name in enumerate(classes):
                if str(name).strip().lower() == "unassigned":
                    keep_ct[class_index] = False
                    break
        slide_ct = ct_counts_all[qc_mask].sum(axis=0).astype(np.float32)[keep_ct]
        slide_distribution = (
            slide_ct / slide_ct.sum()
            if slide_ct.sum() > 0
            else np.full(
                (keep_ct.sum(),),
                1.0 / max(int(keep_ct.sum()), 1),
                dtype=np.float32,
            )
        )
        uniform_distribution = np.full_like(
            slide_distribution, 1.0 / max(slide_distribution.size, 1)
        )
        alpha = float(
            np.clip(
                float(
                    getattr(opts.data, "punch_stage2_ct_blend_alpha", 0.5)
                ),
                0.0,
                1.0,
            )
        )
        target_distribution = (
            (1.0 - alpha) * slide_distribution
            + alpha * uniform_distribution
        )
        slide_cells_total = max(int(n_cells_all[qc_mask].sum()), 1)

        def molecular_rank(values, weights):
            min_patches = int(
                getattr(opts.data, "punch_stage2_min_patches_for_mol", 8)
            )
            if values.shape[0] < min_patches:
                return 0.0
            weights = weights.astype(np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                return 0.0
            values = values.astype(np.float64)
            mean = (values * weights[:, None]).sum(axis=0) / weight_sum
            centered = values - mean
            covariance = (
                (centered * weights[:, None]).T @ centered / weight_sum
            )
            covariance.flat[:: covariance.shape[0] + 1] += float(
                getattr(opts.data, "punch_stage2_mol_ridge", 1e-4)
            )
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            total = float(eigenvalues.sum())
            if total <= 0:
                return 0.0
            proportions = eigenvalues / total
            return float(
                np.clip(
                    np.exp(
                        -np.sum(
                            proportions * np.log(proportions + 1e-12)
                        )
                    )
                    / max(values.shape[1], 1),
                    0.0,
                    1.0,
                )
            )

        stage2_w_ct = float(getattr(opts.data, "punch_stage2_w_ct", 1.0))
        stage2_w_cells = float(
            getattr(opts.data, "punch_stage2_w_cells", 1.0)
        )
        stage2_w_mol = float(getattr(opts.data, "punch_stage2_w_mol", 1.0))
        best_stage2 = None
        best_stage2_score = -1.0
        for candidate in subset:
            center = candidate["center"]
            in_mask = (np.abs(coords_all[:, 0] - center[0]) <= half) & (
                np.abs(coords_all[:, 1] - center[1]) <= half
            )
            qc_in = in_mask & qc_mask
            if not np.any(qc_in):
                continue
            cells_roi = int(n_cells_all[qc_in].sum())
            if cells_roi < int(
                getattr(opts.data, "punch_stage2_min_cells", 1)
            ):
                continue
            ct_roi = ct_counts_all[qc_in].sum(axis=0).astype(np.float32)[keep_ct]
            if ct_roi.sum() > 0:
                ct_proportions = ct_roi / ct_roi.sum()
                ct_score = float(
                    np.dot(ct_proportions, target_distribution)
                    / (
                        np.linalg.norm(ct_proportions)
                        * np.linalg.norm(target_distribution)
                        + 1e-8
                    )
                )
            else:
                ct_score = 0.0
            ct_score = float(np.clip(ct_score, 0.0, 1.0))
            cell_score = float(
                1.0
                / (
                    1.0
                    + np.exp(-2.0 * (cells_roi / float(slide_cells_total)))
                )
            )
            molecular_score = molecular_rank(
                expr_mean_all[qc_in], n_cells_all[qc_in].astype(np.float32)
            )
            score = (
                (ct_score**stage2_w_ct)
                * (cell_score**stage2_w_cells)
                * (molecular_score**stage2_w_mol)
            )
            candidate.update(
                {
                    "stage2_score": float(score),
                    "stage2_ct": float(ct_score),
                    "stage2_cells": float(cell_score),
                    "stage2_mol": float(molecular_score),
                    "stage2_cells_roi": int(cells_roi),
                }
            )
            if score > best_stage2_score:
                best_stage2_score = float(score)
                best_stage2 = candidate
        if best_stage2 is not None:
            best_coord = best_stage2["center"]
            best_meta = dict(best_stage2)
            best_meta.pop("center", None)

    return best_coord, best_meta


def preselect_tma_punch_with_vq(
    model,
    dataset,
    opts,
    device,
    expr_ref_torch,
    expr_ref_torch_map=None,
    classes=None,
    *,
    graph_k: int = 6,
    cache_path=None,
):
    """Run original VQ two-stage selection and write its legacy cache."""

    enabled = bool(
        getattr(opts.data, "punch_select_enabled", True)
        or getattr(opts.data, "tma_select_enabled", True)
    )
    if not enabled:
        return None

    slide_idx = int(getattr(dataset, "slide_idx", -1))
    if cache_path is None:
        cache_path = punch_cache_path(dataset.experiment_path, slide_idx)
    cache_path = os.fspath(cache_path)
    force = bool(getattr(opts.data, "punch_force_reselect", False))
    if os.path.isfile(cache_path) and not force:
        logging.info(
            "[punch] Using cached punch selection for slide=%s: %s",
            slide_idx,
            cache_path,
        )
        return None

    window_um = _as_float_attr(
        opts.data, ("broadcast_window_um", "roi_size_um"), 1000.0
    )
    pixel_um = _as_float_attr(
        opts.data,
        ("pixel_size_um",),
        _as_float_attr(opts.model, ("pixel_size_um",), 0.2125),
    )
    if window_um <= 0 or pixel_um <= 0:
        logging.warning(
            "[punch] Invalid window_um=%.3f pixel_um=%.5f; skipping slide=%s",
            window_um,
            pixel_um,
            slide_idx,
        )
        return None
    window_px = window_um / pixel_um

    logging.info("[punch] Preselecting TMA punch via VQ for slide=%s", slide_idx)
    statistics = _collect_patch_statistics(
        model,
        dataset,
        opts,
        device,
        expr_ref_torch,
        expr_ref_torch_map,
        classes,
        graph_k,
    )
    if statistics is None:
        logging.warning(
            "[punch] No patches found for slide=%s; keeping full slide.",
            slide_idx,
        )
        return None
    selection = _choose_punch(statistics, opts, classes, slide_idx, window_px)
    if selection is None:
        return None
    best_coord, best_meta = selection
    metadata = {
        "punch_center": [float(best_coord[0]), float(best_coord[1])],
        "window_px": float(window_px),
        "punch_select_method": TMA_SELECTION_METHOD,
        "slide_idx": int(slide_idx),
        **best_meta,
    }
    torch.save(metadata, cache_path)
    logging.info(
        "[punch] Selected slide=%s center=%s window_px=%.1f -> %s",
        slide_idx,
        metadata["punch_center"],
        window_px,
        cache_path,
    )
    return None


__all__ = [
    "CoordinateDatasetView",
    "PunchSubset",
    "TMA_SELECTION_METHOD",
    "apply_cached_punch_filter",
    "preselect_tma_punch_with_vq",
    "punch_cache_path",
]

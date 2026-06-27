"""Training entry point and evaluation utilities for GHIST+."""

import argparse
import logging
import os
import sys
import shutil
import hashlib
from types import SimpleNamespace
import inspect
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort

import dataio.dataset_input_tma_select as dataset_input_tma_base
import dataio.dataset_input_union_tma_select as dataset_input_tma
import dataio.references as reference_utils
import dataio.samplers as sampler_utils
import dataio.spatial as spatial_utils
import dataio.tensors as tensor_utils
import model.framework as model_framework
import model.graph as graph_utils
import model.panel_completion as panel_completion
import utils.evaluation as evaluation_utils
import utils.metrics as metric_utils
import utils.utils as utils


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


# TMA punch preselection
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
        logging.warning(
            "[punch] Invalid window_um=%.3f pixel_um=%.5f; skipping slide=%s",
            window_um,
            pixel_um,
            slide_idx,
        )
        return
    window_px = window_um / pixel_um

    opts_stain_norm = opts.stain_norm
    if hasattr(opts_stain_norm, "fp_norm_ref") and isinstance(opts_stain_norm.fp_norm_ref, (list, tuple)):
        opts_stain_norm = SimpleNamespace(**vars(opts_stain_norm))
        opts_stain_norm.fp_norm_ref = opts_stain_norm.fp_norm_ref[0]

    logging.info("[punch] Preselecting TMA punch via VQ for slide=%s", slide_idx)
    ds = dataset_input_tma_base.DataProcessing(
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

            graph = graph_utils.build_cell_graph(
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
        num_samples = max(int(getattr(opts.data, "punch_min_samples", 200)), num_samples)
        num_samples = min(int(getattr(opts.data, "punch_max_samples", 5000)), num_samples, int(pool_coords.shape[0]))
        num_samples = max(num_samples, 1)
        seed = int(getattr(opts.training, "seed", 0))
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
            stage2_min_ratio = float(
                np.clip(
                    float(getattr(opts.data, "punch_stage2_min_stage1_ratio", 0.98)),
                    0.0,
                    1.0,
                )
            )
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
            p_slide = (
                slide_ct / slide_ct.sum()
                if slide_ct.sum() > 0
                else np.full(
                    (keep_ct.sum(),),
                    1.0 / max(int(keep_ct.sum()), 1),
                    dtype=np.float32,
                )
            )
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
                in_mask = (
                    (np.abs(coords_all[:, 0] - center[0]) <= half)
                    & (np.abs(coords_all[:, 1] - center[1]) <= half)
                )
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
    logging.info(
        "[punch] Selected slide=%s center=%s window_px=%.1f -> %s",
        slide_idx,
        meta["punch_center"],
        window_px,
        cache_path,
    )


def main(config):
    opts = _to_namespace(utils.json_file_to_pyobj(config.config_file))
    torch.autograd.set_detect_anomaly(False)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    device = utils.get_device(config.gpu_id)

    # Runtime configuration
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

    # Output and cache paths
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

    if config.resume_epoch != 0:
        make_new = False
    else:
        make_new = True

    timestamp = utils.get_experiment_id(make_new, opts.experiment_dirs.load_dir, config.fold_id)
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

    # Model and source setup
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
        logging.info("Cell types: %s", classes)
        logging.info("Num cell types: %d", n_classes)
    else:
        n_classes = 0
        classes = []
        class_weights_np = None

    def _ensure_list(sources):
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        return [
            _to_namespace(utils.json_file_to_pyobj(src))
            if isinstance(src, str)
            else _to_namespace(src)
            for src in sources
        ]

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

    # Gene panel and training statistics
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

    # Holdout and panel-completion settings
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

    # Expression baselines and imputation statistics
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
    if use_expr_baseline:
        baseline_torch = torch.from_numpy(gene_means_vec.astype(np.float32)).float().to(device)
        logging.info("Using per-gene baseline for delta training (breast_all)")
    else:
        baseline_torch = None

    holdout_genes_by_slide = {}
    holdout_mask_by_slide = {}

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
            ct_vals = ct_numeric.astype(int)
            if ct_vals.min() >= 1 and ct_vals.max() <= len(classes):
                ct_vals = ct_vals - 1
            ct_vals = ct_vals.clip(lower=0, upper=len(classes) - 1)
            return ct_vals

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
                    kept_ids, coords_yx = spatial_utils.centroids_from_label_image(
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
                    return metric_utils.morans_many(expr_model, coords_yx, k=8)
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
                    if len(chosen) < holdout_n_genes_eval:
                        for j in order:
                            g = present[int(j)]
                            if g not in chosen:
                                chosen.append(g)
                                if len(chosen) >= holdout_n_genes_eval:
                                    break
                else:
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
        ct_means_fallback = np.where(
            ct_counts_arr > 0,
            ct_means,
            np.broadcast_to(gene_means_vec, ct_means.shape),
        )

    avgexp_df_by_slide = {}
    if use_avgexp and use_celltype and classes:
        avgexp_df_by_slide = reference_utils.build_avgexp_df_by_slide(
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

    # Imputation cache
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

        src_expr_key = src.fp_expr
        df_expr = pd.read_csv(src_expr_key, index_col=0)
        missing_expr = [g for g in gene_names if g not in df_expr.columns]
        mask_vec = np.ones(len(gene_names), dtype=np.float32)
        for g in missing_expr:
            mask_vec[gene_names.index(g)] = 0.0
        if kind == "trainval":
            for g in holdout_genes_by_slide.get(slide_id_local, []):
                if g in gene_names:
                    mask_vec[gene_names.index(g)] = 0.0
        df_expr = df_expr.reindex(columns=gene_names)
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
        present_frac = 1.0 - (mask_vec == 0).mean()
        logging.debug(
            "Slide %s kind %s: %d/%d genes present (%.2f%%)",
            getattr(src, "slide_idx", "na"),
            kind,
            int(mask_vec.sum()),
            len(mask_vec),
            present_frac * 100,
        )
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

        df_ref = None
        if use_avgexp and use_celltype:
            df_ref = avgexp_df_by_slide.get(slide_id_local)
        return src, df_ref

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

    train_sources = imputed_trainval
    test_sources = imputed_test

    # Spatial graph support
    slide_coord_map_by_slide = {}
    for src in (train_sources + test_sources):
        sid = int(getattr(src, "slide_idx", -1))
        if sid in slide_coord_map_by_slide:
            continue
        cmap = spatial_utils.load_histology_coord_map_from_source(src)
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

    # Reference priors
    expr_ref_torch_map = {}
    if use_avgexp and imputed_refs:
        ref_counts = []
        ref_stack = []
        for slide_id, df_ref_tmp in expr_ref_map.items():
            df_aligned = df_ref_tmp.reindex(columns=gene_names)
            ref_counts.append(df_aligned.shape[0])
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

        ref_stack_arr = np.stack(ref_stack, axis=0)
        expr_ref_mean = np.nanmean(ref_stack_arr, axis=0)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        logging.info("Using avgexp with %d reference(s) per slide", n_ref)
    elif use_avgexp:
        expr_ref_mean = np.zeros((1, len(gene_names)), dtype=np.float32)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        n_ref = 1
        logging.warning("Avgexp enabled but no references loaded; falling back to zeros.")
    else:
        n_ref = None
        expr_ref_torch = None

    n_genes = len(gene_names)
    logging.info("%d genes (union)", n_genes)

    fp_out = os.path.join(experiment_path, "genes.txt")
    with open(fp_out, "w") as f:
        for line in gene_names:
            f.write(f"{line}\n")

    framework_name = f"{model_framework.Framework.__module__}.{model_framework.Framework.__name__}"
    try:
        model = model_framework.Framework(
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
        model = model_framework.Framework(
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
        model.completion_head = panel_completion.PanelCompletionHead(
            n_genes,
            hidden_dim=panel_hidden_dim,
            dropout=panel_dropout,
            use_morph=panel_use_morph,
            morph_gate_init=panel_morph_gate_init,
        )

    try:
        fwd_params = inspect.signature(model.forward).parameters
        supports_cell_graph = all(
            k in fwd_params for k in ("coords_cells", "cell_edge_index", "cell_patch_ids")
        )
    except Exception:
        supports_cell_graph = False
    logging.info("Cell-graph support: %s", supports_cell_graph)
    model.to(device)

    # Datasets, TMA selection, and loaders
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
        avgexp_val_df_by_slide = reference_utils.build_train_region_avgexp_df_by_slide(
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
                expr_ref_torch_val = torch.from_numpy(
                    np.nanmean(np.stack(ref_stack_val, axis=0), axis=0)
                ).float().to(device)
                logging.info(
                    "Validation avgexp refs use train-region-only statistics for %d train slide(s).",
                    len(ref_stack_val),
                )
        else:
            logging.warning(
                "Validation train-region avgexp refs could not be built; using standard refs.",
            )

    immune_sampler_boost = float(getattr(opts.training, "immune_sampler_boost", 1.0))
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
        ds = dataset_input_tma.DataProcessingUnion(
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
        batches = sampler_utils.slide_batch_sampler(
            train_datasets,
            opts.training.batch_size,
            opts.training,
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
    elif getattr(train_dataset, "patch_weights", None) is not None:
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
            ds = dataset_input_tma.DataProcessingUnion(
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
                "batch_sampler": sampler_utils.slide_batch_sampler(
                    datasets,
                    opts.training.batch_size,
                    opts.training,
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

    svg_topk = (20, 50)
    svg_knn_k = 8
    svg_sample_cap = 3000
    val_svg_rank_indices_by_slide = spatial_utils.compute_svg_rank_gene_indices_by_slide(
        sources_trainval,
        opts.regions_val,
        config.fold_id,
        mode_name="val",
        gene_names=gene_names,
        k_neighbors=svg_knn_k,
        sample_cap=svg_sample_cap,
    )
    ext_svg_rank_indices_by_slide = spatial_utils.compute_svg_rank_gene_indices_by_slide(
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

    # Optimizer, resume, and losses
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training.learning_rate,
        betas=(opts.training.beta1, opts.training.beta2),
        weight_decay=opts.training.weight_decay,
        eps=opts.training.eps,
    )

    if config.resume_epoch != 0:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

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
        logging.info("Loaded %s", load_path)

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
            logging.info("Loaded %s", load_path)
        except:
            logging.warning("Optimizer state dict not found")

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
    loss_logits = nn.MSELoss(reduction="mean")

    zero_weight = float(getattr(opts.training, "zero_weight", 0.1))
    zero_threshold = float(getattr(opts.training, "zero_threshold", 0.0))
    pearson_loss_weight = float(getattr(opts.training, "pearson_loss_weight", 1.0))
    expr_ct_embed_loss_weight = float(
        getattr(opts.training, "expr_ct_embed_loss_weight", 1.0)
    )
    logits_loss_weight = float(getattr(opts.training, "logits_loss_weight", 1.0))
    logging.info(
        "Loss setup: pearson=%.3f expr_ct_embed_w=%.3f "
        "logits_w=%.3f expr_ct_embed_internal=100",
        pearson_loss_weight,
        expr_ct_embed_loss_weight,
        logits_loss_weight,
    )

    def expr_loss_weighted(pred, target, mask=None):
        """
        Expression loss = zero-aware weighted MSE with existing missing-gene masking.
        """
        w_zero = pred.new_tensor(zero_weight)
        w_one = pred.new_tensor(1.0)
        w = torch.where(target > zero_threshold, w_one, w_zero)
        if mask is not None:
            w = w * mask
        mse_num = ((pred - target) ** 2 * w).sum()
        mse_den = w.sum().clamp_min(1e-8)
        loss_mse_val = mse_num / mse_den

        return loss_mse_val

    def masked_mse(pred, target, mask):
        return expr_loss_weighted(pred, target, mask)

    def masked_var(x, mask):
        if mask is None:
            return torch.var(x, unbiased=False)
        m = mask.bool()
        if not m.any():
            return torch.tensor(0.0, device=x.device)
        vals = x[m]
        return torch.var(vals, unbiased=False)

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

    # Training loop
    for epoch in range(initial_epoch, total_epochs):
        logging.info("Epoch: %d", epoch + 1)
        model.train()
        if hasattr(model, "set_epoch_progress"):
            total_eps = max(total_epochs - 1, 1)
            model.set_epoch_progress(epoch / total_eps)

        optimizer.param_groups[0]["lr"] = opts.training.learning_rate * (
            1 - epoch / total_epochs
        )

        loss_epoch = 0
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
                graph = graph_utils.build_cell_graph(
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

            batch_expr_mask_pc = tensor_utils.flatten_expr_mask(batch_expr_mask, batch_n_cells)

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

            aux_main = getattr(model, "last_aux_losses", {}) or {}
            ref_base_main = aux_main.get("expr_ref_base")
            pred_expr_for_loss = out_expr
            target_expr_for_loss = batch_expr_pc
            if use_expr_baseline and baseline_torch is not None:
                pred_expr_for_loss = pred_expr_for_loss - baseline_torch
                target_expr_for_loss = target_expr_for_loss - baseline_torch

            loss_expr_val = masked_mse(pred_expr_for_loss, target_expr_for_loss, batch_expr_mask_pc)
            loss_map_val = loss_map(out_map, batch_type_patch)

            loss_panel_completion_val = torch.tensor(0.0, device=device)
            if (
                panel_completion_enabled
                and panel_completion_loss_weight > 0
                and hasattr(model, "completion_head")
                and model.completion_head is not None
            ):
                expr_true_pc = tensor_utils.flatten_expr(batch_expr, batch_n_cells)
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

            pearson_weight_mult = pearson_loss_weight
            if pearson_weight_mult > 0:
                loss_pearson_val = metric_utils.masked_pearson(
                    pred_expr_for_loss,
                    target_expr_for_loss,
                    batch_expr_mask_pc,
                )
            else:
                loss_pearson_val = torch.tensor(0.0, device=device)

            var_pred = masked_var(out_expr, batch_expr_mask_pc)
            var_label = masked_var(batch_expr_pc, batch_expr_mask_pc)
            var_w_cfg = float(getattr(opts.training, "expr_var_penalty_weight", 0.0))
            if var_w_cfg > 0:
                loss_var_val = var_w_cfg * torch.abs(var_pred - var_label)
            else:
                loss_var_val = torch.tensor(0.0, device=device)

            aux_losses = getattr(model, "last_aux_losses", {})
            loss_vq_val = aux_losses.get(
                "vq_patch", torch.tensor(0.0, device=device)
            )

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
                + pearson_weight_mult * loss_pearson_val
                + loss_var_val
                + loss_vq_val
            )

            loss_total = loss.detach()

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

            if hasattr(pbar, "set_description"):
                pbar.set_description(f"loss: {loss_total:.4f}")

            optimizer.step()

        logging.info(
            "Epoch[%d/%d], Loss:%.4f",
            epoch + 1,
            opts.training.total_epochs,
            loss_epoch,
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

        # Evaluation and checkpoint selection
        val_metrics = None
        if val_dataloader is not None:
            val_metrics = evaluation_utils.evaluate_validation(
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
            metric_utils.log_gene_pcc_epoch(
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
            ext_metrics = evaluation_utils.evaluate_validation(
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
            metric_utils.log_gene_pcc_epoch(
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
        if primary_improved or (
            primary_tied and val_ct_macro > (best_val_ct_macro + best_metric_eps)
        ):
            best_val_gene_pooled = val_gene_pooled_mean
            best_val_ct_macro = val_ct_macro
            best_epoch = int(epoch + 1)
            best_ckpt_path = ckpt_model_path

    # Best-checkpoint summary
    strict_best = {
        "selection_metric": "pearson_gene_pooled_mean",
        "selection_constraint": "none_ct_macro_reported_only",
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

"""Validation/evaluation utilities."""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import dataio.tensors as tensor_utils
import model.graph as graph_utils
import utils.metrics as metric_utils


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
            _,
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
                graph = graph_utils.build_cell_graph(
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
                _,
                batch_ct_pc,
                out_expr,
                _,
                _,
                _,
                _,
                _,
                _,
                batch_expr_pc,
                _,
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
            batch_expr_mask_pc = tensor_utils.flatten_expr_mask(batch_expr_mask, batch_n_cells)

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
            expr_true_pc = tensor_utils.flatten_expr(batch_expr, batch_n_cells)
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
                        if (
                            ref_base is not None
                            and isinstance(ref_base, torch.Tensor)
                            and ref_base.shape == out_expr.shape
                        )
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
            dist_entry = {"all": metric_utils.summarize_gene_pcc_distribution(corr)}
            rank_order = svg_rank_gene_indices_by_slide.get(int(sid))
            if rank_order is not None:
                rank_order = np.asarray(rank_order, dtype=np.int64).reshape(-1)
            for k_svg in svg_topk:
                key = f"svg{k_svg}"
                if rank_order is None or rank_order.size == 0:
                    dist_entry[key] = metric_utils.summarize_gene_pcc_distribution(np.array([], dtype=np.float64))
                    continue
                idx_top = rank_order[: min(int(k_svg), int(rank_order.size))]
                idx_top = idx_top[(idx_top >= 0) & (idx_top < corr.shape[0])]
                if idx_top.size == 0:
                    dist_entry[key] = metric_utils.summarize_gene_pcc_distribution(np.array([], dtype=np.float64))
                else:
                    dist_entry[key] = metric_utils.summarize_gene_pcc_distribution(corr[idx_top])
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
            "prop_pred": (
                (preds / max(preds.sum(), 1)).tolist()
                if preds.sum() > 0
                else [0.0 for _ in range(n_classes)]
            ),
            "acc_per_class": acc_per_class_slide.tolist(),
            "acc_micro": float(correct.sum() / max(total, 1)),
            "acc_macro": float(np.mean(acc_per_class_slide[supported])) if supported.any() else 0.0,
        }
    metrics["ct_per_slide"] = ct_per_slide

    model.train()
    return metrics

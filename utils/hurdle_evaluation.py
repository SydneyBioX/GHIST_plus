"""Unique-cell, cohort-gated evaluation for HurdleFramework."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch

import dataio.tensors as tensor_utils
import model.graph as graph_utils
from model.hurdle_distribution import deterministic_hurdle_prediction


def aggregate_unique_hurdle_rows(
    cell_ids,
    signed_mu,
    occurrence_logits,
    target=None,
    mask=None,
):
    """Mean-aggregate duplicate cell rows before cohort gating."""

    ids = np.asarray(cell_ids, dtype=np.int64)
    mu = np.asarray(signed_mu, dtype=np.float32)
    logits = np.asarray(occurrence_logits, dtype=np.float32)
    if mu.shape != logits.shape or mu.ndim != 2 or mu.shape[0] != ids.size:
        raise ValueError("ids, signed_mu, and occurrence_logits have incompatible shapes")
    order = np.argsort(ids, kind="stable")
    ids, mu, logits = ids[order], mu[order], logits[order]
    unique_ids, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    mean_mu = np.add.reduceat(mu, starts, axis=0) / counts[:, None]
    # Overlapping patch views are repeated estimates of q. Average q, not
    # logits, so K=sum(q) remains probability-calibrated after deduplication.
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    mean_probability = np.add.reduceat(probability, starts, axis=0) / counts[:, None]
    mean_probability = np.clip(mean_probability, 1e-7, 1.0 - 1e-7)
    mean_logits = np.log(mean_probability) - np.log1p(-mean_probability)

    mean_target = None
    observed = None
    if target is not None:
        target = np.asarray(target, dtype=np.float32)[order]
        observed_rows = (
            np.ones_like(target, dtype=np.float32)
            if mask is None
            else (np.asarray(mask)[order] > 0).astype(np.float32)
        )
        finite = np.isfinite(target)
        observed_rows *= finite.astype(np.float32)
        safe_target = np.where(finite, target, 0.0)
        target_sum = np.add.reduceat(safe_target * observed_rows, starts, axis=0)
        target_count = np.add.reduceat(observed_rows, starts, axis=0)
        observed = target_count > 0
        mean_target = target_sum / np.maximum(target_count, 1.0)
    return unique_ids, mean_mu, mean_logits, mean_target, observed, counts


def cohort_gate_numpy(signed_mu, occurrence_logits):
    """Apply the fixed top-K rule once to a complete unique-cell cohort."""

    mu_t = torch.from_numpy(np.asarray(signed_mu, dtype=np.float32))
    logit_t = torch.from_numpy(np.asarray(occurrence_logits, dtype=np.float32))
    return deterministic_hurdle_prediction(mu_t, logit_t).numpy()


def hurdle_gate_counts(pred, occurrence_logits):
    """Return requested top-K and effective positive counts per gene."""

    pred = np.asarray(pred)
    logits = np.asarray(occurrence_logits)
    if pred.shape != logits.shape or pred.ndim != 2:
        raise ValueError("pred and occurrence_logits must share [n_cells,n_genes] shape")
    # Match cohort_gate_numpy/deterministic_hurdle_prediction exactly.  A
    # separate NumPy sigmoid/reduction can round a probability sum on the
    # opposite side of x.5 and falsely report that the gate selected K+1.
    logit_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    requested = (
        torch.sigmoid(logit_t)
        .sum(dim=0)
        .round()
        .long()
        .clamp(0, pred.shape[0])
        .numpy()
    )
    effective = np.count_nonzero(pred > 0, axis=0).astype(np.int64)
    if np.any(effective > requested):
        raise RuntimeError("effective positives cannot exceed requested top-K")
    return requested, effective


def summarize_hurdle_gate_counts(requested, effective):
    """JSON-safe diagnostics for ReLU-induced top-K positive shortfall."""

    requested = np.asarray(requested, dtype=np.int64)
    effective = np.asarray(effective, dtype=np.int64)
    if requested.shape != effective.shape or requested.ndim != 1:
        raise ValueError("requested and effective must share [n_genes] shape")
    shortfall = requested - effective
    requested_total = int(requested.sum())
    effective_total = int(effective.sum())
    return {
        "hurdle_requested_positive_total": requested_total,
        "hurdle_effective_positive_total": effective_total,
        "hurdle_positive_shortfall_total": int(shortfall.sum()),
        "hurdle_effective_positive_fraction_of_requested": (
            float(effective_total / requested_total) if requested_total > 0 else 1.0
        ),
        "hurdle_genes_with_positive_shortfall": int(np.count_nonzero(shortfall)),
        "hurdle_genes_with_positive_shortfall_fraction": float(
            np.mean(shortfall > 0) if shortfall.size else 0.0
        ),
        "hurdle_mean_positive_shortfall_per_gene": float(
            np.mean(shortfall) if shortfall.size else 0.0
        ),
    }


def _wasserstein_1d(x, y):
    """Exact empirical one-dimensional Wasserstein-1 distance."""

    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return np.nan
    values = np.sort(np.concatenate([x, y]))
    if values.size <= 1:
        return 0.0
    deltas = np.diff(values)
    cdf_x = np.searchsorted(x, values[:-1], side="right") / x.size
    cdf_y = np.searchsorted(y, values[:-1], side="right") / y.size
    return float(np.sum(np.abs(cdf_x - cdf_y) * deltas))


def hurdle_matrix_metrics(pred, target, mask=None, gene_names=None):
    """Distribution and gene-PCC metrics on the one saved hard matrix."""

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pred and target must share [n_cells,n_genes] shape")
    observed = (
        np.ones_like(target, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    observed &= np.isfinite(target) & np.isfinite(pred)

    pcc = np.full(pred.shape[1], np.nan, dtype=np.float64)
    w1 = np.full_like(pcc, np.nan)
    zero_gap = np.full_like(pcc, np.nan)
    positive_w1 = np.full_like(pcc, np.nan)
    q_gaps = {q: np.full_like(pcc, np.nan) for q in (0.5, 0.9, 0.99)}
    for gene_idx in range(pred.shape[1]):
        keep = observed[:, gene_idx]
        x = pred[keep, gene_idx]
        y = target[keep, gene_idx]
        if x.size == 0:
            continue
        w1[gene_idx] = _wasserstein_1d(x, y)
        zero_gap[gene_idx] = abs(np.mean(x <= 0) - np.mean(y <= 0))
        if x.size > 1 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            pcc[gene_idx] = np.corrcoef(x, y)[0, 1]
        xp, yp = x[x > 0], y[y > 0]
        if xp.size and yp.size:
            positive_w1[gene_idx] = _wasserstein_1d(xp, yp)
            for q in q_gaps:
                q_gaps[q][gene_idx] = abs(np.quantile(xp, q) - np.quantile(yp, q))

    valid_pcc = np.isfinite(pcc)
    flat_keep = observed
    pred_zero = float(np.mean(pred[flat_keep] <= 0)) if flat_keep.any() else np.nan
    target_zero = float(np.mean(target[flat_keep] <= 0)) if flat_keep.any() else np.nan
    metrics = {
        "pearson_gene_pooled_mean": float(np.nanmean(pcc)) if valid_pcc.any() else 0.0,
        "pearson_gene_pooled_median": float(np.nanmedian(pcc)) if valid_pcc.any() else 0.0,
        "pearson_gene_pooled_max": float(np.nanmax(pcc)) if valid_pcc.any() else 0.0,
        "pearson_gene_pooled_p95": float(np.nanpercentile(pcc, 95)) if valid_pcc.any() else 0.0,
        "pearson_gene_pooled_n_genes": int(valid_pcc.sum()),
        "pearson_gene_valid_coverage": float(valid_pcc.mean()),
        "hurdle_pred_zero_fraction": pred_zero,
        "hurdle_target_zero_fraction": target_zero,
        "hurdle_zero_fraction_gap": abs(pred_zero - target_zero),
        "hurdle_mean_per_gene_zero_gap": float(np.nanmean(zero_gap)),
        "hurdle_mean_per_gene_w1": float(np.nanmean(w1)),
        "hurdle_positive_w1_mean": float(np.nanmean(positive_w1)),
        "hurdle_positive_q50_gap_mean": float(np.nanmean(q_gaps[0.5])),
        "hurdle_positive_q90_gap_mean": float(np.nanmean(q_gaps[0.9])),
        "hurdle_positive_q99_gap_mean": float(np.nanmean(q_gaps[0.99])),
        "hurdle_n_unique_cells": int(pred.shape[0]),
    }
    names = list(gene_names) if gene_names is not None else [str(i) for i in range(pred.shape[1])]
    metrics["per_gene"] = {
        names[i]: {
            "pearson": None if not np.isfinite(pcc[i]) else float(pcc[i]),
            "w1": None if not np.isfinite(w1[i]) else float(w1[i]),
            "zero_gap": None if not np.isfinite(zero_gap[i]) else float(zero_gap[i]),
            "positive_w1": None if not np.isfinite(positive_w1[i]) else float(positive_w1[i]),
        }
        for i in range(pred.shape[1])
    }
    return metrics


def _broadcast_panel_mask(mask, shape, *, name):
    """Validate and broadcast a gene or cell-by-gene mask."""

    array = np.asarray(mask)
    if array.ndim == 1:
        if array.shape[0] != shape[1]:
            raise ValueError(f"{name} gene dimension does not match predictions")
        return np.broadcast_to(array > 0, shape)
    if array.ndim == 2 and array.shape == shape:
        return array > 0
    raise ValueError(f"{name} must have [n_genes] or [n_cells,n_genes] shape")


def _panel_mapping_value(mapping, slide_id):
    if slide_id in mapping:
        return mapping[slide_id]
    string_id = str(slide_id)
    if string_id in mapping:
        return mapping[string_id]
    return None


def sanitize_hurdle_panel_holdout_inputs(
    batch_expr,
    batch_n_cells,
    batch_ct,
    expr_ref_batch,
    holdout_mask,
):
    """Replace fixed holdout targets before the base model sees them.

    Hidden values use the slide's non-leaky cell-type reference, or zero when
    no reference is available. The raw ``batch_expr`` tensor is never mutated.
    """

    if not isinstance(batch_expr, torch.Tensor) or batch_expr.ndim != 3:
        raise ValueError("batch_expr must have [batch,cells,genes] shape")
    holdout = (
        np.zeros(batch_expr.shape[2], dtype=np.float32)
        if holdout_mask is None
        else np.asarray(holdout_mask)
    )
    if holdout.ndim != 1 or holdout.size != batch_expr.shape[2]:
        raise ValueError("holdout_mask must have [n_genes] shape")
    fixed_hidden = torch.as_tensor(
        holdout > 0,
        dtype=torch.bool,
        device=batch_expr.device,
    )
    if not bool(fixed_hidden.any().item()):
        return batch_expr

    sanitized = batch_expr.clone()
    has_reference = (
        isinstance(expr_ref_batch, torch.Tensor)
        and expr_ref_batch.ndim == 2
        and expr_ref_batch.shape[0] > 0
        and expr_ref_batch.shape[1] == batch_expr.shape[2]
    )
    for batch_index in range(sanitized.shape[0]):
        n_valid = int(batch_n_cells[batch_index].item())
        if n_valid <= 0:
            continue
        values = sanitized[batch_index, :n_valid]
        if has_reference:
            cell_types = batch_ct[batch_index, :n_valid].long().clamp(
                min=0,
                max=expr_ref_batch.shape[0] - 1,
            )
            reference = expr_ref_batch[cell_types]
            values[:, fixed_hidden] = reference[:, fixed_hidden]
        else:
            values[:, fixed_hidden] = 0.0
    return sanitized


def new_panel_completion_holdout_accumulator():
    """Create the original B holdout-metric running state."""

    return {
        "holdout_sse": 0.0,
        "holdout_sae": 0.0,
        "holdout_n": 0.0,
        "stats_by_slide": {},
    }


def update_panel_completion_holdout_accumulator(
    accumulator,
    slide_id,
    prediction,
    target,
    holdout_mask,
):
    """Accumulate one emitted batch exactly as the original B evaluator did."""

    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.shape != pred.shape:
        raise ValueError("panel prediction and target must share [rows,genes]")
    holdout = _broadcast_panel_mask(
        holdout_mask,
        pred.shape,
        name="holdout_mask",
    ).astype(np.float64)
    difference = pred - target
    accumulator["holdout_sse"] += float(((difference**2) * holdout).sum())
    accumulator["holdout_sae"] += float((np.abs(difference) * holdout).sum())
    accumulator["holdout_n"] += float(holdout.sum())
    update = {
        "count": holdout.sum(axis=0),
        "sum_pred": (pred * holdout).sum(axis=0),
        "sum_targ": (target * holdout).sum(axis=0),
        "sum_pred2": ((pred**2) * holdout).sum(axis=0),
        "sum_targ2": ((target**2) * holdout).sum(axis=0),
        "sum_xy": ((pred * target) * holdout).sum(axis=0),
    }
    slide_stats = accumulator["stats_by_slide"].setdefault(
        int(slide_id),
        {key: np.zeros_like(value) for key, value in update.items()},
    )
    for key, value in update.items():
        slide_stats[key] += value


def panel_completion_holdout_metrics(
    accumulator,
    holdout_mask_by_slide,
    *,
    gene_names=None,
    epoch=None,
    per_gene_dir=None,
):
    """Reproduce the original panel head's emitted-row holdout metrics.

    This path intentionally does not deduplicate cells and does not apply the
    hurdle occurrence gate. Panel completion remains a direct completed-
    expression imputation task; the canonical hurdle matrix is evaluated by
    the separate base path.
    """

    metrics = {"holdout_mse": 0.0, "holdout_mae": 0.0}
    stats_by_slide = accumulator["stats_by_slide"]
    if not stats_by_slide:
        return metrics
    holdout_n = float(accumulator["holdout_n"])
    if holdout_n > 0:
        metrics["holdout_mse"] = float(accumulator["holdout_sse"] / holdout_n)
        metrics["holdout_mae"] = float(accumulator["holdout_sae"] / holdout_n)
    if gene_names is None:
        return metrics

    names = list(gene_names)
    n_genes = next(iter(stats_by_slide.values()))["count"].shape[0]
    if len(names) != n_genes:
        raise ValueError("gene_names length does not match panel predictions")

    def _correlations(stats):
        count = stats["count"]
        denom_x = stats["sum_pred2"] - stats["sum_pred"] ** 2 / np.maximum(
            count, 1e-8
        )
        denom_y = stats["sum_targ2"] - stats["sum_targ"] ** 2 / np.maximum(
            count, 1e-8
        )
        numerator = stats["sum_xy"] - (
            stats["sum_pred"] * stats["sum_targ"]
        ) / np.maximum(count, 1e-8)
        denominator = np.sqrt(
            np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0)
        )
        correlation = np.full_like(numerator, np.nan, dtype=np.float64)
        valid = count > 1
        correlation[valid] = numerator[valid] / np.maximum(
            denominator[valid], 1e-8
        )
        return correlation

    per_slide_mean = {}
    per_slide_gene = {}
    per_slide_gene_files = {}
    all_slide_correlations = []
    for slide_id, stats in stats_by_slide.items():
        correlation = _correlations(stats)
        finite = correlation[np.isfinite(correlation)]
        per_slide_mean[slide_id] = (
            float(np.mean(finite)) if finite.size else float("nan")
        )
        all_slide_correlations.extend(finite.tolist())
        holdout_vector = np.asarray(
            _panel_mapping_value(holdout_mask_by_slide, slide_id)
        ).reshape(-1)
        holdout_indices = np.sort(np.flatnonzero(holdout_vector > 0))
        per_slide_gene[slide_id] = {
            str(names[index]): (
                float(correlation[index])
                if np.isfinite(correlation[index])
                else float("nan")
            )
            for index in holdout_indices
        }
        if per_gene_dir and epoch is not None:
            os.makedirs(per_gene_dir, exist_ok=True)
            output_path = os.path.join(
                per_gene_dir,
                f"slide{slide_id}_epoch{epoch}_holdout_per_gene_pearson.csv",
            )
            pd.DataFrame(
                {
                    "gene": [names[index] for index in holdout_indices],
                    "pearson": correlation[holdout_indices],
                    "n_cells": stats["count"][holdout_indices],
                }
            ).to_csv(output_path, index=False)
            per_slide_gene_files[slide_id] = output_path

    pooled = {
        key: np.sum([stats[key] for stats in stats_by_slide.values()], axis=0)
        for key in next(iter(stats_by_slide.values()))
    }
    pooled_correlation = _correlations(pooled)
    pooled_finite = pooled_correlation[np.isfinite(pooled_correlation)]
    metrics.update(
        {
            "holdout_gene_pearson_mean": (
                float(np.mean(all_slide_correlations))
                if all_slide_correlations
                else 0.0
            ),
            "holdout_gene_pearson_per_slide_mean": per_slide_mean,
            "holdout_gene_pooled_mean": (
                float(np.mean(pooled_finite)) if pooled_finite.size else 0.0
            ),
            "holdout_gene_pooled_median": (
                float(np.median(pooled_finite)) if pooled_finite.size else 0.0
            ),
            "holdout_gene_pooled_max": (
                float(np.max(pooled_finite)) if pooled_finite.size else 0.0
            ),
            "holdout_gene_pooled_p95": (
                float(np.percentile(pooled_finite, 95))
                if pooled_finite.size
                else 0.0
            ),
            "holdout_gene_pooled_n_genes": int(pooled_finite.size),
            "holdout_pearson_per_gene": per_slide_gene,
        }
    )
    if per_slide_gene_files:
        metrics["holdout_pearson_per_gene_files"] = per_slide_gene_files
    return metrics


def fixed_gt_svg_validation_metrics(
    diagnostic_slides,
    fixed_svg_cohort_by_slide,
    *,
    svg_topk=(20, 50),
):
    """Compute strict Figure3 Top-K metrics on the frozen VAL cohort."""

    from utils.hurdle_metrics import within_validation_cmd, within_validation_ssim

    expected_slides = set(int(value) for value in fixed_svg_cohort_by_slide)
    observed_slides = set(int(slide["slide_id"]) for slide in diagnostic_slides)
    if observed_slides != expected_slides:
        raise RuntimeError(
            "Frozen SVG validation slide mismatch: "
            f"expected={sorted(expected_slides)} observed={sorted(observed_slides)}"
        )
    buckets = {
        int(k_value): {
            "pcc": [],
            "ssim": [],
            "cmd": [],
            "per_slide": {},
            "full": True,
        }
        for k_value in svg_topk
    }

    for slide in diagnostic_slides:
        slide_id = int(slide["slide_id"])
        frozen = fixed_svg_cohort_by_slide[slide_id]
        actual_ids = np.asarray(slide["cell_ids"], dtype=np.int64)
        if np.unique(actual_ids).size != actual_ids.size:
            raise RuntimeError(f"Duplicate evaluated cell IDs on validation slide {slide_id}")
        actual_row = {int(cell_id): index for index, cell_id in enumerate(actual_ids)}
        frozen_ids = np.asarray(frozen["cell_ids"], dtype=np.int64)
        if set(actual_row) != set(frozen_ids.tolist()):
            missing = sorted(set(frozen_ids.tolist()).difference(actual_row))
            extra = sorted(set(actual_row).difference(frozen_ids.tolist()))
            raise RuntimeError(
                f"Frozen SVG validation cell mismatch on slide {slide_id}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        rows = np.asarray([actual_row[int(cell_id)] for cell_id in frozen_ids], dtype=np.int64)
        model_indices = np.asarray(
            frozen["model_gene_indices_gt_order"], dtype=np.int64
        )
        prediction = np.asarray(slide["prediction"], dtype=np.float64)[rows][:, model_indices]
        target = np.asarray(slide["target"], dtype=np.float64)[rows][:, model_indices]
        observed = np.asarray(slide["observed"], dtype=bool)[rows][:, model_indices]

        pcc = np.full(target.shape[1], np.nan, dtype=np.float64)
        for gene_index in range(target.shape[1]):
            valid = (
                observed[:, gene_index]
                & np.isfinite(target[:, gene_index])
                & np.isfinite(prediction[:, gene_index])
            )
            if int(valid.sum()) < 2:
                continue
            target_gene = target[valid, gene_index]
            prediction_gene = prediction[valid, gene_index]
            if np.std(target_gene) == 0 or np.std(prediction_gene) == 0:
                continue
            value = np.corrcoef(target_gene, prediction_gene)[0, 1]
            if np.isfinite(value):
                pcc[gene_index] = float(value)

        rank_order = np.asarray(
            frozen["giotto_order_gt_positions"], dtype=np.int64
        )
        gt_gene_names = list(frozen["gene_names_gt_order"])
        max_k = max(int(value) for value in svg_topk)
        ranked_for_ssim = rank_order[:max_k]
        ssim_result = within_validation_ssim(
            target[:, ranked_for_ssim],
            prediction[:, ranked_for_ssim],
            np.asarray(frozen["coordinates_xy"], dtype=np.float64),
            observed[:, ranked_for_ssim],
        )
        ssim_ranked = np.asarray(ssim_result["scores"], dtype=np.float64)

        for k_value, bucket in buckets.items():
            top = rank_order[:k_value]
            rank_is_full = (
                top.size == k_value
                and np.unique(top).size == k_value
                and np.all((top >= 0) & (top < target.shape[1]))
            )
            pcc_top = pcc[top] if rank_is_full else np.asarray([], dtype=np.float64)
            ssim_top = (
                ssim_ranked[:k_value]
                if rank_is_full
                else np.asarray([], dtype=np.float64)
            )
            if rank_is_full:
                cmd_result = within_validation_cmd(
                    target[:, top],
                    prediction[:, top],
                    observed[:, top],
                    gene_names=[gt_gene_names[index] for index in top],
                )
                cmd_value = float(cmd_result["cmd"])
            else:
                cmd_result = {
                    "status": "undefined_incomplete_fixed_svg_rank",
                    "fixed_gt_valid_gene_count": 0,
                    "prediction_valid_fixed_gt_coverage": 0.0,
                }
                cmd_value = float("nan")
            pcc_count = int(np.isfinite(pcc_top).sum())
            ssim_count = int(np.isfinite(ssim_top).sum())
            full = bool(
                rank_is_full
                and pcc_count == k_value
                and ssim_count == k_value
                and np.isfinite(cmd_value)
                and int(cmd_result["fixed_gt_valid_gene_count"]) == k_value
                and float(cmd_result["prediction_valid_fixed_gt_coverage"]) == 1.0
            )
            bucket["pcc"].extend(pcc_top.astype(float).tolist())
            bucket["ssim"].extend(ssim_top.astype(float).tolist())
            if np.isfinite(cmd_value):
                bucket["cmd"].append(cmd_value)
            bucket["full"] = bool(bucket["full"] and full)
            bucket["per_slide"][str(slide_id)] = {
                "frozen_sha256": frozen["frozen_sha256"],
                "ranked_genes": [gt_gene_names[index] for index in top],
                "pcc_values": pcc_top.astype(float).tolist(),
                "ssim_values": ssim_top.astype(float).tolist(),
                "pcc_valid_gene_count": pcc_count,
                "ssim_valid_gene_count": ssim_count,
                "cmd": cmd_value,
                "cmd_status": cmd_result["status"],
                "full_k_of_k": full,
            }

    output = {
        "rank_source": "frozen validation GT only; exact Figure3.ipynb Giotto implementation",
        "selection_scope": "VAL only; external evaluation is never used",
        "frozen_slide_sha256": {
            str(slide_id): fixed_svg_cohort_by_slide[slide_id]["frozen_sha256"]
            for slide_id in sorted(fixed_svg_cohort_by_slide)
        },
    }
    slide_count = len(diagnostic_slides)
    for k_value, bucket in buckets.items():
        pcc_values = np.asarray(bucket["pcc"], dtype=np.float64)
        ssim_values = np.asarray(bucket["ssim"], dtype=np.float64)
        cmd_values = np.asarray(bucket["cmd"], dtype=np.float64)
        finite_pcc = pcc_values[np.isfinite(pcc_values)]
        finite_ssim = ssim_values[np.isfinite(ssim_values)]
        finite_cmd = cmd_values[np.isfinite(cmd_values)]
        requested = int(k_value * slide_count)
        output[f"top{k_value}"] = {
            "k": int(k_value),
            "slide_count": int(slide_count),
            "requested_gene_count": requested,
            "pcc_valid_gene_count": int(np.isfinite(pcc_values).sum()),
            "ssim_valid_gene_count": int(np.isfinite(ssim_values).sum()),
            "pcc_median": (
                float(np.median(finite_pcc))
                if finite_pcc.size
                else float("nan")
            ),
            "pcc_max": float(np.max(finite_pcc)) if finite_pcc.size else float("nan"),
            "pcc_min": float(np.min(finite_pcc)) if finite_pcc.size else float("nan"),
            "ssim_median": (
                float(np.median(finite_ssim))
                if finite_ssim.size
                else float("nan")
            ),
            "ssim_max": float(np.max(finite_ssim)) if finite_ssim.size else float("nan"),
            "ssim_min": float(np.min(finite_ssim)) if finite_ssim.size else float("nan"),
            "cmd": (
                float(np.median(finite_cmd))
                if finite_cmd.size
                else float("nan")
            ),
            "cmd_median": (
                float(np.median(finite_cmd)) if finite_cmd.size else float("nan")
            ),
            "cmd_max": float(np.max(finite_cmd)) if finite_cmd.size else float("nan"),
            "cmd_min": float(np.min(finite_cmd)) if finite_cmd.size else float("nan"),
            "pcc_values": pcc_values.astype(float).tolist(),
            "ssim_values": ssim_values.astype(float).tolist(),
            "cmd_values_by_slide": cmd_values.astype(float).tolist(),
            "full_k_of_k": bool(
                bucket["full"]
                and np.isfinite(pcc_values).sum() == requested
                and np.isfinite(ssim_values).sum() == requested
                and np.isfinite(cmd_values).sum() == slide_count
            ),
            "per_slide": bucket["per_slide"],
        }
    return output


def evaluate_hurdle_validation(
    model,
    dataloader,
    expr_ref_torch,
    device,
    n_classes,
    *,
    expr_scale,
    graph_k=None,
    graph_cross_patch=False,
    graph_cross_patch_k=None,
    graph_cross_patch_radius=None,
    slide_coord_map_by_slide=None,
    expr_ref_torch_map=None,
    gene_names=None,
    epoch=None,
    per_gene_dir=None,
    fixed_svg_cohort_by_slide=None,
    svg_topk=(20, 50),
    panel_completion_enabled=False,
    holdout_mask_by_slide=None,
    **_ignored,
):
    """Evaluate one hard matrix after unique-cell aggregation and per-slide gating."""

    # Kept local so importing the training evaluator adds no plotting imports.
    from utils.hurdle_metrics import (
        COHORT_LABEL,
        select_ssim_coordinate_rows,
        calibration_summary,
        within_validation_cmd,
        within_validation_ssim,
        write_calibration_density_png,
        write_marker_distribution_png,
    )

    model.eval()
    panel_completion_enabled = bool(panel_completion_enabled)
    panel_holdout_accumulator = (
        new_panel_completion_holdout_accumulator()
        if panel_completion_enabled else None
    )
    completion_head = None
    if panel_completion_enabled:
        if holdout_mask_by_slide is None:
            raise ValueError(
                "panel_completion_enabled requires holdout_mask_by_slide"
            )
        completion_head = getattr(model, "completion_head", None)
    expr_ref_torch_map = expr_ref_torch_map or {}
    slide_coord_map_by_slide = slide_coord_map_by_slide or {}
    rows = {}
    ct_true_all, ct_pred_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            (
                batch_nuclei,
                _,
                batch_he_img,
                batch_expr,
                batch_n_cells,
                batch_ct,
                patch_ids,
                batch_expr_mask,
                batch_slide_id,
            ) = batch
            batch_nuclei = batch_nuclei.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_expr_mask = batch_expr_mask.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)
            patch_ids = patch_ids.to(device)
            slide_unique = torch.unique(batch_slide_id)
            if slide_unique.numel() != 1:
                raise RuntimeError("Hurdle evaluation requires one slide per batch")
            slide_id = int(slide_unique.item())
            ref = expr_ref_torch_map.get(slide_id, expr_ref_torch)
            panel_holdout_mask = None
            batch_expr_for_model = batch_expr
            if panel_completion_enabled:
                panel_holdout_mask = _panel_mapping_value(
                    holdout_mask_by_slide, slide_id
                )
                batch_expr_for_model = sanitize_hurdle_panel_holdout_inputs(
                    batch_expr,
                    batch_n_cells,
                    batch_ct,
                    ref,
                    panel_holdout_mask,
                )
            graph = graph_utils.build_cell_graph(
                batch_nuclei,
                patch_ids,
                k_neighbors=graph_k or 6,
                coords_batch=None,
                cell_coord_map=slide_coord_map_by_slide.get(slide_id),
                cross_patch=bool(graph_cross_patch),
                cross_patch_k=graph_cross_patch_k,
                cross_patch_radius=graph_cross_patch_radius,
            )
            output = model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                ref,
                batch_ct,
                batch_expr_for_model,
                patch_ids=patch_ids,
                coords_cells=graph.coords,
                cell_edge_index=graph.edge_index,
                cell_patch_ids=graph.patch_index,
            )
            if output[3].numel() == 0:
                continue
            aux = model.last_aux_losses
            ids = output[13]
            if ids is None:
                continue
            target = tensor_utils.flatten_expr(batch_expr, batch_n_cells)
            observed = tensor_utils.flatten_expr_mask(batch_expr_mask, batch_n_cells)
            entry = rows.setdefault(
                slide_id,
                {"ids": [], "mu": [], "logits": [], "target": [], "mask": []},
            )
            entry["ids"].append(ids.detach().cpu().numpy())
            entry["mu"].append(aux["hurdle_signed_mu"].detach().cpu().numpy())
            entry["logits"].append(aux["hurdle_occurrence_logits"].detach().cpu().numpy())
            entry["target"].append(target.detach().cpu().numpy())
            entry["mask"].append(observed.detach().cpu().numpy())
            if panel_completion_enabled:
                pred_completed = output[3]
                if completion_head is not None:
                    try:
                        ref_base = aux.get("expr_ref_base")
                        ref_base = (
                            ref_base
                            if isinstance(ref_base, torch.Tensor)
                            and ref_base.shape == output[3].shape
                            else torch.zeros_like(output[3])
                        )
                        mask_obs_f = (observed > 0).to(target.dtype)
                        delta_obs = (target - ref_base) * mask_obs_f
                        delta_morph = output[3] - ref_base
                        delta_hat = completion_head(
                            delta_obs,
                            mask_obs_f,
                            delta_morph,
                        )
                        pred_completed = torch.relu(ref_base + delta_hat)
                        pred_completed = (
                            mask_obs_f * target
                            + (1.0 - mask_obs_f) * pred_completed
                        )
                    except Exception as exc:
                        logging.warning(
                            "Panel completion head failed in validation "
                            "(using morph-only preds): %s",
                            exc,
                        )
                if panel_holdout_mask is not None:
                    update_panel_completion_holdout_accumulator(
                        panel_holdout_accumulator,
                        slide_id,
                        pred_completed.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        panel_holdout_mask,
                    )
            if output[0] is not None and output[0].numel() and output[2].numel():
                ct_true_all.append(output[2].detach().cpu().numpy())
                ct_pred_all.append(output[0].argmax(dim=1).detach().cpu().numpy())

    pred_all, target_all, mask_all = [], [], []
    diagnostic_slides = []
    requested_positive_total = None
    effective_positive_total = None
    for slide_id, entry in rows.items():
        unique = aggregate_unique_hurdle_rows(
            np.concatenate(entry["ids"]),
            np.concatenate(entry["mu"]),
            np.concatenate(entry["logits"]),
            np.concatenate(entry["target"]),
            np.concatenate(entry["mask"]),
        )
        unique_ids, mean_mu, mean_logits, mean_target, observed, _ = unique
        pred_slide = cohort_gate_numpy(mean_mu, mean_logits)
        requested, effective = hurdle_gate_counts(pred_slide, mean_logits)
        if requested_positive_total is None:
            requested_positive_total = requested
            effective_positive_total = effective
        else:
            requested_positive_total += requested
            effective_positive_total += effective
        pred_eval = pred_slide / float(expr_scale)
        target_eval = mean_target / float(expr_scale)
        pred_all.append(pred_eval)
        target_all.append(target_eval)
        mask_all.append(observed)
        diagnostic_slides.append(
            {
                "slide_id": int(slide_id),
                "cell_ids": unique_ids,
                "prediction": pred_eval,
                "target": target_eval,
                "observed": observed,
                "requested_positive": requested,
                "effective_positive": effective,
            }
        )
    panel_metrics = {}
    if panel_completion_enabled:
        panel_metrics = panel_completion_holdout_metrics(
            panel_holdout_accumulator,
            holdout_mask_by_slide,
            gene_names=gene_names,
            epoch=epoch,
            per_gene_dir=per_gene_dir,
        )
    if not pred_all:
        if fixed_svg_cohort_by_slide:
            raise RuntimeError("Frozen SVG validation cohort produced no predictions")
        empty_metrics = {
            "pearson_gene_pooled_mean": 0.0,
            "pearson_gene_valid_coverage": 0.0,
            "hurdle_mean_per_gene_w1": float("inf"),
            "ct_accuracy_macro": 0.0,
            "within_val_noncanonical_cohort_label": COHORT_LABEL,
            "within_val_noncanonical_ssim_status": "undefined_empty_validation_cohort",
            "within_val_noncanonical_cmd_status": "undefined_empty_validation_cohort",
        }
        if panel_completion_enabled:
            empty_metrics.update(panel_metrics)
        return empty_metrics

    hard_prediction = np.concatenate(pred_all)
    hard_target = np.concatenate(target_all)
    hard_observed = np.concatenate(mask_all)
    names = (
        list(gene_names)
        if gene_names is not None
        else [str(index) for index in range(hard_prediction.shape[1])]
    )
    metrics = hurdle_matrix_metrics(
        hard_prediction,
        hard_target,
        hard_observed,
        names,
    )
    metrics["hurdle_aggregation_mode"] = "mean_duplicate_then_per_slide_topk"
    metrics.update(
        summarize_hurdle_gate_counts(
            requested_positive_total, effective_positive_total
        )
    )
    shortfall_per_gene = requested_positive_total - effective_positive_total
    for gene_idx, gene_name in enumerate(metrics["per_gene"]):
        metrics["per_gene"][gene_name].update(
            {
                "requested_positive": int(requested_positive_total[gene_idx]),
                "effective_positive": int(effective_positive_total[gene_idx]),
                "positive_shortfall": int(shortfall_per_gene[gene_idx]),
            }
        )

    # Evaluation-only diagnostics. These are the current within-validation
    # cohort, not the immutable cohorts used for any canonical Figure 2 claim.
    n_genes = hard_prediction.shape[1]
    ssim_scores_by_slide = []
    ssim_protocol_by_slide = {}
    for slide in diagnostic_slides:
        slide_id = int(slide["slide_id"])
        coordinate_coverage = None
        try:
            ssim_rows, coordinates_xy, coordinate_coverage = (
                select_ssim_coordinate_rows(
                    slide["cell_ids"], slide_coord_map_by_slide.get(slide_id)
                )
            )
            if ssim_rows.size == 0:
                raise ValueError("no validation cells have finite coordinates")
            ssim_result = within_validation_ssim(
                slide["target"][ssim_rows],
                slide["prediction"][ssim_rows],
                coordinates_xy,
                slide["observed"][ssim_rows],
            )
            ssim_scores_by_slide.append(ssim_result["scores"])
            ssim_protocol_by_slide[str(slide_id)] = {
                "status": ssim_result["status"],
                "n_unique_cells": int(slide["prediction"].shape[0]),
                "valid_gene_count": int(ssim_result["valid_gene_count"]),
                "coordinate_coverage": coordinate_coverage,
                "protocol": ssim_result["protocol"],
            }
        except (KeyError, TypeError, ValueError) as exc:
            ssim_scores_by_slide.append(
                np.full(n_genes, np.nan, dtype=np.float64)
            )
            ssim_protocol_by_slide[str(slide_id)] = {
                "status": "undefined_coordinate_alignment_or_grid",
                "n_unique_cells": int(slide["prediction"].shape[0]),
                "coordinate_coverage": coordinate_coverage,
                "reason": str(exc),
            }

    ssim_matrix = np.stack(ssim_scores_by_slide, axis=0)
    ssim_finite = np.isfinite(ssim_matrix)
    ssim_counts = ssim_finite.sum(axis=0)
    ssim_gene_mean = np.divide(
        np.where(ssim_finite, ssim_matrix, 0.0).sum(axis=0),
        ssim_counts,
        out=np.full(n_genes, np.nan, dtype=np.float64),
        where=ssim_counts > 0,
    )
    ssim_valid = np.isfinite(ssim_gene_mean)
    metrics.update(
        {
            "within_val_noncanonical_cohort_label": COHORT_LABEL,
            "within_val_noncanonical_is_canonical_figure_metric": False,
            "within_val_noncanonical_ssim_mean": (
                float(np.mean(ssim_matrix[ssim_finite]))
                if ssim_finite.any()
                else np.nan
            ),
            "within_val_noncanonical_ssim_median": (
                float(np.median(ssim_matrix[ssim_finite]))
                if ssim_finite.any()
                else np.nan
            ),
            "within_val_noncanonical_ssim_valid_gene_count": int(ssim_valid.sum()),
            "within_val_noncanonical_ssim_valid_gene_coverage": (
                float(ssim_valid.mean()) if ssim_valid.size else 0.0
            ),
            "within_val_noncanonical_ssim_status": (
                "defined"
                if ssim_finite.any()
                else "undefined_no_valid_slide_gene_maps"
            ),
            "within_val_noncanonical_ssim_protocol_by_slide": ssim_protocol_by_slide,
        }
    )

    if fixed_svg_cohort_by_slide:
        metrics["fixed_gt_svg_validation"] = fixed_gt_svg_validation_metrics(
            diagnostic_slides,
            fixed_svg_cohort_by_slide,
            svg_topk=svg_topk,
        )
    for gene_idx, gene_name in enumerate(metrics["per_gene"]):
        score = ssim_gene_mean[gene_idx]
        metrics["per_gene"][gene_name]["within_val_noncanonical_ssim"] = (
            None if not np.isfinite(score) else float(score)
        )

    cmd_result = within_validation_cmd(
        hard_target,
        hard_prediction,
        hard_observed,
        gene_names=names,
    )
    metrics.update(
        {
            "within_val_noncanonical_cmd": cmd_result["cmd"],
            "within_val_noncanonical_cmd_status": cmd_result["status"],
            "within_val_noncanonical_cmd_fixed_gt_valid_gene_count": cmd_result[
                "fixed_gt_valid_gene_count"
            ],
            "within_val_noncanonical_cmd_prediction_valid_gene_count": cmd_result[
                "prediction_valid_fixed_gt_gene_count"
            ],
            "within_val_noncanonical_cmd_prediction_valid_coverage": cmd_result[
                "prediction_valid_fixed_gt_coverage"
            ],
            "within_val_noncanonical_cmd_details": cmd_result,
            "within_val_noncanonical_calibration": calibration_summary(
                hard_target, hard_prediction, hard_observed
            ),
        }
    )

    if ct_true_all:
        truth = np.concatenate(ct_true_all)
        pred_ct = np.concatenate(ct_pred_all)
        acc = []
        for idx in range(n_classes):
            keep = truth == idx
            if keep.any():
                acc.append(np.mean(pred_ct[keep] == truth[keep]))
        metrics["ct_accuracy_macro"] = float(np.mean(acc)) if acc else 0.0
    else:
        metrics["ct_accuracy_macro"] = 0.0

    # In training per_gene_dir is None, so figures are generated only by the
    # inference/export path that provides an artifact directory.
    if per_gene_dir:
        os.makedirs(per_gene_dir, exist_ok=True)
        prefix = f"epoch{epoch}" if epoch is not None else "inference"
        records = [
            dict(gene=gene, **values)
            for gene, values in metrics["per_gene"].items()
        ]
        pd.DataFrame(records).to_csv(
            os.path.join(per_gene_dir, f"{prefix}_hurdle_per_gene.csv"),
            index=False,
        )
        cmd_row = {
            key: (";".join(value) if isinstance(value, list) else value)
            for key, value in cmd_result.items()
            if key != "fixed_gt_valid_genes"
        }
        pd.DataFrame([cmd_row]).to_csv(
            os.path.join(
                per_gene_dir,
                f"{prefix}_within_val_noncanonical_cmd.csv",
            ),
            index=False,
        )
        ssim_rows = []
        for gene_idx, gene_name in enumerate(names):
            row = {
                "gene": gene_name,
                "within_val_noncanonical_ssim_mean_across_slides": ssim_gene_mean[
                    gene_idx
                ],
            }
            for slide_idx, slide in enumerate(diagnostic_slides):
                row[f"slide_{int(slide['slide_id'])}_ssim"] = ssim_matrix[
                    slide_idx, gene_idx
                ]
            ssim_rows.append(row)
        pd.DataFrame(ssim_rows).to_csv(
            os.path.join(
                per_gene_dir,
                f"{prefix}_within_val_noncanonical_ssim.csv",
            ),
            index=False,
        )
        calibration_png = os.path.join(
            per_gene_dir,
            f"{prefix}_within_val_noncanonical_calibration_density.png",
        )
        marker_png = os.path.join(
            per_gene_dir,
            f"{prefix}_within_val_noncanonical_EPCAM_KRT7_distribution.png",
        )
        try:
            write_calibration_density_png(
                hard_target,
                hard_prediction,
                hard_observed,
                calibration_png,
            )
            metrics[
                "within_val_noncanonical_calibration_density_png"
            ] = calibration_png
        except (ImportError, RuntimeError, ValueError) as exc:
            logging.warning("Within-validation calibration PNG skipped: %s", exc)
        try:
            marker_summary = write_marker_distribution_png(
                hard_target,
                hard_prediction,
                hard_observed,
                names,
                marker_png,
            )
            metrics[
                "within_val_noncanonical_marker_distribution_png"
            ] = marker_png
            metrics[
                "within_val_noncanonical_marker_distribution_summary"
            ] = marker_summary
        except (ImportError, RuntimeError, ValueError) as exc:
            logging.warning(
                "Within-validation EPCAM/KRT7 PNG skipped: %s", exc
            )

    if panel_completion_enabled:
        metrics.update(panel_metrics)

    logging.debug(
        "Hurdle validation: unique=%d zero_gap=%.4f mean_gene_W1=%.4f "
        "gene_PCC=%.4f coverage=%.3f effective/requested=%.3f "
        "within-val(noncanonical) SSIM=%s CMD=%s(%s)",
        metrics["hurdle_n_unique_cells"],
        metrics["hurdle_zero_fraction_gap"],
        metrics["hurdle_mean_per_gene_w1"],
        metrics["pearson_gene_pooled_mean"],
        metrics["pearson_gene_valid_coverage"],
        metrics["hurdle_effective_positive_fraction_of_requested"],
        metrics["within_val_noncanonical_ssim_mean"],
        metrics["within_val_noncanonical_cmd"],
        metrics["within_val_noncanonical_cmd_status"],
    )
    return metrics

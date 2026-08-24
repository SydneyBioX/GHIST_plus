"""Evaluation-only diagnostics for the unique-cell hurdle prediction matrix.

These metrics are deliberately labelled *within-validation, non-canonical*.
They use the same unique-cell hard prediction matrix as the distribution and
gene-PCC diagnostics.  They are not the immutable Figure 2 cohorts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


COHORT_LABEL = "within_validation_unique_cell_noncanonical"
SSIM_FORMULA_LABEL = "Eq21/C2 local SSIM (within-validation; non-canonical cohort)"
CMD_FORMULA_LABEL = "Eq17 correlation-matrix cosine distance (within-validation; non-canonical cohort)"


def align_coordinates_xy(cell_ids, coordinate_map_yx) -> np.ndarray:
    """Align a ``{cell_id: (y, x)}`` map and return an ``(x, y)`` matrix.

    No cell is silently dropped: missing or nonfinite coordinates are an error.
    This makes the coordinate row order auditable against the hard prediction
    matrix row order.
    """

    ids = np.asarray(cell_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("cell_ids must be one-dimensional")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must be unique before coordinate alignment")
    if coordinate_map_yx is None:
        raise ValueError("coordinate map is unavailable")

    coordinates = np.full((ids.size, 2), np.nan, dtype=np.float64)
    missing = []
    invalid = []
    for row, cell_id in enumerate(ids.tolist()):
        value = coordinate_map_yx.get(int(cell_id))
        if value is None or len(value) < 2:
            missing.append(int(cell_id))
            continue
        y, x = float(value[0]), float(value[1])
        if not np.isfinite(x) or not np.isfinite(y):
            invalid.append(int(cell_id))
            continue
        coordinates[row] = (x, y)
    if missing or invalid:
        raise ValueError(
            "coordinate alignment failed without row dropping: "
            f"missing={len(missing)} invalid={len(invalid)} "
            f"first_missing={missing[:5]} first_invalid={invalid[:5]}"
        )
    return coordinates


def select_ssim_coordinate_rows(cell_ids, coordinate_map_yx):
    """Select coordinate-covered rows for SSIM and report every exclusion."""

    ids = np.asarray(cell_ids, dtype=np.int64)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must be a one-dimensional unique vector")
    if coordinate_map_yx is None:
        coordinate_map_yx = {}
    kept_rows, coordinates, excluded_ids = [], [], []
    for row, cell_id in enumerate(ids.tolist()):
        value = coordinate_map_yx.get(int(cell_id))
        if value is None or len(value) < 2:
            excluded_ids.append(int(cell_id))
            continue
        y, x = float(value[0]), float(value[1])
        if not np.isfinite(x) or not np.isfinite(y):
            excluded_ids.append(int(cell_id))
            continue
        kept_rows.append(row)
        coordinates.append((x, y))
    kept_rows = np.asarray(kept_rows, dtype=np.int64)
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    retained_ids = ids[kept_rows]
    retained_hash = hashlib.sha256(
        np.ascontiguousarray(retained_ids).view(np.uint8)
    ).hexdigest()
    summary = {
        "requested_cell_count": int(ids.size),
        "retained_coordinate_cell_count": int(kept_rows.size),
        "excluded_missing_or_nonfinite_coordinate_cell_count": int(len(excluded_ids)),
        "retained_coordinate_cell_coverage": (
            float(kept_rows.size / ids.size) if ids.size else 0.0
        ),
        "retained_cell_ids_sha256": retained_hash,
        "first_excluded_cell_ids": excluded_ids[:20],
        "coordinate_source_order": "map values are (y,x)",
        "metric_coordinate_order": "returned columns are (x,y)",
        "note": (
            "Coordinate exclusions apply only to SSIM; distribution, PCC, CMD, "
            "and calibration retain all unique cells."
        ),
    }
    return kept_rows, coordinates, summary


@dataclass(frozen=True)
class SsimGrid:
    height: int
    width: int
    flat_index: np.ndarray
    count: np.ndarray
    crop: tuple[slice, slice]


def build_ssim_grid(
    x: np.ndarray,
    y: np.ndarray,
    width: int = 400,
    minimum_height: int = 20,
) -> SsimGrid:
    """Build the locked Eq. 21/C2 mean-aggregation grid geometry."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be nonempty aligned vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("SSIM coordinates must all be finite")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("SSIM coordinates must span both x and y")
    height = max(minimum_height, int(round(width * (ymax - ymin) / (xmax - xmin))))
    xi = np.clip(
        (((x - xmin) / (xmax - xmin)) * (width - 1)).astype(np.int32),
        0,
        width - 1,
    )
    yi = np.clip(
        (((y - ymin) / (ymax - ymin)) * (height - 1)).astype(np.int32),
        0,
        height - 1,
    )
    flat_index = yi * width + xi
    count = np.bincount(flat_index, minlength=height * width).reshape(
        height, width
    ).astype(np.float64)
    occupied_y, occupied_x = np.where(count > 0)
    crop = (
        slice(int(occupied_y.min()), int(occupied_y.max()) + 1),
        slice(int(occupied_x.min()), int(occupied_x.max()) + 1),
    )
    return SsimGrid(height, width, flat_index, count, crop)


def ssim_spatial_map(values: np.ndarray, grid: SsimGrid, sigma: float = 1.0) -> np.ndarray:
    """Mean-bin, retain zero-filled bins, smooth, then crop as in Eq. 21/C2."""

    sums = np.bincount(
        grid.flat_index,
        weights=np.asarray(values, dtype=np.float64),
        minlength=grid.height * grid.width,
    ).reshape(grid.height, grid.width)
    mean_map = np.divide(sums, grid.count, out=np.zeros_like(sums), where=grid.count > 0)
    return gaussian_filter(
        mean_map, sigma=sigma, mode="nearest", truncate=4.0
    )[grid.crop]


def ssim_local(
    target_map: np.ndarray,
    prediction_map: np.ndarray,
    data_range: float,
    window: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Exact local Eq. 21/C2 SSIM with the locked constants and crop."""

    if not np.isfinite(target_map).all() or not np.isfinite(prediction_map).all():
        return np.nan
    if not np.isfinite(data_range) or data_range <= 0:
        return np.nan
    if min(target_map.shape) < window or target_map.shape != prediction_map.shape:
        return np.nan
    arguments = {"size": window, "mode": "reflect", "origin": 0}
    mean_target = uniform_filter(target_map, **arguments)
    mean_prediction = uniform_filter(prediction_map, **arguments)
    mean_target_sq = uniform_filter(target_map * target_map, **arguments)
    mean_prediction_sq = uniform_filter(prediction_map * prediction_map, **arguments)
    mean_cross = uniform_filter(target_map * prediction_map, **arguments)
    sample_factor = window**2 / (window**2 - 1.0)
    variance_target = sample_factor * (mean_target_sq - mean_target**2)
    variance_prediction = sample_factor * (mean_prediction_sq - mean_prediction**2)
    covariance = sample_factor * (mean_cross - mean_target * mean_prediction)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mean_target * mean_prediction + c1) * (2 * covariance + c2)
    denominator = (mean_target**2 + mean_prediction**2 + c1) * (
        variance_target + variance_prediction + c2
    )
    score_map = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    border = (window - 1) // 2
    interior = score_map[border:-border, border:-border]
    return float(np.mean(interior)) if interior.size and np.isfinite(interior).all() else np.nan


def within_validation_ssim(
    target: np.ndarray,
    prediction: np.ndarray,
    coordinates_xy: np.ndarray,
    observed: np.ndarray | None = None,
) -> dict:
    """Compute non-canonical within-validation SSIM for every target-valid gene."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    coordinates_xy = np.asarray(coordinates_xy, dtype=np.float64)
    if target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("target and prediction must share [n_cells,n_genes] shape")
    if coordinates_xy.shape != (target.shape[0], 2):
        raise ValueError("coordinates_xy must align one-to-one with matrix rows")
    observed = (
        np.ones_like(target, dtype=bool)
        if observed is None
        else np.asarray(observed, dtype=bool)
    )
    if observed.shape != target.shape:
        raise ValueError("observed mask must match target")

    grid = build_ssim_grid(coordinates_xy[:, 0], coordinates_xy[:, 1], width=400)
    scores = np.full(target.shape[1], np.nan, dtype=np.float64)
    ranges = np.full(target.shape[1], np.nan, dtype=np.float64)
    target_complete = observed.all(axis=0) & np.isfinite(target).all(axis=0)
    prediction_complete = np.isfinite(prediction).all(axis=0)
    for gene_idx in range(target.shape[1]):
        if not target_complete[gene_idx] or not prediction_complete[gene_idx]:
            continue
        target_map = ssim_spatial_map(target[:, gene_idx], grid, sigma=1.0)
        ranges[gene_idx] = float(np.ptp(target_map))
        prediction_map = ssim_spatial_map(prediction[:, gene_idx], grid, sigma=1.0)
        scores[gene_idx] = ssim_local(
            target_map,
            prediction_map,
            ranges[gene_idx],
            window=7,
            k1=0.01,
            k2=0.03,
        )

    valid = np.isfinite(scores)
    occupied = grid.count[grid.crop] > 0
    return {
        "cohort_label": COHORT_LABEL,
        "canonical_figure_metric": False,
        "formula": SSIM_FORMULA_LABEL,
        "scores": scores,
        "gt_smoothed_map_ranges": ranges,
        "requested_gene_count": int(target.shape[1]),
        "valid_gene_count": int(valid.sum()),
        "valid_gene_coverage": float(valid.mean()) if valid.size else 0.0,
        "mean": float(np.mean(scores[valid])) if valid.any() else np.nan,
        "median": float(np.median(scores[valid])) if valid.any() else np.nan,
        "status": "defined" if valid.any() else "undefined_no_valid_gene_maps",
        "protocol": {
            "grid_width": int(grid.width),
            "grid_height": int(grid.height),
            "aggregation": "per-bin mean; empty bins zero-filled and retained",
            "gaussian": {"sigma": 1.0, "mode": "nearest", "truncate": 4.0},
            "crop": [
                grid.crop[0].start,
                grid.crop[0].stop,
                grid.crop[1].start,
                grid.crop[1].stop,
            ],
            "occupied_bins": int(occupied.sum()),
            "zero_filled_bins_retained": int(occupied.size - occupied.sum()),
            "local_window": "7 x 7 uniform",
            "local_boundary": "reflect",
            "sample_covariance_factor": "49/48",
            "border_excluded": 3,
            "K1": 0.01,
            "K2": 0.03,
            "data_range": "ptp(smoothed log1p GT map), fixed for prediction",
        },
    }


def cmd_eq17(target: np.ndarray, prediction: np.ndarray, columns: np.ndarray) -> float:
    """Eq. 17 correlation-matrix cosine distance on explicit columns."""

    columns = np.asarray(columns, dtype=np.int64)
    target = np.asarray(target[:, columns], dtype=np.float64)
    prediction = np.asarray(prediction[:, columns], dtype=np.float64)
    if target.shape[1] < 2:
        return np.nan
    target_correlation = np.corrcoef(target, rowvar=False)
    prediction_correlation = np.corrcoef(prediction, rowvar=False)
    if not np.isfinite(target_correlation).all() or not np.isfinite(prediction_correlation).all():
        return np.nan
    numerator = float(np.trace(prediction_correlation @ target_correlation))
    denominator = float(
        np.linalg.norm(prediction_correlation, "fro")
        * np.linalg.norm(target_correlation, "fro")
        + 1e-12
    )
    return 1.0 - numerator / denominator


def within_validation_cmd(
    target: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray | None = None,
    gene_names=None,
    fixed_gt_valid_mask: np.ndarray | None = None,
) -> dict:
    """Compute Eq. 17 on a GT-only fixed mask without prediction mask shrinkage."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("target and prediction must share [n_cells,n_genes] shape")
    observed = (
        np.ones_like(target, dtype=bool)
        if observed is None
        else np.asarray(observed, dtype=bool)
    )
    if observed.shape != target.shape:
        raise ValueError("observed mask must match target")
    names = np.asarray(
        list(gene_names) if gene_names is not None else [str(i) for i in range(target.shape[1])],
        dtype=str,
    )
    if names.size != target.shape[1]:
        raise ValueError("gene_names length does not match matrix columns")

    gt_valid_now = (
        observed.all(axis=0)
        & np.isfinite(target).all(axis=0)
        & (np.std(target, axis=0, dtype=np.float64) > 0)
    )
    if fixed_gt_valid_mask is None:
        retained = gt_valid_now
    else:
        retained = np.asarray(fixed_gt_valid_mask, dtype=bool)
        if retained.shape != (target.shape[1],):
            raise ValueError("fixed_gt_valid_mask must have one entry per gene")
        if np.any(retained & ~gt_valid_now):
            raise ValueError("fixed_gt_valid_mask contains a gene invalid in GT")
    prediction_valid = np.isfinite(prediction).all(axis=0) & (
        np.std(prediction, axis=0, dtype=np.float64) > 0
    )
    invalid_prediction = retained & ~prediction_valid
    columns = np.flatnonzero(retained)
    if columns.size < 2:
        score = np.nan
        status = "undefined_fewer_than_two_gt_valid_genes"
    elif invalid_prediction.any():
        score = np.nan
        status = "undefined_prediction_constant_or_nonfinite_on_fixed_gt_mask"
    else:
        score = cmd_eq17(target, prediction, columns)
        status = "defined_on_fixed_gt_valid_mask" if np.isfinite(score) else "undefined_nonfinite_correlation"

    mask_hash = hashlib.sha256(np.ascontiguousarray(retained).view(np.uint8)).hexdigest()
    retained_count = int(retained.sum())
    prediction_valid_count = int((retained & prediction_valid).sum())
    return {
        "cohort_label": COHORT_LABEL,
        "canonical_figure_metric": False,
        "formula": CMD_FORMULA_LABEL,
        "cmd": float(score),
        "status": status,
        "requested_gene_count": int(target.shape[1]),
        "fixed_gt_valid_gene_count": retained_count,
        "prediction_valid_fixed_gt_gene_count": prediction_valid_count,
        "prediction_valid_fixed_gt_coverage": (
            float(prediction_valid_count / retained_count) if retained_count else 0.0
        ),
        "fixed_gt_valid_mask_sha256": mask_hash,
        "fixed_gt_valid_genes": names[retained].tolist(),
        "invalid_prediction_genes_on_fixed_gt_mask": names[invalid_prediction].tolist(),
        "gt_invalid_gene_count": int((~gt_valid_now).sum()),
    }


def calibration_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray | None = None,
) -> dict:
    """Pooled hard-matrix calibration used only for the diagnostic density plot."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    observed = (
        np.ones_like(target, dtype=bool)
        if observed is None
        else np.asarray(observed, dtype=bool)
    )
    keep = observed & np.isfinite(target) & np.isfinite(prediction)
    x = target[keep]
    y = prediction[keep]
    if x.size < 2:
        return {"n_finite_pairs": int(x.size), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    x_mean, y_mean = float(x.mean()), float(y.mean())
    xc, yc = x - x_mean, y - y_mean
    ss_x, ss_y = float(np.dot(xc, xc)), float(np.dot(yc, yc))
    covariance = float(np.dot(xc, yc))
    slope = covariance / ss_x if ss_x > 0 else np.nan
    intercept = y_mean - slope * x_mean if np.isfinite(slope) else np.nan
    r2 = covariance**2 / (ss_x * ss_y) if ss_x > 0 and ss_y > 0 else np.nan
    return {
        "cohort_label": COHORT_LABEL,
        "n_finite_pairs": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(np.clip(r2, 0.0, 1.0)) if np.isfinite(r2) else np.nan,
    }


def _plot_values(target, prediction, observed):
    keep = np.asarray(observed, dtype=bool) & np.isfinite(target) & np.isfinite(prediction)
    return np.asarray(target, dtype=np.float64)[keep], np.asarray(prediction, dtype=np.float64)[keep]


def write_calibration_density_png(
    target: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    output_path,
    max_points: int = 500_000,
) -> dict:
    """Write compact pooled GT-vs-pred density using the hard validation matrix."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    x, y = _plot_values(target, prediction, observed)
    if not x.size:
        raise ValueError("no finite observed pairs available for calibration plot")
    sample_size = min(int(max_points), int(x.size))
    indices = np.linspace(0, x.size - 1, sample_size, dtype=np.int64)
    xs, ys = x[indices], y[indices]
    x_hi = max(float(np.quantile(xs, 0.995)), 1e-6)
    y_hi = max(float(np.quantile(ys, 0.995)), 1e-6)
    histogram, x_edges, y_edges = np.histogram2d(
        xs,
        ys,
        bins=140,
        range=((0.0, x_hi), (0.0, y_hi)),
    )
    summary = calibration_summary(target, prediction, observed)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    positive = histogram[histogram > 0]
    norm = mcolors.LogNorm(vmin=1, vmax=max(1.0, float(positive.max()) if positive.size else 1.0))
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        histogram.T,
        shading="auto",
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )
    common_hi = min(x_hi, y_hi)
    axis.plot([0, common_hi], [0, common_hi], "--", color="white", linewidth=1.0, label="Identity")
    if np.isfinite(summary["slope"]) and np.isfinite(summary["intercept"]):
        line_x = np.asarray([0.0, x_hi])
        axis.plot(line_x, summary["intercept"] + summary["slope"] * line_x, color="#E45756", linewidth=1.2, label="OLS")
    axis.set(
        xlim=(0.0, x_hi),
        ylim=(0.0, y_hi),
        xlabel="GT log1p(raw count)",
        ylabel="Hard prediction (log1p scale)",
        title="Within-validation calibration\n(0-99.5% view; non-canonical cohort)",
    )
    axis.text(
        0.03,
        0.97,
        f"slope={summary['slope']:.3f}\n$R^2$={summary['r2']:.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    axis.legend(frameon=False, loc="lower right", fontsize=8)
    figure.colorbar(image, ax=axis, pad=0.02, label="Pair count (log scale)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return summary


def write_marker_distribution_png(
    target: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    gene_names,
    output_path,
    marker_genes=("EPCAM", "KRT7"),
) -> dict:
    """Write exact-zero-retaining GT/pred histograms for requested marker genes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = np.asarray(list(gene_names), dtype=str)
    figure, axes = plt.subplots(1, len(marker_genes), figsize=(8.6, 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    summaries = {}
    for axis, marker in zip(axes, marker_genes):
        matches = np.flatnonzero(names == marker)
        if matches.size != 1:
            axis.text(0.5, 0.5, f"{marker} unavailable", ha="center", va="center")
            axis.set_axis_off()
            summaries[marker] = {"status": "gene_unavailable"}
            continue
        index = int(matches[0])
        keep = np.asarray(observed[:, index], dtype=bool)
        gt = np.asarray(target[keep, index], dtype=np.float64)
        pred = np.asarray(prediction[keep, index], dtype=np.float64)
        finite = np.isfinite(gt) & np.isfinite(pred)
        gt, pred = gt[finite], pred[finite]
        if not gt.size:
            axis.text(0.5, 0.5, f"{marker}: no finite values", ha="center", va="center")
            axis.set_axis_off()
            summaries[marker] = {"status": "no_finite_values"}
            continue
        # Retain the complete finite marker distribution, including the tail.
        high = max(float(np.max(gt)), float(np.max(pred)), 1e-6)
        bins = np.linspace(0.0, high, 36)
        weights = np.full(gt.shape, 100.0 / gt.size)
        axis.hist(gt, bins=bins, weights=weights, histtype="step", linewidth=1.6, color="#3B73B9", label="GT")
        axis.hist(pred, bins=bins, weights=weights, histtype="step", linewidth=1.6, color="#D78725", label="Prediction")
        gt_zero = float(np.mean(gt == 0))
        pred_zero = float(np.mean(pred == 0))
        summaries[marker] = {
            "status": "defined",
            "n_cells": int(gt.size),
            "gt_zero_fraction": gt_zero,
            "prediction_zero_fraction": pred_zero,
            "gt_mean": float(np.mean(gt)),
            "prediction_mean": float(np.mean(pred)),
        }
        axis.set(
            title=f"{marker}\nGT zero {gt_zero:.1%}; pred zero {pred_zero:.1%}",
            xlabel="log1p expression",
            ylabel="Cells (%)",
            xlim=(0.0, high),
        )
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Within-validation marker distributions (non-canonical cohort)", fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return summaries

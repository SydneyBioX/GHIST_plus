#!/usr/bin/env python3
"""One-shot ROI residual correction for full-slide GHIST+ predictions.

The command reads one JSON config, fits shrunk ridge residual correction (RRC)
models from nested ROI boxes, writes corrected predictions, and exports the
long per-gene Pearson table used by Fig. 4b-style plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


RUN_ORDER = ["ZeroShot"] + [f"ROI_{s:.1f}mm" for s in np.arange(0.5, 3.5 + 1e-9, 0.5)]
SUBSET_COLUMNS = ["Top 20 SVG", "Top 50 SVG", "Top 20 non-SVG", "Top 50 non-SVG"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def read_matrix_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    print(f"[load] matrix {path}", flush=True)
    with path.open(newline="") as f:
        header = next(csv.reader(f))
    genes = header[1:]
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    ids = data[:, 0].astype(np.int64)
    values = np.ascontiguousarray(data[:, 1:])
    del data
    print(f"[load] rows={len(ids)} genes={len(genes)}", flush=True)
    return ids, genes, values


def read_coords(path: Path) -> pd.DataFrame:
    print(f"[load] coords {path}", flush=True)
    df = pd.read_csv(path)
    colmap = {c.lower(): c for c in df.columns}

    def pick(candidates: list[str], desc: str) -> str:
        for name in candidates:
            if name.lower() in colmap:
                return colmap[name.lower()]
        raise ValueError(f"Could not find {desc} in {path}; columns={df.columns.tolist()}")

    id_col = pick(["cell_id", "c_id", "cellID", "id"], "cell id column")
    x_col = pick(["x_coord", "x", "x_um", "x0"], "x coordinate column")
    y_col = pick(["y_coord", "y", "y_um", "y0"], "y coordinate column")
    out = df.rename(columns={id_col: "cell_id", x_col: "x_coord", y_col: "y_coord"})
    out = out[["cell_id", "x_coord", "y_coord"]].dropna()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    out = out.set_index("cell_id")
    print(f"[load] coords rows={len(out)}", flush=True)
    return out


def read_first_col_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                ids.add(int(float(row[0])))
            except ValueError:
                continue
    return ids


def read_image_shape(path: Path) -> tuple[int, int]:
    try:
        import tifffile as tf

        shape = tf.memmap(path).shape
        return int(shape[0]), int(shape[1])
    except Exception:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            width, height = im.size
        return int(height), int(width)


def normalize_sources(src: Any) -> list[dict[str, Any]]:
    if isinstance(src, list):
        return src
    if isinstance(src, dict):
        return [src]
    return []


def pick_source(model_config: dict[str, Any], slide_idx: int) -> dict[str, Any]:
    pools = [
        normalize_sources(model_config.get("data_sources_test")),
        normalize_sources(model_config.get("data_sources_train_val")),
    ]
    for pool in pools:
        for src in pool:
            try:
                if int(src.get("slide_idx", src.get("slide_id", -1))) == int(slide_idx):
                    return src
            except Exception:
                continue
    for pool in pools:
        if pool:
            return pool[0]
    raise ValueError("No data_sources_test or data_sources_train_val entries found")


def align_inputs(
    pred_ids: np.ndarray,
    pred_genes: list[str],
    pred: np.ndarray,
    gt_ids: np.ndarray,
    gt_genes: list[str],
    gt: np.ndarray,
    coords: pd.DataFrame,
    feature_scope: str,
) -> dict[str, Any]:
    print("[align] build cell/gene indexes", flush=True)
    gt_index = {int(cid): i for i, cid in enumerate(gt_ids)}
    coord_index = set(int(cid) for cid in coords.index)

    print("[align] match cells", flush=True)
    pred_keep: list[int] = []
    gt_keep: list[int] = []
    keep_ids: list[int] = []
    for i, cid in enumerate(pred_ids):
        cid = int(cid)
        gi = gt_index.get(cid)
        if gi is None or cid not in coord_index:
            continue
        pred_keep.append(i)
        gt_keep.append(gi)
        keep_ids.append(cid)

    if not keep_ids:
        raise ValueError("No common cells across prediction, GT, and coordinates")

    print(f"[align] matched_cells={len(keep_ids)}", flush=True)
    pred_gene_index = {g: i for i, g in enumerate(pred_genes)}
    gt_gene_index = {g: i for i, g in enumerate(gt_genes)}
    out_genes = [g for g in gt_genes if g in pred_gene_index]
    if not out_genes:
        raise ValueError("No common genes across prediction and GT")

    pred_out_idx = np.asarray([pred_gene_index[g] for g in out_genes], dtype=np.int64)
    gt_out_idx = np.asarray([gt_gene_index[g] for g in out_genes], dtype=np.int64)
    pred_keep_arr = np.asarray(pred_keep, dtype=np.int64)
    gt_keep_arr = np.asarray(gt_keep, dtype=np.int64)
    keep_ids_arr = np.asarray(keep_ids, dtype=np.int64)

    print("[align] slice matrices", flush=True)
    if pred_keep_arr.size == pred.shape[0] and np.array_equal(pred_keep_arr, np.arange(pred.shape[0])):
        pred_aligned = pred
    else:
        pred_aligned = np.take(pred, pred_keep_arr, axis=0)
    pred_out = np.ascontiguousarray(np.take(pred_aligned, pred_out_idx, axis=1))
    gt_aligned = np.take(gt, gt_keep_arr, axis=0)
    gt_out = np.ascontiguousarray(np.take(gt_aligned, gt_out_idx, axis=1))

    if feature_scope == "output_genes":
        features = pred_out
        feature_genes = out_genes
    elif feature_scope == "all_prediction_genes":
        features = np.ascontiguousarray(pred_aligned)
        feature_genes = pred_genes
    else:
        raise ValueError(f"Unsupported feature_scope={feature_scope!r}")

    print("[align] slice coords", flush=True)
    coords_aligned = coords.loc[keep_ids_arr]
    return {
        "ids": keep_ids_arr,
        "out_genes": out_genes,
        "feature_genes": feature_genes,
        "features": features.astype(np.float32, copy=False),
        "pred_out": pred_out.astype(np.float32, copy=False),
        "gt_out": gt_out.astype(np.float32, copy=False),
        "x": coords_aligned["x_coord"].to_numpy(dtype=np.float64),
        "y": coords_aligned["y_coord"].to_numpy(dtype=np.float64),
    }


def make_roi_boxes(coords: pd.DataFrame, roi_ids: set[int], roi_sizes_mm: list[float]) -> list[dict[str, float]]:
    roi_coords = coords.loc[coords.index.intersection(list(roi_ids))]
    if roi_coords.empty:
        raise ValueError("ROI seed ids do not overlap coordinate file")

    xmin, ymin = roi_coords[["x_coord", "y_coord"]].min().to_numpy(dtype=np.float64)
    xmax, ymax = roi_coords[["x_coord", "y_coord"]].max().to_numpy(dtype=np.float64)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half_w0 = (xmax - xmin) / 2.0
    half_h0 = (ymax - ymin) / 2.0
    base_mm = min(roi_sizes_mm)

    boxes = []
    for size in roi_sizes_mm:
        scale = float(size) / float(base_mm)
        half_w = half_w0 * scale
        half_h = half_h0 * scale
        boxes.append(
            {
                "roi_mm": float(size),
                "label": f"ROI_{float(size):.1f}mm",
                "dir_label": f"roi{str(float(size)).replace('.', 'p')}mm_rrc",
                "x0": cx - half_w,
                "x1": cx + half_w,
                "y0": cy - half_h,
                "y1": cy + half_h,
                "area": (2.0 * half_w) * (2.0 * half_h),
            }
        )
    return boxes


def pcc_per_gene(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt64 = gt.astype(np.float64, copy=False)
    pred64 = pred.astype(np.float64, copy=False)
    gt_center = gt64 - gt64.mean(axis=0, keepdims=True)
    pred_center = pred64 - pred64.mean(axis=0, keepdims=True)
    denom = np.sqrt((gt_center * gt_center).sum(axis=0) * (pred_center * pred_center).sum(axis=0))
    out = np.zeros(gt64.shape[1], dtype=np.float64)
    np.divide((gt_center * pred_center).sum(axis=0), denom, out=out, where=denom > 0)
    return out


def mean_pcc(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(pcc_per_gene(gt, pred).mean())


def fit_rrc(x_train: np.ndarray, residual_train: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = residual_train.mean(axis=0, keepdims=True)
    xc = (x_train - x_mean).astype(np.float64, copy=False)
    yc = (residual_train - y_mean).astype(np.float64, copy=False)
    xtx = xc.T @ xc
    xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
    xty = xc.T @ yc
    evals, evecs = np.linalg.eigh(xtx)
    coef = evecs @ ((evecs.T @ xty) / np.maximum(evals[:, None], 1e-12))
    return x_mean.astype(np.float32), y_mean.astype(np.float32), coef.astype(np.float32)


def predict_residual(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray], chunk_size: int) -> np.ndarray:
    x_mean, y_mean, coef = model
    out = np.empty((x.shape[0], coef.shape[1]), dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = (x[start:end] - x_mean) @ coef + y_mean
    return out


def make_folds(n: int, n_folds: int, seed: int, val_fraction: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n_folds <= 1:
        n_val = max(10, int(round(val_fraction * n)))
        n_val = min(n_val, max(1, n - 2))
        return [perm[:n_val]]
    return [fold for fold in np.array_split(perm, min(n_folds, n)) if len(fold) > 0]


def choose_hyperparams(
    x_roi: np.ndarray,
    pred_roi: np.ndarray,
    gt_roi: np.ndarray,
    alphas: list[float],
    blends: list[float],
    n_folds: int,
    seed: int,
    val_fraction: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    residual_roi = gt_roi - pred_roi
    folds = make_folds(x_roi.shape[0], n_folds, seed, val_fraction)
    scores = {(float(alpha), float(blend)): [] for alpha in alphas for blend in blends}
    all_idx = np.arange(x_roi.shape[0])

    for val_idx in folds:
        train_mask = np.ones(x_roi.shape[0], dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        if len(train_idx) < 2 or len(val_idx) < 2:
            continue

        x_train = x_roi[train_idx]
        residual_train = residual_roi[train_idx]
        x_val = x_roi[val_idx]
        pred_val = pred_roi[val_idx]
        gt_val = gt_roi[val_idx]

        x_mean = x_train.mean(axis=0, keepdims=True)
        y_mean = residual_train.mean(axis=0, keepdims=True)
        xc = (x_train - x_mean).astype(np.float64, copy=False)
        yc = (residual_train - y_mean).astype(np.float64, copy=False)
        xtx = xc.T @ xc
        xty = xc.T @ yc
        evals, evecs = np.linalg.eigh(xtx)
        qtb = evecs.T @ xty
        xvq = (x_val - x_mean).astype(np.float64, copy=False) @ evecs

        for alpha in alphas:
            pred_residual = xvq @ (qtb / (evals[:, None] + float(alpha))) + y_mean
            pred_residual = pred_residual.astype(np.float32, copy=False)
            for blend in blends:
                pred_corrected = pred_val + float(blend) * pred_residual
                pred_corrected[pred_corrected < 0] = 0.0
                scores[(float(alpha), float(blend))].append(mean_pcc(gt_val, pred_corrected))

    best: dict[str, float] | None = None
    results: list[dict[str, float]] = []
    for (alpha, blend), vals in scores.items():
        if not vals:
            continue
        score = float(np.mean(vals))
        row = {"alpha": alpha, "blend": blend, "cv_mean_pcc": score}
        results.append(row)
        if best is None or score > best["cv_mean_pcc"]:
            best = row

    if best is None:
        raise ValueError("Could not select RRC hyperparameters")
    return best, results


def write_matrix(path: Path, ids: np.ndarray, genes: list[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", *genes])
        for cid, row in zip(ids, mat):
            writer.writerow([int(cid), *[f"{float(v):.7g}" for v in row]])


def giotto_rank_scores(expr: np.ndarray, coords_xy: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(coords_xy)
    neigh_idx = tree.query(coords_xy, k=k + 1)[1][:, 1:]
    scores = np.zeros(expr.shape[1], dtype=np.float32)
    for gene_idx in range(expr.shape[1]):
        ranks = rankdata(expr[:, gene_idx], method="average").astype(np.float32)
        neigh_mean = ranks[neigh_idx].mean(axis=1)
        r_center = ranks - ranks.mean()
        n_center = neigh_mean - neigh_mean.mean()
        denom = np.sqrt((r_center * r_center).sum() * (n_center * n_center).sum())
        if denom > 0:
            scores[gene_idx] = float(np.dot(r_center, n_center) / denom)
    return scores


def build_fig4b_table(
    run_predictions: dict[str, np.ndarray],
    gt_val: np.ndarray,
    genes: list[str],
    giotto_scores: np.ndarray,
    run_info: dict[str, dict[str, float]],
    sample_label: str,
) -> pd.DataFrame:
    svg_order = np.argsort(giotto_scores)[::-1]
    nonsvg_order = np.argsort(giotto_scores)
    subset_masks = {
        "Top 20 SVG": np.zeros(len(genes), dtype=bool),
        "Top 50 SVG": np.zeros(len(genes), dtype=bool),
        "Top 20 non-SVG": np.zeros(len(genes), dtype=bool),
        "Top 50 non-SVG": np.zeros(len(genes), dtype=bool),
    }
    subset_masks["Top 20 SVG"][svg_order[:20]] = True
    subset_masks["Top 50 SVG"][svg_order[:50]] = True
    subset_masks["Top 20 non-SVG"][nonsvg_order[:20]] = True
    subset_masks["Top 50 non-SVG"][nonsvg_order[:50]] = True

    rows = []
    for run in [r for r in RUN_ORDER if r in run_predictions]:
        pcc = pcc_per_gene(gt_val, run_predictions[run])
        info = run_info[run]
        for gene_idx, gene in enumerate(genes):
            row = {
                "Sample": sample_label,
                "Run": run,
                "ROI size (mm)": info["roi_mm"],
                "ROI area (% of slide)": info["roi_area_pct"],
                "Gene": gene,
                "Pearson r on validation cells": pcc[gene_idx],
            }
            for col in SUBSET_COLUMNS:
                row[col] = bool(subset_masks[col][gene_idx])
            rows.append(row)

    columns = [
        "Sample",
        "Run",
        "ROI size (mm)",
        "ROI area (% of slide)",
        "Gene",
        "Top 20 SVG",
        "Top 50 SVG",
        "Top 20 non-SVG",
        "Top 50 non-SVG",
        "Pearson r on validation cells",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_one_shot_rrc(cfg: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(cfg["out_root"]).expanduser().resolve()
    outputs_root = out_root / "outputs"
    full_root = outputs_root / "full"
    val_root = outputs_root / "val_top20pct"
    tables_root = out_root / "tables"
    outputs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    write_full_predictions = bool(cfg.get("write_full_predictions", False))

    model_config = json.load(Path(cfg["model_config"]).open())
    source = pick_source(model_config, int(cfg["slide_idx"]))
    gt_fp = Path(cfg.get("gt_fp", source["fp_expr"]))
    hist_fp = Path(cfg.get("hist_fp", source["fp_hist"]))

    pred_ids, pred_genes, pred = read_matrix_csv(Path(cfg["base_prediction"]))
    gt_ids, gt_genes, gt = read_matrix_csv(gt_fp)
    coords = read_coords(Path(cfg["coords_fp"]))

    print("[align] inputs", flush=True)
    aligned = align_inputs(
        pred_ids,
        pred_genes,
        pred,
        gt_ids,
        gt_genes,
        gt,
        coords,
        cfg.get("feature_scope", "output_genes"),
    )

    ids = aligned["ids"]
    genes = aligned["out_genes"]
    features = aligned["features"]
    pred_out = aligned["pred_out"]
    gt_out = aligned["gt_out"]
    xs = aligned["x"]
    ys = aligned["y"]
    print(f"[align] cells={len(ids)} output_genes={len(genes)} feature_genes={len(aligned['feature_genes'])}", flush=True)

    print("[standardize] features", flush=True)
    feat_mean = features.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    feat_var = ((features - feat_mean) ** 2).mean(axis=0, dtype=np.float64, keepdims=True)
    feat_std = np.sqrt(feat_var).astype(np.float32)
    feat_std[feat_std == 0] = 1.0
    features_std = ((features - feat_mean) / feat_std).astype(np.float32, copy=False)

    print(f"[load] image shape {hist_fp}", flush=True)
    height, width = read_image_shape(hist_fp)
    val_top_frac = float(cfg.get("val_top_frac", 0.2))
    val_cutoff = val_top_frac * height
    val_mask = ys < val_cutoff
    if not np.any(val_mask):
        raise ValueError("Validation mask is empty")

    roi_sizes_mm = [float(x) for x in cfg.get("roi_sizes_mm", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])]
    roi_ids = read_first_col_ids(Path(cfg["roi_small_fp"]))
    boxes = make_roi_boxes(coords, roi_ids, roi_sizes_mm)
    alphas = [float(x) for x in cfg.get("alphas", [1, 3, 10, 30, 100, 300, 1000, 3000, 10000])]
    blends = [float(x) for x in cfg.get("blends", [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])]
    blend_cap_n0 = float(cfg.get("blend_cap_n0", 1000.0))
    chunk_size = int(cfg.get("chunk_size", 10000))
    seed = int(cfg.get("seed", 7))

    run_predictions = {"ZeroShot": pred_out[val_mask]}
    run_info = {"ZeroShot": {"roi_mm": 0.0, "roi_area_pct": 0.0}}
    baseline_pcc = pcc_per_gene(gt_out[val_mask], pred_out[val_mask])

    zero_shot_target = "full and validation outputs" if write_full_predictions else "validation output"
    print(f"[write] ZeroShot {zero_shot_target}", flush=True)
    if write_full_predictions:
        write_matrix(full_root / "ZeroShot" / "gene_expression_predictions_ZeroShot.csv", ids, genes, pred_out)
    write_matrix(val_root / "ZeroShot" / "gene_expression_predictions_ZeroShot.csv", ids[val_mask], genes, pred_out[val_mask])

    summary_rows = [
        {
            "method": "ZeroShot",
            "roi_mm": "",
            "roi_cells": "",
            "alpha": "",
            "blend": "",
            "val_cells": int(val_mask.sum()),
            "mean_pcc": float(baseline_pcc.mean()),
            "delta_vs_zeroshot": 0.0,
            "median_gene_delta": 0.0,
            "genes_improved": 0,
            "n_genes": len(genes),
        }
    ]

    for box in boxes:
        roi_mask = (xs >= box["x0"]) & (xs <= box["x1"]) & (ys >= box["y0"]) & (ys <= box["y1"])
        roi_idx = np.where(roi_mask)[0]
        if roi_idx.size < 10:
            print(f"[skip] {box['label']} too few ROI cells: {roi_idx.size}")
            continue
        if box["y0"] < val_cutoff:
            print(f"[skip] {box['label']} crosses validation cutoff: y0={box['y0']:.1f} < {val_cutoff:.1f}")
            continue

        if blend_cap_n0 > 0:
            blend_cap = float(roi_idx.size / (roi_idx.size + blend_cap_n0))
            active_blends = sorted(set([b for b in blends if b <= blend_cap] + [blend_cap]))
        else:
            blend_cap = 1.0
            active_blends = blends

        print(
            f"[fit] {box['label']} roi_cells={roi_idx.size} blend_cap={blend_cap:.3f}",
            flush=True,
        )
        x_roi = features_std[roi_idx]
        pred_roi = pred_out[roi_idx]
        gt_roi = gt_out[roi_idx]
        best, cv_results = choose_hyperparams(
            x_roi,
            pred_roi,
            gt_roi,
            alphas,
            active_blends,
            int(cfg.get("n_folds", 1)),
            seed,
            float(cfg.get("val_fraction", 0.2)),
        )

        model = fit_rrc(x_roi, gt_roi - pred_roi, best["alpha"])
        if write_full_predictions:
            pred_residual = predict_residual(features_std, model, chunk_size)
            corrected = pred_out + float(best["blend"]) * pred_residual
            corrected[corrected < 0] = 0.0
            corrected_val = corrected[val_mask]
        else:
            pred_residual_val = predict_residual(features_std[val_mask], model, chunk_size)
            corrected_val = pred_out[val_mask] + float(best["blend"]) * pred_residual_val
            corrected_val[corrected_val < 0] = 0.0
        pcc = pcc_per_gene(gt_out[val_mask], corrected_val)
        delta = pcc - baseline_pcc
        roi_area_pct = 100.0 * float(box["area"]) / float(width * height)

        full_dir = full_root / box["dir_label"]
        val_dir = val_root / box["dir_label"]
        output_target = "full and validation outputs" if write_full_predictions else "validation output"
        print(f"[write] {box['label']} {output_target}", flush=True)
        if write_full_predictions:
            write_matrix(full_dir / "gene_expression_predictions_rrc.csv", ids, genes, corrected)
        write_matrix(val_dir / "gene_expression_predictions_rrc.csv", ids[val_mask], genes, corrected_val)

        meta = {
            "method": "shrunk_ridge_residual_correction",
            "roi_mm": box["roi_mm"],
            "roi_label": box["label"],
            "roi_cells": int(roi_idx.size),
            "roi_area_pct": roi_area_pct,
            "roi_box": {k: float(v) for k, v in box.items() if k not in {"label", "dir_label"}},
            "hyperparameter_selection": {
                "scope": "ROI-only holdout" if int(cfg.get("n_folds", 1)) <= 1 else "ROI-only cross-validation",
                "blend_cap_n0": blend_cap_n0,
                "blend_cap": blend_cap,
                "candidate_blends": active_blends,
                "best": best,
                "results": cv_results,
            },
            "validation": {
                "rule": f"y_coord < {val_top_frac} * H",
                "H": int(height),
                "cutoff_y": float(val_cutoff),
                "n_cells": int(val_mask.sum()),
            },
            "top20pct_metrics": {
                "mean_pcc": float(pcc.mean()),
                "delta_vs_zeroshot": float(delta.mean()),
                "median_gene_delta": float(np.median(delta)),
                "genes_improved": int((delta > 0).sum()),
                "n_genes": len(genes),
            },
        }
        metadata_dirs = [val_dir]
        if write_full_predictions:
            metadata_dirs.append(full_dir)
        for directory in metadata_dirs:
            with (directory / "rrc_metadata.json").open("w") as f:
                json.dump(meta, f, indent=2)

        run_predictions[box["label"]] = corrected_val
        run_info[box["label"]] = {"roi_mm": box["roi_mm"], "roi_area_pct": roi_area_pct}
        summary_rows.append(
            {
                "method": "RRC",
                "roi_mm": box["roi_mm"],
                "roi_cells": int(roi_idx.size),
                "alpha": best["alpha"],
                "blend": best["blend"],
                "val_cells": int(val_mask.sum()),
                "mean_pcc": float(pcc.mean()),
                "delta_vs_zeroshot": float(delta.mean()),
                "median_gene_delta": float(np.median(delta)),
                "genes_improved": int((delta > 0).sum()),
                "n_genes": len(genes),
            }
        )
        print(
            f"[done] {box['label']} mean_pcc={pcc.mean():.6f} "
            f"delta={delta.mean():+.6f} alpha={best['alpha']} blend={best['blend']}",
            flush=True,
        )

    summary_fp = outputs_root / "summary_top20pct_metrics.csv"
    pd.DataFrame(summary_rows).to_csv(summary_fp, index=False)

    coords_xy = np.column_stack([xs, ys]).astype(np.float32)
    print("[table] ranking SVGs from full GT", flush=True)
    giotto_scores = giotto_rank_scores(gt_out, coords_xy, int(cfg.get("giotto_k", 8)))
    table = build_fig4b_table(
        run_predictions,
        gt_out[val_mask],
        genes,
        giotto_scores,
        run_info,
        cfg.get("sample_label", "Breast2"),
    )
    table_fp = tables_root / cfg.get("fig4b_table_name", "Breast2_all_runs_per_gene_pearson_long_NEW.csv")
    table.to_csv(table_fp, index=False)

    run_meta = {
        "method": "shrunk_ridge_residual_correction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": cfg["_config_path"],
        "out_root": str(out_root),
        "base_prediction": cfg["base_prediction"],
        "checkpoint": cfg.get("checkpoint"),
        "model_config": cfg["model_config"],
        "coords_fp": cfg["coords_fp"],
        "gt_fp": str(gt_fp),
        "hist_fp": str(hist_fp),
        "roi_small_fp": cfg["roi_small_fp"],
        "validation": {
            "rule": f"y_coord < {val_top_frac} * H",
            "H": int(height),
            "cutoff_y": float(val_cutoff),
            "n_cells": int(val_mask.sum()),
        },
        "n_aligned_cells": int(len(ids)),
        "n_output_genes": int(len(genes)),
        "feature_scope": cfg.get("feature_scope", "output_genes"),
        "n_feature_genes": int(len(aligned["feature_genes"])),
        "outputs": {
            "summary": str(summary_fp),
            "fig4b_table": str(table_fp),
            "full_predictions": str(full_root) if write_full_predictions else None,
            "validation_predictions": str(val_root),
        },
    }
    with (out_root / "run_metadata.json").open("w") as f:
        json.dump(run_meta, f, indent=2)
    return run_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON run config")
    args = parser.parse_args()

    meta = run_one_shot_rrc(load_config(args.config))
    print(f"[summary] {meta['outputs']['summary']}")
    print(f"[table] {meta['outputs']['fig4b_table']}")


if __name__ == "__main__":
    main()

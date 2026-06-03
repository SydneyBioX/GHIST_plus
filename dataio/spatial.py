"""Spatial coordinate and region helpers."""

import logging
import os

import numpy as np
import pandas as pd

import utils.metrics as metric_utils
import utils.utils as utils


def resolve_divisions_fold(opts_regions, fold_id: int):
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


def read_image_hw(fp_img: str):
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
    img = utils.load_image(fp_img)
    return int(img.shape[0]), int(img.shape[1])


def select_region_rows(y_coords: np.ndarray, whole_h: int, divisions_fold, mode: str):
    if divisions_fold is None:
        return np.ones(y_coords.shape[0], dtype=bool)
    div_a = int(round(float(divisions_fold[0]) * whole_h))
    div_b = int(round(float(divisions_fold[1]) * whole_h))
    in_band = (y_coords >= div_a) & (y_coords < div_b)
    if str(mode).lower() == "train":
        return ~in_band
    return in_band


def compute_svg_rank_gene_indices_by_slide(
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
    divisions_fold = resolve_divisions_fold(regions_obj, fold_id)

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

        coord_map = load_histology_coord_map_from_source(src)
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
            kept_ids, coords_yx = centroids_from_label_image(fp_seg, ids_pick, chunk_rows=256)
            if kept_ids.size < 3:
                logging.warning("SVG rank skipped for slide %s: too few centroid-matched cells", slide_id)
                continue
            idx = pd.Index(kept_ids.astype(np.int64))
            expr_arr = df_expr.reindex(idx).to_numpy(dtype=np.float32)
            coords_arr = coords_yx.astype(np.float32, copy=False)
            fp_hist = getattr(src, "fp_hist", None)
            if fp_hist and os.path.isfile(fp_hist):
                whole_h, _ = read_image_hw(fp_hist)
                keep_region = select_region_rows(
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
                whole_h, _ = read_image_hw(fp_hist)
                keep_region = select_region_rows(
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
            scores = metric_utils.giotto_rank_scores(expr_arr, coords_arr, k=k_neighbors)
        except Exception as exc:
            logging.warning("SVG rank failed for slide %s: %s", slide_id, exc)
            continue
        scores = np.nan_to_num(scores.astype(np.float64), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        order = np.argsort(-scores, kind="stable").astype(np.int64)
        ranks_by_slide[slide_id] = order

    return ranks_by_slide


def centroids_from_label_image(
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


def load_histology_coord_map_from_source(src_obj):
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

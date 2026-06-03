"""Reference expression and avgexp prior builders."""

import logging
import os

import numpy as np
import pandas as pd

import dataio.spatial as spatial_utils


def load_ct_series_for_classes(fp_ct, classes):
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


def source_domain_id(src):
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
        domain_id = source_domain_id(src)
        fp_expr_key = getattr(src, "fp_expr", None)
        if fp_expr_key is None or not os.path.isfile(fp_expr_key):
            continue
        ct_series_tmp = load_ct_series_for_classes(getattr(src, "fp_cell_type", None), classes)
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
        domain_id = source_domain_id(src)
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


def build_train_region_avgexp_df_by_slide(
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

    divisions_fold = spatial_utils.resolve_divisions_fold(train_regions, fold_id)

    for src in src_list:
        slide_id = int(getattr(src, "slide_idx", -1))
        domain_id = source_domain_id(src)
        fp_expr = getattr(src, "fp_expr", None)
        if fp_expr is None or not os.path.isfile(fp_expr):
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: missing fp_expr",
                slide_id,
            )
            continue

        ct_series_tmp = load_ct_series_for_classes(getattr(src, "fp_cell_type", None), classes)
        if ct_series_tmp is None:
            logging.warning(
                "Validation train-region avgexp skipped for slide %s: missing fp_cell_type",
                slide_id,
            )
            continue

        coord_map = spatial_utils.load_histology_coord_map_from_source(src)
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

        whole_h, _ = spatial_utils.read_image_hw(getattr(src, "fp_hist"))
        y_coords = np.asarray([float(coord_map[cid][0]) for cid in common_ids], dtype=np.float64)
        keep_train = spatial_utils.select_region_rows(y_coords, whole_h, divisions_fold, mode="train")
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
        domain_id = source_domain_id(src)
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

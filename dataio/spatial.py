"""Spatial coordinate and region helpers."""

import hashlib
import logging
import os

import numpy as np
import pandas as pd

import dataio.utils as image_utils
import utils.metrics as metric_utils


_HISTOLOGY_COORD_MAP_CACHE = {}
_SEGMENTATION_COORD_COMPLETION_CACHE = {}
_HISTOLOGY_COORD_STATS_CACHE = {}


class CoordinateAlignmentError(RuntimeError):
    """Raised when configured global coordinates disagree with segmentation labels."""


def _cell_coords_id_space(src_obj) -> str:
    value = str(getattr(src_obj, "cell_coords_id_space", "auto")).strip().lower()
    if value not in {"auto", "xenium", "histology"}:
        raise ValueError(
            "cell_coords_id_space must be 'auto', 'xenium', or 'histology'; "
            f"got {value!r}"
        )
    return value


def _coordinate_map_from_table(
    matched_ids,
    coordinates,
    *,
    id_column: str,
    x_column: str,
    y_column: str,
    id_space: str,
):
    """Return a histology-ID keyed map for one coordinate ID interpretation."""

    if id_space not in {"xenium", "histology"}:
        raise ValueError(f"cannot build coordinate map for id_space={id_space!r}")
    right = coordinates[[id_column, x_column, y_column]].copy()
    right.columns = ["_coord_id", "_coord_x", "_coord_y"]
    for column in ("_coord_id", "_coord_x", "_coord_y"):
        right[column] = pd.to_numeric(right[column], errors="coerce")
    right = right.dropna(subset=["_coord_id", "_coord_x", "_coord_y"])
    left_key = "id_histology" if id_space == "histology" else "id_xenium"
    matched_nonnull = matched_ids.dropna(subset=[left_key])
    merged = matched_nonnull.merge(
        right, left_on=left_key, right_on="_coord_id", how="inner"
    ).dropna(subset=["id_histology", "_coord_x", "_coord_y"])

    coordinate_map = {}
    for histology_id, coord_y, coord_x in merged[
        ["id_histology", "_coord_y", "_coord_x"]
    ].itertuples(index=False, name=None):
        histology_id = int(histology_id)
        if histology_id > 0 and histology_id not in coordinate_map:
            coordinate_map[histology_id] = (float(coord_y), float(coord_x))
    return coordinate_map


def _coordinate_maps_equal(left, right) -> bool:
    return left.keys() == right.keys() and all(left[key] == right[key] for key in left)


def _coordinate_map_strictly_extends(candidate, subset) -> bool:
    return len(candidate) > len(subset) and all(
        key in candidate and candidate[key] == value for key, value in subset.items()
    )


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
    img = image_utils.load_image(fp_img)
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


def _histology_coord_cache_key(src_obj):
    fp_match = getattr(src_obj, "fp_nuc_sizes", None)
    fp_coords = (
        os.path.join(os.path.dirname(fp_match), "cell_coords.csv")
        if fp_match
        else None
    )
    fp_seg = getattr(src_obj, "fp_nuc_seg", None)

    def _resolved(path):
        return os.path.realpath(os.path.abspath(path)) if path else ""

    requested_id_space = _cell_coords_id_space(src_obj)
    validation_requested = bool(
        getattr(src_obj, "cell_coords_validate_alignment", False)
    )
    validation_settings_needed = validation_requested or requested_id_space == "auto"
    validation_key = None
    if validation_settings_needed:
        validation_key = (
            validation_requested,
            int(getattr(src_obj, "cell_coords_alignment_sample_size", 256)),
            float(getattr(src_obj, "cell_coords_alignment_tolerance_px", 1.0)),
            int(getattr(src_obj, "cell_coords_alignment_window_radius_px", 64)),
        )
    return (
        _resolved(fp_match),
        _resolved(fp_coords),
        _resolved(fp_seg),
        requested_id_space,
        validation_key,
    )


def clear_histology_coord_caches():
    """Clear coordinate caches; intended for isolated tests and explicit reloads."""

    _HISTOLOGY_COORD_MAP_CACHE.clear()
    _SEGMENTATION_COORD_COMPLETION_CACHE.clear()
    _HISTOLOGY_COORD_STATS_CACHE.clear()


def get_histology_coord_completion_stats(src_obj):
    """Return JSON-safe counts from the most recent coordinate-map load."""

    return dict(_HISTOLOGY_COORD_STATS_CACHE.get(_histology_coord_cache_key(src_obj), {}))


def validate_histology_coord_alignment(
    src_obj,
    coordinate_map,
    *,
    sample_size: int | None = None,
    tolerance_px: float | None = None,
    id_space: str | None = None,
):
    """Validate mapped (y,x) coordinates against segmentation centroids.

    The check is deterministic and intentionally opt-in. It memory-maps the
    label TIFF and reads only bounded windows around sampled coordinates; it
    does not compute centroids by scanning the full image. Callers can enable
    it with cell_coords_validate_alignment on a data source. A failed
    configured check raises instead of silently using misaligned coordinates.
    """

    resolved_id_space = str(id_space or _cell_coords_id_space(src_obj)).strip().lower()
    if resolved_id_space not in {"auto", "xenium", "histology"}:
        raise ValueError(f"invalid alignment id_space={resolved_id_space!r}")
    fp_seg = getattr(src_obj, "fp_nuc_seg", None)
    if fp_seg is None or not os.path.isfile(fp_seg):
        raise CoordinateAlignmentError(
            "coordinate alignment validation requires an existing fp_nuc_seg"
        )
    ids = np.asarray(
        sorted(int(cell_id) for cell_id in coordinate_map if int(cell_id) > 0),
        dtype=np.int64,
    )
    if ids.size == 0:
        raise CoordinateAlignmentError(
            "coordinate alignment validation received an empty map"
        )

    if sample_size is None:
        sample_size = int(getattr(src_obj, "cell_coords_alignment_sample_size", 256))
    sample_size = max(1, min(int(sample_size), int(ids.size)))
    if sample_size < ids.size:
        positions = np.linspace(0, ids.size - 1, num=sample_size, dtype=np.int64)
        sample_ids = np.unique(ids[positions])
    else:
        sample_ids = ids

    import tifffile

    try:
        labels = tifffile.memmap(fp_seg)
    except Exception as exc:
        raise CoordinateAlignmentError(
            f"failed to memory-map segmentation for coordinate alignment: {fp_seg}"
        ) from exc
    if labels.ndim != 2:
        raise CoordinateAlignmentError(
            f"coordinate alignment expects a 2D label image, got shape={labels.shape}"
        )
    radius = int(getattr(src_obj, "cell_coords_alignment_window_radius_px", 64))
    if radius < 1:
        raise ValueError("cell_coords_alignment_window_radius_px must be positive")

    errors = []
    missing_near_coordinate = []
    height, width = labels.shape
    for cell_id in sample_ids.tolist():
        mapped_y, mapped_x = coordinate_map[int(cell_id)]
        if not np.isfinite(mapped_y) or not np.isfinite(mapped_x):
            raise CoordinateAlignmentError(
                f"non-finite mapped coordinate for segmentation label {cell_id}"
            )
        center_y, center_x = int(round(mapped_y)), int(round(mapped_x))
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        if y0 >= y1 or x0 >= x1:
            missing_near_coordinate.append(int(cell_id))
            continue
        local_y, local_x = np.nonzero(np.asarray(labels[y0:y1, x0:x1]) == cell_id)
        if local_y.size == 0:
            missing_near_coordinate.append(int(cell_id))
            continue
        centroid_y = float(local_y.mean() + y0)
        centroid_x = float(local_x.mean() + x0)
        errors.append(float(np.hypot(mapped_y - centroid_y, mapped_x - centroid_x)))

    if missing_near_coordinate:
        raise CoordinateAlignmentError(
            "mapped coordinates do not contain their segmentation labels within "
            f"id_space={resolved_id_space} radius={radius}px: "
            f"sampled={sample_ids.size} "
            f"missing={len(missing_near_coordinate)} "
            f"first_missing={missing_near_coordinate[:5]}"
        )
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size != sample_ids.size or not np.isfinite(errors).all():
        raise CoordinateAlignmentError(
            "coordinate alignment produced incomplete or non-finite errors"
        )

    if tolerance_px is None:
        tolerance_px = float(
            getattr(src_obj, "cell_coords_alignment_tolerance_px", 1.0)
        )
    tolerance_px = float(tolerance_px)
    if not np.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError(
            "cell_coords_alignment_tolerance_px must be finite and non-negative"
        )

    stats = {
        "status": "passed",
        "method": "memory_mapped_local_windows",
        "cell_coords_id_space": resolved_id_space,
        "sampled_labels": int(sample_ids.size),
        "window_radius_px": radius,
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.quantile(errors, 0.95)),
        "max_error_px": float(np.max(errors)),
        "tolerance_px": tolerance_px,
    }
    if stats["max_error_px"] > tolerance_px:
        stats["status"] = "failed"
        raise CoordinateAlignmentError(
            "global cell coordinates disagree with segmentation centroids: "
            f"id_space={resolved_id_space} "
            f"sampled={stats['sampled_labels']} "
            f"median_error_px={stats['median_error_px']:.6f} "
            f"p95_error_px={stats['p95_error_px']:.6f} "
            f"max_error_px={stats['max_error_px']:.6f} "
            f"tolerance_px={tolerance_px:.6f}"
        )
    return stats


def _auto_candidate_alignment(
    src_obj,
    coordinate_map,
    *,
    id_space: str,
    id_column: str,
    target_count: int,
):
    """Return JSON-safe alignment evidence for one automatic candidate."""

    evidence = {
        "id_space": id_space,
        "id_column": id_column,
        "mapped_histology_ids": int(len(coordinate_map)),
        "coverage": float(len(coordinate_map) / target_count) if target_count else 0.0,
    }
    if not coordinate_map:
        evidence.update({"status": "failed", "error": "empty coordinate map"})
        return evidence, None
    try:
        alignment = validate_histology_coord_alignment(
            src_obj, coordinate_map, id_space=id_space
        )
    except CoordinateAlignmentError as exc:
        evidence.update({"status": "failed", "error": str(exc)})
        return evidence, None
    evidence.update({"status": "passed", "alignment": alignment})
    return evidence, alignment


def _resolve_coordinate_map_from_table(src_obj, matched_ids, coordinates, x_col, y_col):
    """Resolve a coordinate table's ID namespace without guessing.

    Explicit semantic headers are authoritative. A generic ``cell_id``/``id``
    column is ambiguous, so automatic mode constructs both interpretations and
    validates their CSV coordinates against segmentation labels before any
    centroid completion can hide a bad join.
    """

    requested = _cell_coords_id_space(src_obj)
    generic_id_col = next(
        (column for column in ("cell_id", "id") if column in coordinates.columns),
        None,
    )

    if requested == "histology":
        id_col = "id_histology" if "id_histology" in coordinates.columns else generic_id_col
        if id_col is None:
            raise CoordinateAlignmentError(
                "histology coordinate namespace requested but no compatible ID column exists"
            )
        coordinate_map = _coordinate_map_from_table(
            matched_ids,
            coordinates,
            id_column=id_col,
            x_column=x_col,
            y_column=y_col,
            id_space="histology",
        )
        alignment = None
        if bool(getattr(src_obj, "cell_coords_validate_alignment", False)):
            alignment = validate_histology_coord_alignment(
                src_obj, coordinate_map, id_space="histology"
            )
        return coordinate_map, "histology", id_col, "explicit_override", {}, alignment

    if requested == "xenium":
        id_col = "id_xenium" if "id_xenium" in coordinates.columns else generic_id_col
        if id_col is None:
            raise CoordinateAlignmentError(
                "Xenium coordinate namespace requested but no compatible ID column exists"
            )
        coordinate_map = _coordinate_map_from_table(
            matched_ids,
            coordinates,
            id_column=id_col,
            x_column=x_col,
            y_column=y_col,
            id_space="xenium",
        )
        alignment = None
        if bool(getattr(src_obj, "cell_coords_validate_alignment", False)):
            alignment = validate_histology_coord_alignment(
                src_obj, coordinate_map, id_space="xenium"
            )
        return coordinate_map, "xenium", id_col, "explicit_override", {}, alignment

    if "id_histology" in coordinates.columns:
        coordinate_map = _coordinate_map_from_table(
            matched_ids,
            coordinates,
            id_column="id_histology",
            x_column=x_col,
            y_column=y_col,
            id_space="histology",
        )
        alignment = None
        if bool(getattr(src_obj, "cell_coords_validate_alignment", False)):
            alignment = validate_histology_coord_alignment(
                src_obj, coordinate_map, id_space="histology"
            )
        return (
            coordinate_map,
            "histology",
            "id_histology",
            "explicit_id_histology_column",
            {},
            alignment,
        )

    if "id_xenium" in coordinates.columns:
        coordinate_map = _coordinate_map_from_table(
            matched_ids,
            coordinates,
            id_column="id_xenium",
            x_column=x_col,
            y_column=y_col,
            id_space="xenium",
        )
        alignment = None
        if bool(getattr(src_obj, "cell_coords_validate_alignment", False)):
            alignment = validate_histology_coord_alignment(
                src_obj, coordinate_map, id_space="xenium"
            )
        return (
            coordinate_map,
            "xenium",
            "id_xenium",
            "explicit_id_xenium_column",
            {},
            alignment,
        )

    if generic_id_col is None:
        raise CoordinateAlignmentError(
            "automatic coordinate namespace resolution requires an ID column"
        )

    candidates = {}
    evidence = {}
    alignments = {}
    target_count = int(matched_ids["id_histology"].nunique())
    for id_space in ("histology", "xenium"):
        candidate = _coordinate_map_from_table(
            matched_ids,
            coordinates,
            id_column=generic_id_col,
            x_column=x_col,
            y_column=y_col,
            id_space=id_space,
        )
        candidates[id_space] = candidate
        evidence[id_space], alignments[id_space] = _auto_candidate_alignment(
            src_obj,
            candidate,
            id_space=id_space,
            id_column=generic_id_col,
            target_count=target_count,
        )

    passed = [space for space in ("histology", "xenium") if alignments[space] is not None]
    if len(passed) == 1:
        selected = passed[0]
        method = "segmentation_alignment_unique_pass"
    elif len(passed) == 2:
        histology_map = candidates["histology"]
        xenium_map = candidates["xenium"]
        if _coordinate_maps_equal(histology_map, xenium_map):
            selected = "xenium"
            method = "segmentation_alignment_equivalent_maps"
        elif _coordinate_map_strictly_extends(histology_map, xenium_map):
            selected = "histology"
            method = "segmentation_alignment_consistent_superset"
        elif _coordinate_map_strictly_extends(xenium_map, histology_map):
            selected = "xenium"
            method = "segmentation_alignment_consistent_superset"
        else:
            raise CoordinateAlignmentError(
                "automatic coordinate namespace is ambiguous: both histology and "
                "Xenium interpretations align but produce different maps; set an "
                "explicit cell_coords_id_space override"
            )
    else:
        failures = {space: evidence[space].get("error") for space in evidence}
        raise CoordinateAlignmentError(
            "automatic coordinate namespace resolution failed: neither histology "
            f"nor Xenium coordinates align with segmentation labels; failures={failures}"
        )

    return (
        candidates[selected],
        selected,
        generic_id_col,
        method,
        evidence,
        alignments[selected],
    )


def load_histology_coord_map_from_source(src_obj):
    """Load complete histology coordinates, preserving CSV values.

    ``cell_coords.csv`` remains authoritative. Explicit semantic ID headers are
    honored. Generic ``cell_id`` tables are resolved automatically by checking
    both namespace interpretations against segmentation labels before missing
    coordinates are centroid-filled. An explicit ``cell_coords_id_space``
    override remains available for exceptional or legacy sources.
    """

    fp_match = getattr(src_obj, "fp_nuc_sizes", None)
    if fp_match is None or not os.path.isfile(fp_match):
        return {}
    fp_coords = os.path.join(os.path.dirname(fp_match), "cell_coords.csv")
    fp_seg = getattr(src_obj, "fp_nuc_seg", None)
    requested_coord_id_space = _cell_coords_id_space(src_obj)
    cache_key = _histology_coord_cache_key(src_obj)
    cached = _HISTOLOGY_COORD_MAP_CACHE.get(cache_key)
    if cached is not None:
        stats = dict(_HISTOLOGY_COORD_STATS_CACHE.get(cache_key, {}))
        stats["cache_hit"] = True
        _HISTOLOGY_COORD_STATS_CACHE[cache_key] = stats
        return dict(cached)

    try:
        df_match = pd.read_csv(fp_match)
        if not {"id_histology", "id_xenium"}.issubset(set(df_match.columns)):
            logging.warning("Coordinate completion skipped: invalid match columns in %s", fp_match)
            return {}
        left = df_match[["id_histology", "id_xenium"]].copy()
        left["id_histology"] = pd.to_numeric(left["id_histology"], errors="coerce")
        left["id_xenium"] = pd.to_numeric(left["id_xenium"], errors="coerce")
        left = left.dropna(subset=["id_histology"])
        left["id_histology"] = left["id_histology"].astype(np.int64)
        target_ids = np.unique(left["id_histology"].to_numpy(dtype=np.int64))
        target_ids = target_ids[target_ids > 0]

        coord_map = {}
        resolved_coord_id_space = None
        coordinate_id_column = None
        resolution_method = "no_coordinate_csv"
        candidate_evidence = {}
        alignment_stats = None
        if os.path.isfile(fp_coords):
            df_coords = pd.read_csv(fp_coords)
            x_col = next((c for c in ("x_coord", "x", "X") if c in df_coords.columns), None)
            y_col = next((c for c in ("y_coord", "y", "Y") if c in df_coords.columns), None)
            if x_col is not None and y_col is not None:
                (
                    coord_map,
                    resolved_coord_id_space,
                    coordinate_id_column,
                    resolution_method,
                    candidate_evidence,
                    alignment_stats,
                ) = _resolve_coordinate_map_from_table(
                    src_obj, left, df_coords, x_col, y_col
                )
            else:
                raise CoordinateAlignmentError(
                    f"coordinate CSV has no recognized x/y columns: {fp_coords}"
                )

        if resolved_coord_id_space is None:
            # With no usable CSV, centroid completion is already keyed by
            # histology segmentation labels and no namespace inference occurs.
            resolved_coord_id_space = (
                requested_coord_id_space
                if requested_coord_id_space in {"histology", "xenium"}
                else "histology"
            )
            resolution_method = "segmentation_completion_only"

        csv_count = len(coord_map)
        alignment_requested = bool(
            getattr(src_obj, "cell_coords_validate_alignment", False)
        )
        missing_ids = np.asarray(
            [cell_id for cell_id in target_ids.tolist() if int(cell_id) not in coord_map],
            dtype=np.int64,
        )
        centroid_filled = 0
        segmentation_scan_performed = False
        if missing_ids.size and fp_seg and os.path.isfile(fp_seg):
            missing_ids_contiguous = np.ascontiguousarray(missing_ids, dtype=np.int64)
            missing_ids_sha256 = hashlib.sha256(
                missing_ids_contiguous.tobytes(order="C")
            ).hexdigest()
            seg_key = (
                os.path.realpath(os.path.abspath(fp_seg)),
                resolved_coord_id_space,
                int(missing_ids_contiguous.size),
                missing_ids_sha256,
            )
            completion = _SEGMENTATION_COORD_COMPLETION_CACHE.get(seg_key)
            if completion is None:
                kept_ids, coords_yx = centroids_from_label_image(
                    fp_seg, missing_ids, chunk_rows=256
                )
                completion = {
                    int(cell_id): (float(coord[0]), float(coord[1]))
                    for cell_id, coord in zip(kept_ids.tolist(), coords_yx.tolist())
                }
                _SEGMENTATION_COORD_COMPLETION_CACHE[seg_key] = completion
                segmentation_scan_performed = True
            for cell_id in missing_ids.tolist():
                cell_id = int(cell_id)
                if cell_id not in coord_map and cell_id in completion:
                    coord_map[cell_id] = completion[cell_id]
                    centroid_filled += 1

        unresolved = int(sum(int(cell_id) not in coord_map for cell_id in target_ids.tolist()))
        if alignment_requested and alignment_stats is None:
            alignment_stats = validate_histology_coord_alignment(
                src_obj, coord_map, id_space=resolved_coord_id_space
            )
        stats = {
            "cell_coords_id_space": resolved_coord_id_space,
            "cell_coords_id_space_requested": requested_coord_id_space,
            "cell_coords_id_space_resolution": resolution_method,
            "cell_coords_id_column": coordinate_id_column,
            "cell_coords_auto_candidates": candidate_evidence,
            "matched_histology_ids": int(target_ids.size),
            "csv_coordinates": int(csv_count),
            "csv_coordinate_coverage": (
                float(csv_count / target_ids.size) if target_ids.size else 0.0
            ),
            "centroid_filled": int(centroid_filled),
            "unresolved": unresolved,
            "total_coordinates": int(len(coord_map)),
            "alignment": alignment_stats,
            "segmentation_scan_performed": bool(segmentation_scan_performed),
            "cache_hit": False,
            "fp_match": cache_key[0],
            "fp_coords": cache_key[1],
            "fp_nuc_seg": cache_key[2],
        }
        _HISTOLOGY_COORD_MAP_CACHE[cache_key] = dict(coord_map)
        _HISTOLOGY_COORD_STATS_CACHE[cache_key] = stats
        logging.info(
            "Histology coordinates slide=%s requested_id_space=%s id_space=%s "
            "resolution=%s id_column=%s matched=%d csv=%d "
            "centroid_filled=%d unresolved=%d total=%d segmentation_scan=%s "
            "alignment=%s",
            getattr(src_obj, "slide_idx", "unknown"),
            requested_coord_id_space,
            resolved_coord_id_space,
            resolution_method,
            coordinate_id_column,
            stats["matched_histology_ids"],
            stats["csv_coordinates"],
            stats["centroid_filled"],
            stats["unresolved"],
            stats["total_coordinates"],
            stats["segmentation_scan_performed"],
            alignment_stats["status"] if alignment_stats is not None else "not_requested",
        )
        return dict(coord_map)
    except CoordinateAlignmentError:
        raise
    except Exception as exc:
        if os.path.isfile(fp_coords):
            raise CoordinateAlignmentError(
                f"failed to load or resolve coordinate table: {fp_coords}"
            ) from exc
        logging.warning("Failed to complete histology coordinates: %s", exc)
        return {}

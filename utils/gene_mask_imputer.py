"""Task helpers for fixed masked-gene imputation training."""

import hashlib
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd


TASK_NAME = "gene_mask_imputer"
TASK_DESCRIPTION = "Masked-gene imputation from observed genes and morphology"


def to_namespace(obj):
    if obj is None:
        return None
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, "_asdict"):
            return to_namespace(obj._asdict())
        return tuple(to_namespace(v) for v in obj)
    return obj


def to_serialisable(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: to_serialisable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def resolve_local_path(path: str, *, repo_root: str) -> str:
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def resolve_config(raw_cfg, *, repo_root: str):
    cfg = to_namespace(raw_cfg) or SimpleNamespace()
    resolved = {
        "enabled": bool(getattr(cfg, "enabled", True)),
        "mask_n_genes": int(getattr(cfg, "mask_n_genes", 10)),
        "mask_strategy": str(getattr(cfg, "mask_strategy", "top_giotto_svg_per_slide")),
        "svg_knn_k": int(getattr(cfg, "svg_knn_k", 8)),
        "svg_sample_cap": int(getattr(cfg, "svg_sample_cap", 3000)),
        "fixed_gene_csv_dir": str(
            getattr(cfg, "fixed_gene_csv_dir", "configs/gene_mask_imputer_fixed_giotto")
        ),
        "create_fixed_gene_csv_if_missing": bool(
            getattr(cfg, "create_fixed_gene_csv_if_missing", True)
        ),
        "random_mask_frac": float(getattr(cfg, "random_mask_frac", 0.30)),
        "use_morph": bool(getattr(cfg, "use_morph", True)),
        "copy_observed": bool(getattr(cfg, "copy_observed", True)),
        "zero_masked_gene_avgexp": bool(getattr(cfg, "zero_masked_gene_avgexp", False)),
        "mask_seed": int(getattr(cfg, "mask_seed", 0)),
    }
    resolved["fixed_gene_csv_dir"] = resolve_local_path(
        resolved["fixed_gene_csv_dir"], repo_root=repo_root
    )
    if resolved["mask_n_genes"] <= 0:
        raise ValueError("gene_mask_imputer.mask_n_genes must be positive.")
    if resolved["svg_knn_k"] <= 0:
        raise ValueError("gene_mask_imputer.svg_knn_k must be positive.")
    if resolved["svg_sample_cap"] <= 0:
        raise ValueError("gene_mask_imputer.svg_sample_cap must be positive.")
    if not (0.0 <= resolved["random_mask_frac"] <= 1.0):
        raise ValueError("gene_mask_imputer.random_mask_frac must be in [0, 1].")
    return resolved


def log_config(cfg):
    logging.info(
        "Gene-mask imputer config: enabled=%s mask_n_genes=%d strategy=%s "
        "random_mask_frac=%.2f svg_knn_k=%d svg_sample_cap=%d "
        "fixed_gene_csv_dir=%s create_fixed_csv=%s use_morph=%s "
        "copy_observed=%s mask_seed=%d",
        cfg["enabled"],
        cfg["mask_n_genes"],
        cfg["mask_strategy"],
        cfg["random_mask_frac"],
        cfg["svg_knn_k"],
        cfg["svg_sample_cap"],
        cfg["fixed_gene_csv_dir"],
        cfg["create_fixed_gene_csv_if_missing"],
        cfg["use_morph"],
        cfg["copy_observed"],
        cfg["mask_seed"],
    )


def avgexp_holdout_fill_strategy(cfg):
    return "zero" if cfg["enabled"] and cfg["zero_masked_gene_avgexp"] else "leave_one_slide_out"


def fixed_giotto_csv_path(csv_dir: str, slide_id: int) -> str:
    return os.path.join(csv_dir, f"slide{int(slide_id)}_giotto_ranked_genes.csv")


def _read_fixed_giotto_csv(csv_dir: str, slide_id: int, gene_names, mask_n: int):
    fp = fixed_giotto_csv_path(csv_dir, slide_id)
    if not os.path.isfile(fp):
        return None

    df = pd.read_csv(fp)
    if "gene" not in df.columns:
        raise ValueError(f"Fixed Giotto CSV is missing required 'gene' column: {fp}")
    if "rank" in df.columns:
        df = df.assign(_rank=pd.to_numeric(df["rank"], errors="coerce")).sort_values(
            "_rank", kind="stable"
        )

    gene_set = set(str(g) for g in gene_names)
    genes = []
    for g_raw in df["gene"].tolist():
        gene = str(g_raw)
        if gene not in gene_set:
            raise ValueError(f"Fixed Giotto CSV gene '{gene}' is not in gene union: {fp}")
        if gene not in genes:
            genes.append(gene)
        if len(genes) >= int(mask_n):
            break

    if len(genes) < int(mask_n):
        raise ValueError(
            f"Fixed Giotto CSV has only {len(genes)} valid unique gene(s), "
            f"expected {int(mask_n)}: {fp}"
        )
    return genes


def _write_fixed_giotto_csv(csv_dir: str, slide_id: int, rank_order, gene_names, mask_n: int):
    os.makedirs(csv_dir, exist_ok=True)
    rows = []
    seen = set()
    for gene_idx_raw in rank_order:
        gene_idx = int(gene_idx_raw)
        if gene_idx < 0 or gene_idx >= len(gene_names):
            continue
        gene = str(gene_names[gene_idx])
        if gene in seen:
            continue
        seen.add(gene)
        rows.append(
            {
                "slide_id": int(slide_id),
                "rank": len(rows) + 1,
                "gene": gene,
                "gene_index": gene_idx,
                "selected_for_mask": True,
            }
        )
        if len(rows) >= int(mask_n):
            break

    if len(rows) < int(mask_n):
        raise ValueError(
            f"Could not write fixed Giotto CSV for slide {int(slide_id)}: "
            f"only {len(rows)} ranked gene(s), expected {int(mask_n)}."
        )

    fp = fixed_giotto_csv_path(csv_dir, slide_id)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def prepare_holdout_masks(
    *,
    sources_trainval,
    expr_per_source,
    gene_names,
    mask_n,
    cfg,
    fold_id,
    experiment_path,
    metrics_dir,
    spatial_utils,
):
    holdout_genes_by_slide = {}
    holdout_mask_by_slide = {}
    if int(mask_n) <= 0:
        return holdout_genes_by_slide, holdout_mask_by_slide, "none"

    supported_mask_strategies = {"top_giotto_svg_per_slide", "fixed_giotto_csv_per_slide"}
    if cfg["mask_strategy"] not in supported_mask_strategies:
        raise ValueError(
            "gene_mask_imputer.mask_strategy must be one of "
            f"{sorted(supported_mask_strategies)}."
        )

    fixed_gene_csv_dir = cfg["fixed_gene_csv_dir"]
    os.makedirs(fixed_gene_csv_dir, exist_ok=True)
    missing_csv_sources = [
        src
        for src in sources_trainval
        if not os.path.isfile(
            fixed_giotto_csv_path(fixed_gene_csv_dir, int(getattr(src, "slide_idx", -1)))
        )
    ]

    if missing_csv_sources:
        if not cfg["create_fixed_gene_csv_if_missing"]:
            missing_ids = ", ".join(
                str(int(getattr(src, "slide_idx", -1))) for src in missing_csv_sources
            )
            raise RuntimeError(
                "Missing fixed Giotto CSV(s) for slide(s): "
                f"{missing_ids}. Expected files under {fixed_gene_csv_dir}."
            )

        whole_slide_ranks = spatial_utils.compute_svg_rank_gene_indices_by_slide(
            missing_csv_sources,
            None,
            fold_id,
            mode_name="all",
            gene_names=gene_names,
            k_neighbors=int(cfg["svg_knn_k"]),
            sample_cap=int(cfg["svg_sample_cap"]),
        )
        for src in missing_csv_sources:
            slide_id = int(getattr(src, "slide_idx", -1))
            rank_order = whole_slide_ranks.get(slide_id)
            if rank_order is None or len(rank_order) == 0:
                raise RuntimeError(
                    f"Failed to compute whole-slide Giotto rank for slide {slide_id}; "
                    "fixed masked-gene CSV was not created."
                )
            fp_written = _write_fixed_giotto_csv(
                fixed_gene_csv_dir, slide_id, rank_order, gene_names, mask_n
            )
            logging.info("Created fixed Giotto masked-gene CSV: %s", fp_written)

    logging.info(
        "Using fixed whole-slide Giotto masked-gene CSVs: dir=%s slides=%d "
        "mask_n_genes=%d kNN=%d sample_cap=%d",
        fixed_gene_csv_dir,
        len(sources_trainval),
        int(mask_n),
        int(cfg["svg_knn_k"]),
        int(cfg["svg_sample_cap"]),
    )

    for src in sources_trainval:
        slide_id = int(getattr(src, "slide_idx", -1))
        chosen = _read_fixed_giotto_csv(fixed_gene_csv_dir, slide_id, gene_names, mask_n)
        if chosen is None:
            raise RuntimeError(
                f"Missing fixed Giotto CSV for slide {slide_id}: "
                f"{fixed_giotto_csv_path(fixed_gene_csv_dir, slide_id)}"
            )
        df_expr_tmp = expr_per_source[src.fp_expr]
        missing_in_slide = [g for g in chosen if g not in df_expr_tmp.columns]
        if missing_in_slide:
            raise RuntimeError(
                f"Fixed Giotto CSV for slide {slide_id} contains gene(s) not measured "
                f"in that slide: {', '.join(missing_in_slide)}"
            )

        holdout_genes_by_slide[slide_id] = chosen
        mask = np.zeros(len(gene_names), dtype=np.float32)
        for gene in chosen:
            mask[gene_names.index(gene)] = 1.0
        holdout_mask_by_slide[slide_id] = mask
        logging.info(
            "Fixed holdout slide %s: %d genes from %s",
            slide_id,
            len(chosen),
            fixed_giotto_csv_path(fixed_gene_csv_dir, slide_id),
        )
        logging.info("Holdout genes slide %s: %s", slide_id, ", ".join(chosen))

    holdout_hash = _write_holdout_manifest(
        sources_trainval=sources_trainval,
        holdout_genes_by_slide=holdout_genes_by_slide,
        mask_n=mask_n,
        cfg=cfg,
        experiment_path=experiment_path,
        metrics_dir=metrics_dir,
    )
    return holdout_genes_by_slide, holdout_mask_by_slide, holdout_hash


def _write_holdout_manifest(
    *,
    sources_trainval,
    holdout_genes_by_slide,
    mask_n,
    cfg,
    experiment_path,
    metrics_dir,
):
    invalid_holdout = []
    for src in sources_trainval:
        sid = int(getattr(src, "slide_idx", -1))
        genes = holdout_genes_by_slide.get(sid, [])
        if len(genes) != int(mask_n):
            invalid_holdout.append((sid, len(genes)))
    if invalid_holdout:
        raise RuntimeError(
            "Top-SVG holdout selection failed for slide(s): "
            + ", ".join(f"{sid}:{n}" for sid, n in invalid_holdout)
        )

    holdout_manifest = {
        str(int(sid)): [str(g) for g in genes]
        for sid, genes in sorted(holdout_genes_by_slide.items(), key=lambda item: int(item[0]))
    }
    holdout_hash = hashlib.md5(
        json.dumps(holdout_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    holdout_payload = {
        "task_name": TASK_NAME,
        "mask_strategy": cfg["mask_strategy"],
        "mask_n_genes": int(mask_n),
        "svg_knn_k": int(cfg["svg_knn_k"]),
        "svg_sample_cap": int(cfg["svg_sample_cap"]),
        "holdout_hash": holdout_hash,
        "genes_by_slide": holdout_manifest,
    }
    write_json(os.path.join(experiment_path, "holdout_genes_by_slide.json"), holdout_payload)
    write_json(os.path.join(metrics_dir, "holdout_genes_by_slide.json"), holdout_payload)
    logging.info(
        "Saved top-SVG holdout manifest: slides=%d mask_n_genes=%d hash=%s",
        len(holdout_manifest),
        int(mask_n),
        holdout_hash,
    )
    return holdout_hash


def log_holdout_svg_pearson(metrics: dict, *, split_tag: str, epoch: int):
    if not isinstance(metrics, dict):
        logging.info("%s HoldoutSVG10PCC epoch=%d unavailable", split_tag, int(epoch))
        return

    vals = []
    for slide_vals in (metrics.get("holdout_pearson_per_gene") or {}).values():
        if not isinstance(slide_vals, dict):
            continue
        for val in slide_vals.values():
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val_f):
                vals.append(val_f)

    if not vals:
        logging.info("%s HoldoutSVG10PCC epoch=%d unavailable", split_tag, int(epoch))
        return

    vals_np = np.asarray(vals, dtype=np.float64)
    logging.info(
        "%s HoldoutSVG10PCC epoch=%d median=%.6f min=%.6f max=%.6f mean=%.6f n=%d",
        str(split_tag).upper(),
        int(epoch),
        float(np.median(vals_np)),
        float(np.min(vals_np)),
        float(np.max(vals_np)),
        float(np.mean(vals_np)),
        int(vals_np.size),
    )

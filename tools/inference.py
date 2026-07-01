#!/usr/bin/env python3
"""
Unified GHIST+ inference/evaluation entrypoint.

- Export per-cell expression / cell-type predictions
- Optionally score predictions with the same evaluate_validation() path used in training
"""

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _load_nature_trainer(nature_root: Path):
    fp = nature_root / "train.py"
    nature_root_str = str(nature_root.resolve())
    if nature_root_str not in sys.path:
        sys.path.insert(0, nature_root_str)
    spec = importlib.util.spec_from_file_location("train_module", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cached_source_fp(impute_dir: Path, split_name: str, slide_id: int, domain_id: int, suffix: str):
    return impute_dir / f"{split_name}_slide{int(slide_id)}_domain{int(domain_id)}_{suffix}"


def _resolve_cached_source_fp(impute_dir: Path, split_name: str, slide_id: int, domain_id: int, suffix: str):
    direct = _cached_source_fp(impute_dir, split_name, slide_id, domain_id, suffix)
    if direct.is_file():
        return direct

    basename = impute_dir.name
    search_roots = []
    for root_s in os.environ.get("SEARCH_ROOTS", "").split(os.pathsep):
        root_s = root_s.strip()
        if root_s:
            search_roots.append(Path(root_s).expanduser().resolve())
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for alt_dir in root.glob(f"*/{basename}"):
            cand = _cached_source_fp(alt_dir, split_name, slide_id, domain_id, suffix)
            if cand.is_file():
                candidates.append(cand)
    if not candidates:
        return direct
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _epoch_num(path: Path):
    try:
        return int(path.name.split("_")[1])
    except Exception:
        return -1


def _find_best_checkpoint(experiment_path: Path, metrics_dir: Path | None):
    candidate_jsons = [experiment_path / "strict_best.json"]
    if metrics_dir is not None:
        candidate_jsons.append(metrics_dir / "strict_best.json")

    for fp_json in candidate_jsons:
        if not fp_json.is_file():
            continue
        try:
            with open(fp_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        best_checkpoint = payload.get("best_checkpoint")
        if best_checkpoint:
            fp_ckpt = Path(best_checkpoint).expanduser().resolve()
            if fp_ckpt.is_file():
                return fp_ckpt

        best_epoch = payload.get("best_epoch")
        if best_epoch:
            fp_ckpt = experiment_path / "models" / f"epoch_{int(best_epoch)}_model.pth"
            if fp_ckpt.is_file():
                return fp_ckpt.resolve()
    return None


def _find_checkpoint_candidates(experiment_path: Path, checkpoint_path: str | None, epoch: int, metrics_dir: Path | None = None):
    if checkpoint_path:
        fp = Path(checkpoint_path).expanduser().resolve()
        if not fp.is_file():
            raise FileNotFoundError(fp)
        return [fp]

    model_dir = experiment_path / "models"
    if epoch > 0:
        fp = model_dir / f"epoch_{int(epoch)}_model.pth"
        if not fp.is_file():
            raise FileNotFoundError(fp)
        return [fp]

    all_ckpts = sorted(model_dir.glob("epoch_*_model.pth"), key=_epoch_num, reverse=True)
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    best_ckpt = _find_best_checkpoint(experiment_path, metrics_dir)
    if best_ckpt is None:
        return all_ckpts
    remaining = [fp for fp in all_ckpts if fp.resolve() != best_ckpt.resolve()]
    return [best_ckpt] + remaining


def _resolve_regions(opts, split_name: str):
    if split_name == "test":
        reg_test = getattr(opts, "regions_test", None)
        if reg_test is None:
            raise ValueError(
                "regions_test must be set for test inference; no fallback to regions_val is allowed."
            )
        return reg_test
    reg_val = getattr(opts, "regions_val", None)
    if reg_val is None:
        raise ValueError("regions_val must be set for non-test inference.")
    return reg_val


def _build_model(gba, opts, n_classes, n_genes, n_ref, use_avgexp, use_celltype, use_neighb, device):
    fm_cfg = getattr(opts.model, "foundation_model", None)
    if fm_cfg is None:
        fm_cfg = SimpleNamespace(pretrained=False)
        opts.model.foundation_model = fm_cfg
    else:
        try:
            setattr(fm_cfg, "pretrained", False)
        except Exception:
            pass
    framework_cls = gba.model_framework.Framework
    try:
        model = framework_cls(
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
    except TypeError:
        model = framework_cls(
            n_classes,
            n_genes,
            opts.model.emb_dim,
            device,
            n_ref,
            use_avgexp,
            use_celltype,
            use_neighb,
        )
    holdout_n_genes = int(getattr(getattr(opts, "training", None), "holdout_n_genes", 20))
    panel_completion_enabled = holdout_n_genes > 0
    completion_head_cls = getattr(gba.panel_completion, "PanelCompletionHead", None)
    if panel_completion_enabled and completion_head_cls is not None:
        model.completion_head = completion_head_cls(
            n_genes,
            hidden_dim=256,
            dropout=0.0,
            use_morph=True,
            morph_gate_init=-2.0,
        )
    return model.to(device)


def _flatten_patch_ids(batch_patch_ids: torch.Tensor, batch_n_cells: torch.Tensor):
    ids = []
    n_cells_flat = batch_n_cells.view(-1)
    for b in range(batch_patch_ids.shape[0]):
        n_valid = int(n_cells_flat[b].item()) if b < n_cells_flat.numel() else 0
        if n_valid <= 0:
            continue
        ids.append(batch_patch_ids[b, :n_valid])
    if not ids:
        return None
    return torch.cat(ids, dim=0)


def _infer_default_config(experiment_path: Path):
    config_candidates = sorted(experiment_path.glob("*.json"))
    if len(config_candidates) == 1:
        return config_candidates[0]
    for fp in config_candidates:
        if fp.name.startswith("config_"):
            return fp
    raise FileNotFoundError(
        f"Could not infer config_file from {experiment_path}. Pass --config_file explicitly."
    )


def _log(msg: str):
    print(msg, flush=True)


def _extract_per_gene_summary(metrics: dict):
    summary = {
        "mean": metrics.get("pearson_gene_pooled_mean"),
        "median": metrics.get("pearson_gene_pooled_median"),
        "max": metrics.get("pearson_gene_pooled_max"),
        "min": None,
        "n_genes": metrics.get("pearson_gene_pooled_n_genes"),
    }
    dist = metrics.get("gene_pcc_distribution_per_slide") or {}
    if isinstance(dist, dict) and dist:
        first_sid = sorted(dist.keys(), key=lambda x: int(x))[0]
        all_stats = (dist.get(first_sid) or {}).get("all") or {}
        summary["min"] = all_stats.get("min")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run GHIST+ inference from a trained checkpoint and export predictions."
    )
    parser.add_argument("--config_file", type=str, default="", help="Defaults to the run config inside experiment_path.")
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=-1, help="-1 means best/latest checkpoint resolution")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--fold_id", type=int, default=1)
    parser.add_argument("--slide_id", type=int, default=3)
    parser.add_argument("--impute_dir", type=str, required=True, help="Directory containing cached imputed expr.csv/mask.npy files.")
    parser.add_argument("--nature_root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Path to the GHIST+ package root.")
    parser.add_argument("--num_workers", type=int, default=-1, help="-1 uses config data.num_workers")
    parser.add_argument("--batch_size", type=int, default=0, help="0 uses config training.batch_size")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_counts_csv", action="store_true")
    parser.add_argument("--log_every", type=int, default=50, help="Log progress every N batches during forward inference")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip evaluate_validation-style metric computation even if targets are available.")
    args = parser.parse_args()

    nature_root = Path(args.nature_root).expanduser().resolve()
    for path_str in (str(nature_root),):
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(nature_root))

    gba = _load_nature_trainer(nature_root)
    from utils.utils import get_device, json_file_to_pyobj, read_txt

    experiment_path = Path(args.experiment_path).expanduser().resolve()
    config_file = (
        Path(args.config_file).expanduser().resolve()
        if args.config_file
        else _infer_default_config(experiment_path)
    )
    impute_dir = Path(args.impute_dir).expanduser().resolve()
    cache_root_env = os.environ.get("CACHE_ROOT")
    cache_root = (
        Path(cache_root_env).expanduser().resolve()
        if cache_root_env
        else impute_dir.parent
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["CACHE_ROOT"] = str(cache_root)
    _log(f"[INFO] CACHE_ROOT={cache_root}")

    norm_fp = experiment_path / f"standardisation_hist_fold_{int(args.fold_id)}.npy"
    if not norm_fp.is_file():
        candidates = sorted(
            experiment_path.parent.glob(f"fold*/standardisation_hist_fold_{int(args.fold_id)}.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            experiment_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], norm_fp)
            _log(f"[WARN] restored missing standardisation file from: {candidates[0]}")
        else:
            raise FileNotFoundError(
                f"Missing {norm_fp} and no fallback standardisation file found under {experiment_path.parent}"
            )

    opts = gba._to_namespace(json_file_to_pyobj(str(config_file)))
    if not hasattr(opts, "model") or opts.model is None:
        opts.model = SimpleNamespace()
    if not hasattr(opts.model, "ecrm") or opts.model.ecrm is None:
        opts.model.ecrm = SimpleNamespace()

    opts.model.use_gt_ct_ref_weights = False
    opts.model.ecrm.use_gt_ct = False

    metrics_dir = Path(getattr(getattr(opts, "experiment_dirs", None), "metrics_dir", nature_root / "metrics"))
    checkpoint_candidates = _find_checkpoint_candidates(
        experiment_path, args.checkpoint_path, args.epoch, metrics_dir=metrics_dir
    )
    _log(f"[INFO] checkpoint candidates (best-first): {[p.name for p in checkpoint_candidates[:5]]}")

    device = get_device(args.gpu_id)
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
    _log(f"[INFO] device={device}")

    classes = list(getattr(opts.data, "cell_types", []))
    n_classes = len(classes)
    use_avgexp = bool(opts.comps.avgexp)
    use_celltype = bool(opts.comps.celltype)
    use_neighb = (
        bool(getattr(opts.comps, "neighb", getattr(opts.comps, "use_neighb", False)))
        if use_celltype
        else False
    )
    expr_scale = float(getattr(opts.data, "expr_scale", 1.0))
    avgexp_domain_specific = bool(
        getattr(getattr(opts, "model", SimpleNamespace()), "avgexp_domain_specific", False)
    )

    genes_fp = experiment_path / "genes.txt"
    gene_names = read_txt(str(genes_fp))
    n_genes = len(gene_names)
    _log(f"[INFO] n_genes={n_genes} n_classes={n_classes}")

    train_sources = [gba._to_namespace(s) for s in list(getattr(opts, "data_sources_train_val", []))]
    test_sources = [gba._to_namespace(s) for s in list(getattr(opts, "data_sources_test", []))]
    resolved_cache_roots = set()
    for split_name, src_list in (("trainval", train_sources), ("test", test_sources)):
        for src in src_list:
            sid = int(getattr(src, "slide_idx", -1))
            domain_id = int(getattr(src, "domain_id", 0))
            fp_expr = _resolve_cached_source_fp(impute_dir, split_name, sid, domain_id, "expr.csv")
            fp_mask = _resolve_cached_source_fp(impute_dir, split_name, sid, domain_id, "mask.npy")
            if fp_expr.is_file():
                src.fp_expr = str(fp_expr)
                resolved_cache_roots.add(str(fp_expr.parent))
            if fp_mask.is_file():
                src.fp_mask = str(fp_mask)
    if resolved_cache_roots:
        _log(f"[INFO] resolved imputed cache roots: {sorted(resolved_cache_roots)}")

    src_eval = next((s for s in train_sources if int(getattr(s, "slide_idx", -1)) == int(args.slide_id)), None)
    eval_split = "trainval"
    if src_eval is None:
        src_eval = next((s for s in test_sources if int(getattr(s, "slide_idx", -1)) == int(args.slide_id)), None)
        eval_split = "test"
    if src_eval is None:
        raise ValueError(f"slide_id={args.slide_id} not found in data_sources_train_val or data_sources_test")
    eval_domain_id = int(getattr(src_eval, "domain_id", 0))
    _log(f"[INFO] eval_slide={args.slide_id} split={eval_split} domain_id={eval_domain_id}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        epoch_tag = checkpoint_candidates[0].stem.split("_")[1] if checkpoint_candidates and checkpoint_candidates[0].stem.startswith("epoch_") else "latest"
        out_dir = experiment_path / f"inference_epoch{epoch_tag}_gpu{int(args.gpu_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[INFO] out_dir={out_dir}")

    src_eval = copy.deepcopy(src_eval)
    fp_expr = _resolve_cached_source_fp(impute_dir, eval_split, args.slide_id, eval_domain_id, "expr.csv")
    fp_mask = _resolve_cached_source_fp(impute_dir, eval_split, args.slide_id, eval_domain_id, "mask.npy")
    if fp_expr.is_file():
        src_eval.fp_expr = str(fp_expr)
    if fp_mask.is_file():
        src_eval.fp_mask = str(fp_mask)

    has_eval_targets = (
        hasattr(src_eval, "fp_expr") and src_eval.fp_expr and os.path.isfile(str(src_eval.fp_expr))
        and bool(opts.comps.celltype)
        and hasattr(src_eval, "fp_cell_type") and src_eval.fp_cell_type and os.path.isfile(str(src_eval.fp_cell_type))
    )
    export_mode = "val" if has_eval_targets else "test"
    if has_eval_targets:
        _log("[INFO] targets available; export/eval dataset will use mode=val for training-aligned semantics")
    else:
        _log("[INFO] targets unavailable; export dataset will use mode=test")

    regions_eval = _resolve_regions(opts, "test" if eval_split == "test" else "val")
    dataset = gba.dataset_input.DataProcessingUnion(
        src_eval,
        opts.data,
        regions_eval,
        opts.comps,
        opts.stain_norm,
        classes,
        gene_names,
        device,
        str(experiment_path),
        False,
        int(args.fold_id),
        mode=export_mode,
        immune_sampler_boost=1.0,
        immune_class_multipliers=None,
    )
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(getattr(opts.training, "batch_size", 64))
    num_workers = int(args.num_workers)
    if num_workers < 0:
        num_workers = int(getattr(opts.data, "num_workers", 0))
    num_workers = max(0, num_workers)
    pin_memory = bool(device_type == "cuda" and getattr(opts.data, "pin_memory", True))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(opts.data, "prefetch_factor", 2))
    dataloader = DataLoader(**loader_kwargs)
    _log(
        f"[INFO] dataloader batches={len(dataloader)} batch_size={batch_size} "
        f"num_workers={num_workers} pin_memory={pin_memory}"
    )

    all_sources = train_sources + test_sources
    expr_ref_torch_map = {}
    avgexp_mode = None
    if use_avgexp:
        if eval_split == "trainval":
            train_regions = getattr(opts, "regions_train", None)
            if train_regions is None:
                train_regions = getattr(opts, "regions_val", None)
            if train_regions is None:
                raise ValueError(
                    "regions_train or regions_val is required for trainval avgexp priors."
                )
            avgexp_df_by_slide = gba.reference_utils.build_train_region_avgexp_df_by_slide(
                train_sources,
                train_regions,
                int(args.fold_id),
                gene_names,
                classes,
                expr_scale,
                holdout_mask_by_slide=None,
                domain_specific=avgexp_domain_specific,
            )
            avgexp_mode = (
                "domain-specific train-region-only"
                if avgexp_domain_specific
                else "train-region-only"
            )
        else:
            avgexp_df_by_slide = gba.reference_utils.build_avgexp_df_by_slide(
                all_sources,
                train_sources,
                gene_names,
                classes,
                expr_scale,
                holdout_mask_by_slide=None,
                domain_specific=avgexp_domain_specific,
            )
            avgexp_mode = (
                "domain-specific source-trainval-only"
                if avgexp_domain_specific
                else "source-trainval-only"
            )
        if not avgexp_df_by_slide:
            raise RuntimeError(f"Failed to compute avgexp priors for mode={avgexp_mode}.")
        _log("[INFO] avgexp ref mode=" + avgexp_mode)

        ref_counts = []
        ref_stack = []
        for slide_id, df_ref_tmp in avgexp_df_by_slide.items():
            df_aligned = df_ref_tmp.reindex(columns=gene_names)
            ref_counts.append(df_aligned.shape[0])
            ref_np = df_aligned.to_numpy(dtype=np.float32, copy=False)
            expr_ref_torch_map[int(slide_id)] = torch.from_numpy(ref_np).float().to(device)
            ref_stack.append(ref_np)

        unique_counts = set(ref_counts)
        if len(unique_counts) != 1:
            raise RuntimeError(
                f"Avgexp references per slide differ: {sorted(unique_counts)}. "
                "All slides must have the same number of refs."
            )
        n_ref = ref_counts[0]
        if n_ref <= 0:
            raise RuntimeError("Avgexp references found but none valid (n_ref <= 0).")

        expr_ref_mean = np.nanmean(np.stack(ref_stack, axis=0), axis=0)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        _log(
            "[INFO] slide-specific ref map built from avgexp mode %s for slides: %s"
            % (avgexp_mode, sorted(expr_ref_torch_map.keys()))
        )
        if int(args.slide_id) in expr_ref_torch_map:
            _log(f"[INFO] using slide-specific avgexp ref for eval slide {int(args.slide_id)}")
        else:
            _log(f"[WARN] eval slide {int(args.slide_id)} missing from ref map; falling back to mean ref")
    else:
        n_ref = None
        expr_ref_torch = None
    _log("[INFO] backbone pretrained loading disabled for inference; checkpoint weights will be used")
    model = _build_model(
        gba,
        opts,
        n_classes,
        n_genes,
        n_ref,
        use_avgexp,
        use_celltype,
        use_neighb,
        device,
    )

    checkpoint_fp = None
    state = None
    last_exc = None
    for cand in checkpoint_candidates:
        try:
            state = torch.load(str(cand), map_location=device)
            checkpoint_fp = cand
            break
        except Exception as exc:
            _log(f"[WARN] skipping unreadable checkpoint: {cand} ({exc})")
            last_exc = exc
    if checkpoint_fp is None or state is None:
        raise RuntimeError("Failed to load any checkpoint candidate.") from last_exc

    _log(f"[INFO] checkpoint selected={checkpoint_fp}")
    sd = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _log(f"[INFO] state_dict loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        _log("[ERROR] checkpoint mismatch detected.")
        if missing:
            _log("[ERROR] missing keys:")
            for k in missing[:50]:
                _log(f"  MISSING {k}")
        if unexpected:
            _log("[ERROR] unexpected keys:")
            for k in unexpected[:50]:
                _log(f"  UNEXPECTED {k}")
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    _log("[INFO] model ready; starting forward inference")

    try:
        import inspect

        fwd_params = inspect.signature(model.forward).parameters
        supports_cell_graph = all(
            k in fwd_params for k in ("coords_cells", "cell_edge_index", "cell_patch_ids")
        )
    except Exception:
        supports_cell_graph = False

    ecrm_cfg = getattr(opts.model, "ecrm", None)
    graph_k = max(
        int(getattr(ecrm_cfg, "graph_k", getattr(ecrm_cfg, "k_target", 8)))
        if ecrm_cfg is not None
        else 8,
        2,
    )
    graph_cross_patch = bool(getattr(ecrm_cfg, "cross_patch", False)) if ecrm_cfg is not None else False
    graph_cross_patch_k = max(
        int(getattr(ecrm_cfg, "cross_patch_k", getattr(ecrm_cfg, "graph_k", graph_k)))
        if ecrm_cfg is not None
        else 1,
        1,
    )
    graph_cross_patch_radius = (
        getattr(ecrm_cfg, "cross_patch_radius", None) if ecrm_cfg is not None else None
    )
    if graph_cross_patch_radius is not None:
        graph_cross_patch_radius = float(graph_cross_patch_radius)
        if graph_cross_patch_radius <= 0:
            graph_cross_patch_radius = None
    slide_coord_map_by_slide = {}
    if supports_cell_graph and getattr(model, "use_ecrm", False) and hasattr(gba, "spatial_utils"):
        for src in all_sources:
            sid = int(getattr(src, "slide_idx", -1))
            if sid in slide_coord_map_by_slide:
                continue
            coord_map = gba.spatial_utils.load_histology_coord_map_from_source(src)
            if coord_map:
                slide_coord_map_by_slide[sid] = coord_map
        _log(f"[INFO] loaded cell-coordinate maps for {len(slide_coord_map_by_slide)} slide(s)")

    metrics = None
    per_gene_summary = None

    all_ids = []
    all_preds = []
    all_ct_probs = []
    n_output_cells = 0
    t_infer_start = time.time()
    n_batches = max(len(dataloader), 1)
    log_every = max(1, int(args.log_every))

    with torch.inference_mode():
        for batch_idx, (
            batch_nuclei,
            _batch_type_patch,
            batch_he_img,
            _batch_expr,
            batch_n_cells,
            _batch_ct,
            patch_ids,
            _batch_expr_mask,
            batch_slide_id,
        ) in enumerate(dataloader):
            batch_nuclei = batch_nuclei.to(device, non_blocking=pin_memory)
            batch_he_img = batch_he_img.to(device, non_blocking=pin_memory)
            batch_n_cells = batch_n_cells.to(device, non_blocking=pin_memory)
            patch_ids = patch_ids.to(device, non_blocking=pin_memory)
            slide_ids_unique = torch.unique(batch_slide_id)
            if slide_ids_unique.numel() != 1:
                raise RuntimeError("Mixed slides in batch; set batch_size=1 for per-slide avgexp.")
            slide_id_val = int(slide_ids_unique.item())
            expr_ref_batch = expr_ref_torch_map.get(slide_id_val, expr_ref_torch)

            model_extra_kwargs = {}
            if supports_cell_graph and getattr(model, "use_ecrm", False):
                coord_map_slide = None
                if isinstance(slide_coord_map_by_slide, dict):
                    coord_map_slide = slide_coord_map_by_slide.get(slide_id_val)
                graph = gba.graph_utils.build_cell_graph(
                    batch_nuclei,
                    patch_ids,
                    k_neighbors=graph_k,
                    coords_batch=None,
                    cell_coord_map=coord_map_slide,
                    cross_patch=bool(graph_cross_patch),
                    cross_patch_k=graph_cross_patch_k,
                    cross_patch_radius=graph_cross_patch_radius,
                )
                model_extra_kwargs = {
                    "coords_cells": graph.coords,
                    "cell_edge_index": graph.edge_index,
                    "cell_patch_ids": graph.patch_index,
                }

            (
                out_cell_type,
                _out_map,
                _batch_ct_pc,
                out_expr,
                _out_expr_immune,
                _out_expr_invasive,
                _out_cell_type_expr,
                _fv_cell_type_expr,
                _out_cell_type_gt_expr,
                _fv_cell_type_gt_expr,
                _batch_expr_pc,
                _comp_estimated,
                _areas,
                patch_ids_pc,
            ) = model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                expr_ref_batch,
                batch_ct=None,
                batch_expr=None,
                patch_ids=patch_ids,
                **model_extra_kwargs,
            )

            if out_expr is None or out_expr.numel() == 0:
                continue

            if patch_ids_pc is None:
                patch_ids_pc = _flatten_patch_ids(patch_ids, batch_n_cells)
            if patch_ids_pc is None:
                continue

            cell_ids = patch_ids_pc.detach().cpu().numpy().astype(np.int64)
            pred = out_expr.detach().cpu().numpy().astype(np.float32)

            valid = cell_ids > 0
            if not np.any(valid):
                continue

            all_ids.append(cell_ids[valid])
            all_preds.append(pred[valid])
            n_output_cells += int(valid.sum())

            if out_cell_type is not None and out_cell_type.numel() > 0:
                ct_probs = torch.softmax(out_cell_type.detach(), dim=1).cpu().numpy().astype(np.float32)
                all_ct_probs.append(ct_probs[valid])

            if batch_idx == 0 or (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t_infer_start
                _log(
                    "[INFO] forward batches=%d/%d elapsed=%.1fs output_cells=%d"
                    % (batch_idx + 1, n_batches, elapsed, n_output_cells)
                )

    if not all_ids:
        raise RuntimeError("No predicted cells were produced for this slide.")

    ids = np.concatenate(all_ids, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    preds = preds[order]

    uniq_ids, idx_start, counts = np.unique(ids, return_index=True, return_counts=True)
    sum_preds = np.add.reduceat(preds, idx_start, axis=0)
    pred_scaled = sum_preds / counts[:, None].astype(np.float32)
    aggregation_mode = "mean"
    mean_ct_probs = None
    ct_pred = None
    ct_conf = None
    if all_ct_probs:
        ct_probs_all = np.concatenate(all_ct_probs, axis=0)[order]
        sum_ct_probs = np.add.reduceat(ct_probs_all, idx_start, axis=0)
        mean_ct_probs = sum_ct_probs / counts[:, None].astype(np.float32)
        ct_pred = mean_ct_probs.argmax(axis=1).astype(np.int64)
        ct_conf = mean_ct_probs.max(axis=1).astype(np.float32)
    _log(
        "[INFO] aggregation complete; unique_cells=%d mode=%s"
        % (uniq_ids.size, aggregation_mode)
    )

    stem = (
        f"{eval_split}_slide{int(args.slide_id)}_train_epoch"
        f"{checkpoint_fp.stem.split('_')[1] if checkpoint_fp.stem.startswith('epoch_') else 'latest'}"
    )

    fp_scaled_csv = out_dir / f"{stem}_pred_expr_scaled.csv"
    df_scaled = pd.DataFrame(pred_scaled, index=uniq_ids, columns=gene_names)
    df_scaled.index.name = "c_id"
    df_scaled.to_csv(fp_scaled_csv)
    _log(f"[INFO] wrote scaled_csv={fp_scaled_csv}")

    fp_counts_csv = None
    if args.save_counts_csv:
        pred_counts = np.expm1(np.maximum(pred_scaled, 0.0) / max(expr_scale, 1e-8)).astype(np.float32)
        fp_counts_csv = out_dir / f"{stem}_pred_expr_counts.csv"
        df_counts = pd.DataFrame(pred_counts, index=uniq_ids, columns=gene_names)
        df_counts.index.name = "c_id"
        df_counts.to_csv(fp_counts_csv)
        _log(f"[INFO] wrote counts_csv={fp_counts_csv}")

    fp_npz = out_dir / f"{stem}_pred_expr_scaled.npz"
    npz_payload = {
        "cell_ids": uniq_ids.astype(np.int64),
        "pred_expr_scaled": pred_scaled.astype(np.float32),
        "gene_names": np.asarray(gene_names, dtype=object),
        "aggregation_mode": np.asarray(aggregation_mode, dtype=object),
        "occurrence_counts": counts.astype(np.int64),
    }
    if ct_pred is not None:
        npz_payload["ct_pred"] = ct_pred
    if ct_conf is not None:
        npz_payload["ct_conf"] = ct_conf
    if mean_ct_probs is not None:
        npz_payload["ct_probs"] = mean_ct_probs.astype(np.float32)
        npz_payload["ct_classes"] = np.asarray(classes, dtype=object)
    np.savez_compressed(fp_npz, **npz_payload)
    _log(f"[INFO] wrote scaled_npz={fp_npz}")

    fp_ct_csv = None
    fp_ct_probs_csv = None
    if ct_pred is not None:
        df_ct = pd.DataFrame(
            {
                "c_id": uniq_ids.astype(np.int64),
                "ct_pred_idx": ct_pred.astype(np.int64),
                "ct_pred_label": [classes[idx] if 0 <= idx < len(classes) else str(idx) for idx in ct_pred.tolist()],
                "ct_conf": ct_conf.astype(np.float32) if ct_conf is not None else np.nan,
            }
        )
        fp_ct_csv = out_dir / f"{stem}_pred_celltype.csv"
        df_ct.to_csv(fp_ct_csv, index=False)
        _log(f"[INFO] wrote celltype_csv={fp_ct_csv}")

        df_ct_probs = pd.DataFrame(mean_ct_probs, index=uniq_ids, columns=classes)
        df_ct_probs.index.name = "c_id"
        fp_ct_probs_csv = out_dir / f"{stem}_pred_celltype_probs.csv"
        df_ct_probs.to_csv(fp_ct_probs_csv)
        _log(f"[INFO] wrote celltype_probs_csv={fp_ct_probs_csv}")

    if has_eval_targets and not args.skip_metrics:
        eval_epoch = None
        if checkpoint_fp.stem.startswith("epoch_"):
            try:
                eval_epoch = int(checkpoint_fp.stem.split("_")[1])
            except Exception:
                eval_epoch = None
        per_gene_dir = out_dir / "per_gene"
        _log("[INFO] computing metrics via train.evaluate_validation()")
        metrics = gba.evaluation_utils.evaluate_validation(
            model,
            dataloader,
            expr_ref_torch,
            device,
            n_classes,
            graph_k=graph_k,
            graph_cross_patch=graph_cross_patch,
            graph_cross_patch_k=graph_cross_patch_k,
            graph_cross_patch_radius=graph_cross_patch_radius,
            slide_coord_map_by_slide=slide_coord_map_by_slide,
            expr_ref_torch_map=expr_ref_torch_map,
            holdout_mask_by_slide=None,
            gene_names=gene_names,
            epoch=eval_epoch,
            per_gene_dir=str(per_gene_dir),
            svg_rank_gene_indices_by_slide=None,
            svg_topk=(20, 50),
        )
        per_gene_summary = _extract_per_gene_summary(metrics)
        _log(
            "[INFO] per-gene pearson summary: "
            "mean={mean:.6f} median={median:.6f} max={max:.6f} min={min:.6f} n_genes={n_genes}".format(
                mean=float(per_gene_summary["mean"]) if per_gene_summary["mean"] is not None else float("nan"),
                median=float(per_gene_summary["median"]) if per_gene_summary["median"] is not None else float("nan"),
                max=float(per_gene_summary["max"]) if per_gene_summary["max"] is not None else float("nan"),
                min=float(per_gene_summary["min"]) if per_gene_summary["min"] is not None else float("nan"),
                n_genes=int(per_gene_summary["n_genes"] or 0),
            )
        )
    elif args.skip_metrics:
        _log("[INFO] metric computation skipped by --skip_metrics")
    else:
        _log("[INFO] metric computation skipped because targets are unavailable")

    fp_meta = out_dir / f"{stem}_meta.json"
    meta = {
        "config_file": str(config_file),
        "experiment_path": str(experiment_path),
        "checkpoint_path": str(checkpoint_fp),
        "trainer_module": str(nature_root / "train.py"),
        "slide_id": int(args.slide_id),
        "domain_id": eval_domain_id,
        "split": eval_split,
        "dataset_mode": export_mode,
        "reference_mode": avgexp_mode,
        "prior_domain_id": eval_domain_id,
        "prior_slides": sorted(int(x) for x in expr_ref_torch_map.keys()),
        "n_cells_output": int(uniq_ids.size),
        "n_patch_cell_predictions": int(n_output_cells),
        "aggregation_mode": aggregation_mode,
        "n_genes": int(n_genes),
        "n_classes": int(n_classes),
        "expr_scale": float(expr_scale),
        "gpu_id": int(args.gpu_id),
        "num_workers": int(num_workers),
        "batch_size": int(batch_size),
        "use_gt_ct_ref_weights": False,
        "metrics": metrics,
        "per_gene_pearson_summary": per_gene_summary,
        "outputs": {
            "pred_expr_scaled_csv": str(fp_scaled_csv),
            "pred_expr_scaled_npz": str(fp_npz),
            "pred_expr_counts_csv": str(fp_counts_csv) if fp_counts_csv else None,
            "pred_celltype_csv": str(fp_ct_csv) if fp_ct_csv else None,
            "pred_celltype_probs_csv": str(fp_ct_probs_csv) if fp_ct_probs_csv else None,
        },
    }
    with open(fp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    _log(f"[INFO] wrote meta={fp_meta}")

    _log(f"[DONE] cells={uniq_ids.size} genes={n_genes} classes={n_classes}")
    _log(f"[DONE] scaled_csv={fp_scaled_csv}")
    _log(f"[DONE] scaled_npz={fp_npz}")
    if fp_counts_csv is not None:
        _log(f"[DONE] counts_csv={fp_counts_csv}")
    if fp_ct_csv is not None:
        _log(f"[DONE] celltype_csv={fp_ct_csv}")
    if fp_ct_probs_csv is not None:
        _log(f"[DONE] celltype_probs_csv={fp_ct_probs_csv}")
    _log(f"[DONE] meta={fp_meta}")


if __name__ == "__main__":
    main()

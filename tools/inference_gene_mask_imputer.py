#!/usr/bin/env python3
"""Inference for the hurdle fixed-10-Giotto-SVG panel-completion run.

This script is intentionally pinned to one completed experiment.  It predicts the
ten predefined masked genes for the five validation slides, using observed genes,
morphology, and the training-region leave-one-slide-out prior.  A ten-check
preflight is mandatory and cannot be skipped.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get(
        "GHIST_GENE_MASK_ARTIFACT_ROOT",
        str(REPO_ROOT / "local_artifacts" / "gene_mask_imputer_hurdle_figure3"),
    )
).expanduser().resolve()
RUN_DIR = (
    ARTIFACT_ROOT
    / "results"
    / "fold1_gene_mask_imputer_hurdle_figure3_30ep"
)
CONFIG_PATH = RUN_DIR / "config_gene_mask_imputer.json"
GENES_PATH = RUN_DIR / "genes.txt"
MANIFEST_PATH = RUN_DIR / "holdout_genes_by_slide.json"
IMPUTED_CACHE_DIR = (
    ARTIFACT_ROOT
    / "cache"
    / "imputed_trainregionfb_f1_33f50e3f_topgiotto10_defb72b6"
)
DATASET_CACHE_ROOT = ARTIFACT_ROOT / "cache"
OUTPUT_ROOT = ARTIFACT_ROOT / "diagnostics" / "inference_gene_mask_imputer"

EXPECTED_SLIDES = (0, 1, 2, 4, 5)
EXPECTED_N_GENES = 413
EXPECTED_N_TARGETS = 10
EXPECTED_MANIFEST_HASH = "defb72b6"
EXPECTED_GENES_SHA256 = (
    "7d1aec6900e34840bcf0c851f66f6e1394477feb9caa9fda84f44a817beca389"
)
EXPECTED_CONFIG_SHA256 = (
    "468c1374aa3984f8a19735b865dc1ac3bb0817b53704ee2590801da032a9eb1c"
)
EXPECTED_MANIFEST_SHA256 = (
    "97866a3b68815b4779d120c2346b9c898a1a41db3911ed8bbf7930f5f4c1ea6d"
)
EXPECTED_STANDARDISATION_SHA256 = (
    "3f172e9742c1ceb7783b1187e9900d8a51f67e2322a85ca441b0a9621dcf5e79"
)
STRICT_BEST_EPOCH = 6
STRICT_BEST_METRIC = "holdout_gene_pooled_median"
STRICT_BEST_VALUE = 0.7169966932559004
FOLD_ID = 1
PARITY_SLIDE = 0
TARGET_PERTURBATION = 1_000_000.0
PRIOR_PERTURBATION = 10_000.0
STRICT_DESIGN_CAVEAT = (
    "Training-region SVG ranking for ECRM read raw cached target columns before "
    "the expression mask was applied. Targets received no loss, but their ranks "
    "could displace observed genes in the top-20/top-50 auxiliary loss."
)
DESIGN_QUALIFICATIONS = (
    "The spatial 20% band is validation because it was evaluated every epoch and "
    "used for checkpoint selection.",
    "The ten targets per slide were predefined from whole-slide Giotto rankings; "
    "claims are limited to those targets.",
    "Training validation PCC was patch-occurrence weighted; unique-cell PCC may "
    "differ because overlapping patches repeat cells.",
    "The export mean-aggregates repeated validation patch occurrences by c_id "
    "and records each cell's occurrence count.",
    STRICT_DESIGN_CAVEAT,
    "The completed run used seed 20260807 for Python, NumPy, PyTorch, and CUDA, "
    "with deterministic cuDNN enabled.",
)
REQUIRED_TESTS = (
    "01_manifest",
    "02_cache_mask",
    "03_checkpoint",
    "04_prior_invariance",
    "05_target_perturbation",
    "06_no_gt_ct",
    "07_one_batch_formula",
    "08_one_slide_parity",
    "09_unique_cell_export",
    "10_scale",
)
PRE_EXPORT_TESTS = set(REQUIRED_TESTS) - {"09_unique_cell_export"}


def _log(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _record_test(results: dict, name: str, passed: bool, details: dict) -> None:
    results[name] = {"passed": bool(passed), **details}
    status = "PASS" if passed else "FAIL"
    _log(f"[PREFLIGHT] {status} {name}")
    if not passed:
        raise RuntimeError(f"Mandatory preflight failed: {name}: {details}")


def _load_run_state(trainer):
    required = [
        CONFIG_PATH,
        GENES_PATH,
        MANIFEST_PATH,
        RUN_DIR / "standardisation_hist_fold_1.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing exact run artifact(s): " + ", ".join(missing))

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    opts = trainer._to_namespace(trainer.utils.json_file_to_pyobj(str(CONFIG_PATH)))
    genes = GENES_PATH.read_text(encoding="utf-8").splitlines()
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)

    if len(genes) != EXPECTED_N_GENES or len(set(genes)) != EXPECTED_N_GENES:
        raise RuntimeError("genes.txt must contain exactly 413 unique genes.")
    if _sha256(GENES_PATH) != EXPECTED_GENES_SHA256:
        raise RuntimeError("genes.txt SHA256 does not match the completed run.")
    if _sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Run-config SHA256 does not match the completed run.")
    if _sha256(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Manifest-file SHA256 does not match the completed run.")
    if (
        _sha256(RUN_DIR / "standardisation_hist_fold_1.npy")
        != EXPECTED_STANDARDISATION_SHA256
    ):
        raise RuntimeError("Standardisation SHA256 does not match the completed run.")

    genes_by_slide = {
        int(slide_id): [str(gene) for gene in target_genes]
        for slide_id, target_genes in manifest_payload.get("genes_by_slide", {}).items()
    }
    canonical_manifest = {
        str(slide_id): genes_by_slide[slide_id]
        for slide_id in sorted(genes_by_slide)
    }
    manifest_hash = hashlib.md5(
        json.dumps(canonical_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    slide_ids_ok = tuple(sorted(genes_by_slide)) == EXPECTED_SLIDES
    target_counts_ok = all(
        len(genes_by_slide[slide_id]) == EXPECTED_N_TARGETS
        and len(set(genes_by_slide[slide_id])) == EXPECTED_N_TARGETS
        for slide_id in EXPECTED_SLIDES
    )
    targets_in_union = all(
        gene in set(genes)
        for target_genes in genes_by_slide.values()
        for gene in target_genes
    )
    manifest_ok = (
        slide_ids_ok
        and target_counts_ok
        and targets_in_union
        and manifest_hash == EXPECTED_MANIFEST_HASH
        and manifest_payload.get("holdout_hash") == EXPECTED_MANIFEST_HASH
    )

    if float(opts.data.expr_scale) != 2.0:
        raise RuntimeError("The exact run must use data.expr_scale=2.0.")
    if int(opts.training.batch_size) != 64:
        raise RuntimeError("The exact run must use training.batch_size=64.")
    if int(opts.training.total_epochs) != 30:
        raise RuntimeError("The exact run must contain 30 training epochs.")
    if int(getattr(opts.training, "holdout_n_genes", -1)) != 0:
        raise RuntimeError("Unexpected training.holdout_n_genes value.")
    if not bool(opts.gene_mask_imputer.enabled):
        raise RuntimeError("gene_mask_imputer must be enabled.")
    if int(opts.gene_mask_imputer.mask_n_genes) != EXPECTED_N_TARGETS:
        raise RuntimeError("gene_mask_imputer.mask_n_genes must be 10.")
    if bool(getattr(opts.gene_mask_imputer, "zero_masked_gene_avgexp", False)):
        raise RuntimeError("The completed run requires LOSO, not zero-filled, priors.")

    return (
        opts,
        config_dict,
        genes,
        genes_by_slide,
        manifest_hash,
        manifest_ok,
    )


def _prepare_options(opts) -> None:
    """Apply the same runtime defaults as train_gene_mask_imputer.py."""
    if not hasattr(opts, "model") or opts.model is None:
        opts.model = SimpleNamespace()
    if not hasattr(opts.model, "ecrm") or opts.model.ecrm is None:
        opts.model.ecrm = SimpleNamespace()

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
    opts.model.ecrm.message_dropout = float(
        getattr(opts.model.ecrm, "message_dropout", 0.05)
    )
    opts.model.ecrm.residual_gate_init = float(
        getattr(opts.model.ecrm, "residual_gate_init", -1.4)
    )

    foundation_cfg = getattr(opts.model, "foundation_model", None)
    if foundation_cfg is None:
        foundation_cfg = SimpleNamespace()
        opts.model.foundation_model = foundation_cfg
    foundation_cfg.pretrained = False

    if float(getattr(opts.model, "avgexp_residual_scale", 0.0)) <= 0.0:
        opts.model.avgexp_residual_scale = 0.1


def _raw_and_cached_sources(trainer, opts, genes, genes_by_slide):
    raw_sources = [
        trainer.imputer_task.to_namespace(copy.deepcopy(source))
        for source in list(opts.data_sources_train_val)
    ]
    raw_by_slide = {int(source.slide_idx): source for source in raw_sources}
    if tuple(sorted(raw_by_slide)) != EXPECTED_SLIDES:
        raise RuntimeError("The config does not contain the exact five train/validation slides.")

    cached_by_slide = {}
    holdout_mask_by_slide = {}
    cache_details = {}
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    for slide_id in EXPECTED_SLIDES:
        raw_source = raw_by_slide[slide_id]
        domain_id = int(getattr(raw_source, "domain_id", 0))
        if domain_id != 0:
            raise RuntimeError("The completed run expects domain_id=0 for every slide.")

        expr_path = IMPUTED_CACHE_DIR / f"trainval_slide{slide_id}_domain{domain_id}_expr.csv"
        mask_path = IMPUTED_CACHE_DIR / f"trainval_slide{slide_id}_domain{domain_id}_mask.npy"
        if not expr_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing imputed cache for slide {slide_id}.")

        cached_header = pd.read_csv(expr_path, nrows=0, index_col=0).columns.tolist()
        if cached_header != genes:
            raise RuntimeError(f"Cached gene order differs from genes.txt for slide {slide_id}.")

        raw_header = pd.read_csv(raw_source.fp_expr, nrows=0, index_col=0).columns.tolist()
        measured = set(raw_header)
        targets = genes_by_slide[slide_id]
        if not set(targets).issubset(measured):
            raise RuntimeError(f"A target gene is not measured on slide {slide_id}.")

        cache_mask = np.load(mask_path)
        expected_mask = np.asarray(
            [float(gene in measured and gene not in set(targets)) for gene in genes],
            dtype=np.float32,
        )
        mask_ok = (
            cache_mask.shape == (EXPECTED_N_GENES,)
            and np.array_equal(cache_mask.astype(np.float32), expected_mask)
        )
        if not mask_ok:
            raise RuntimeError(
                f"Cache mask is not exactly measured-and-not-target for slide {slide_id}."
            )

        target_indices = [gene_to_index[gene] for gene in targets]
        holdout_vector = np.zeros(EXPECTED_N_GENES, dtype=np.float32)
        holdout_vector[target_indices] = 1.0  # 1 means target for LOSO/scoring.
        holdout_mask_by_slide[slide_id] = holdout_vector

        cached_source = copy.deepcopy(raw_source)
        cached_source.fp_expr = str(expr_path)
        cached_source.fp_mask = str(mask_path)
        cached_by_slide[slide_id] = cached_source
        cache_details[str(slide_id)] = {
            "mask_path": str(mask_path),
            "n_measured": int(len(measured)),
            "n_observed": int(expected_mask.sum()),
            "n_zero": int((expected_mask == 0).sum()),
        }

    return raw_sources, raw_by_slide, cached_by_slide, holdout_mask_by_slide, cache_details


def _expected_dataset_cache(opts, source, genes) -> Path:
    divisions = opts.regions_val.divisions[FOLD_ID - 1]
    meta = {
        "fp_hist": source.fp_hist,
        "fp_nuc_seg": source.fp_nuc_seg,
        "fp_expr": source.fp_expr,
        "fp_cell_type": source.fp_cell_type,
        "fp_nuc_sizes": source.fp_nuc_sizes,
        "mode": "val",
        "fold_id": FOLD_ID,
        "hsize": opts.data.hsize,
        "wsize": opts.data.wsize,
        "overlap": opts.data.overlap,
        "divisions": divisions,
        "gene_names": genes,
        "hist_mtime": os.path.getmtime(source.fp_hist),
        "nuc_mtime": os.path.getmtime(source.fp_nuc_seg),
        "expr_mtime": os.path.getmtime(source.fp_expr),
        "celltype_mtime": os.path.getmtime(source.fp_cell_type),
        "nuc_sizes_mtime": os.path.getmtime(source.fp_nuc_sizes),
    }
    digest = hashlib.md5(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    return DATASET_CACHE_ROOT / "cache" / f"dataset_{digest}.pt"


def _build_references(trainer, opts, raw_sources, genes, classes, holdout_masks):
    train_regions = getattr(opts, "regions_train", None)
    if train_regions is None:
        train_regions = getattr(opts, "regions_val", None)
    if train_regions is None:
        raise RuntimeError("Neither regions_train nor regions_val is configured.")

    domain_specific = bool(getattr(opts.model, "avgexp_domain_specific", False))
    if domain_specific:
        raise RuntimeError("The completed run expects global (not domain-specific) priors.")

    refs = trainer.reference_utils.build_train_region_avgexp_df_by_slide(
        raw_sources,
        train_regions,
        FOLD_ID,
        genes,
        classes,
        float(opts.data.expr_scale),
        holdout_mask_by_slide=holdout_masks,
        domain_specific=False,
        holdout_fill_strategy="leave_one_slide_out",
    )
    if tuple(sorted(int(slide_id) for slide_id in refs)) != EXPECTED_SLIDES:
        raise RuntimeError("LOSO references were not reconstructed for all five slides.")
    for slide_id in EXPECTED_SLIDES:
        aligned = refs[slide_id].reindex(index=classes, columns=genes)
        if aligned.shape != (len(classes), EXPECTED_N_GENES):
            raise RuntimeError(f"Invalid reference shape for slide {slide_id}.")
        if not np.isfinite(aligned.to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite reference value for slide {slide_id}.")
        refs[slide_id] = aligned
    return refs, train_regions


def _test_prior_invariance(
    trainer,
    raw_sources,
    train_regions,
    genes,
    classes,
    refs,
    genes_by_slide,
    holdout_masks,
):
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    source_paths = {
        slide_id: Path(
            next(source.fp_expr for source in raw_sources if int(source.slide_idx) == slide_id)
        ).resolve()
        for slide_id in EXPECTED_SLIDES
    }
    original_read_csv = trainer.reference_utils.pd.read_csv
    errors = {}

    for slide_id in EXPECTED_SLIDES:
        target_path = source_paths[slide_id]
        target_genes = genes_by_slide[slide_id]

        def read_csv_with_perturbation(path, *args, **kwargs):
            frame = original_read_csv(path, *args, **kwargs)
            if Path(path).resolve() == target_path:
                frame = frame.copy()
                frame.loc[:, target_genes] = (
                    frame.loc[:, target_genes].astype(np.float64)
                    + PRIOR_PERTURBATION
                )
            return frame

        with patch.object(
            trainer.reference_utils.pd,
            "read_csv",
            side_effect=read_csv_with_perturbation,
        ):
            perturbed_refs = trainer.reference_utils.build_train_region_avgexp_df_by_slide(
                raw_sources,
                train_regions,
                FOLD_ID,
                genes,
                classes,
                2.0,
                holdout_mask_by_slide=holdout_masks,
                domain_specific=False,
                holdout_fill_strategy="leave_one_slide_out",
            )

        indices = [gene_to_index[gene] for gene in target_genes]
        baseline = refs[slide_id].to_numpy(dtype=np.float64)[:, indices]
        perturbed = (
            perturbed_refs[slide_id]
            .reindex(index=classes, columns=genes)
            .to_numpy(dtype=np.float64)[:, indices]
        )
        errors[str(slide_id)] = float(np.max(np.abs(baseline - perturbed)))
        del perturbed_refs
        gc.collect()

    max_error = max(errors.values())
    passed = max_error <= 1e-6
    return passed, {
        "slides_checked": list(EXPECTED_SLIDES),
        "perturbation": PRIOR_PERTURBATION,
        "max_abs_error_by_slide": errors,
        "max_abs_error": max_error,
        "tolerance": 1e-6,
    }


def _resolve_checkpoint(args):
    if args.epoch is not None:
        checkpoint = RUN_DIR / "models" / f"epoch_{int(args.epoch)}_model.pth"
    else:
        checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint = checkpoint.resolve()
    model_dir = (RUN_DIR / "models").resolve()
    valid_names = {f"epoch_{epoch}_model.pth": epoch for epoch in range(1, 31)}
    if checkpoint.parent != model_dir or checkpoint.name not in valid_names:
        raise RuntimeError(
            "Checkpoint must be one of epoch_1_model.pth through "
            f"epoch_30_model.pth under {model_dir}."
        )
    return checkpoint, valid_names[checkpoint.name]


def _build_and_load_model(
    trainer,
    opts,
    checkpoint_path,
    requested_epoch,
    n_classes,
    n_genes,
    n_ref,
    device,
):
    _log(f"[INFO] loading checkpoint on CPU: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint must contain model_state_dict.")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch <= 0:
        raise RuntimeError("Checkpoint does not contain a valid epoch.")
    if requested_epoch is not None and checkpoint_epoch != int(requested_epoch):
        raise RuntimeError(
            f"Requested epoch {requested_epoch}, but checkpoint stores epoch {checkpoint_epoch}."
        )

    hurdle_cfg = getattr(opts.model, "hurdle", None)
    if not bool(getattr(hurdle_cfg, "enabled", False)):
        raise RuntimeError("The pinned imputer checkpoint requires model.hurdle.enabled=true.")
    model = trainer.hurdle_framework.HurdleFramework(
        n_classes,
        n_genes,
        int(opts.model.emb_dim),
        device,
        n_ref,
        bool(opts.comps.avgexp),
        bool(opts.comps.celltype),
        bool(opts.comps.neighb) if bool(opts.comps.celltype) else False,
        model_cfg=opts.model,
    )
    model.completion_head = trainer.panel_completion.PanelCompletionHead(
        n_genes,
        hidden_dim=256,
        dropout=0.0,
        use_morph=True,
        morph_gate_init=-2.0,
    )

    state = checkpoint["model_state_dict"]
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = sorted(
        key
        for key in set(expected).intersection(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    completion_keys = sorted(key for key in state if key.startswith("completion_head."))
    parity_ok = (
        not missing
        and not unexpected
        and not shape_mismatches
        and len(completion_keys) == 5
    )
    if not parity_ok:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"shape_mismatches={len(shape_mismatches)} completion_keys={len(completion_keys)}"
        )
    model.load_state_dict(state, strict=True)

    total_epochs = int(opts.training.total_epochs)
    epoch_progress = (checkpoint_epoch - 1) / max(total_epochs - 1, 1)
    model.set_epoch_progress(epoch_progress)
    del state, expected, checkpoint
    gc.collect()

    model.to(device)
    model.eval()
    return model, checkpoint_epoch, epoch_progress, {
        "state_key_count": len(model.state_dict()),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
        "completion_head_keys": completion_keys,
    }


def _make_loader(trainer, opts, source, genes, classes, device, num_workers):
    expected_cache = _expected_dataset_cache(opts, source, genes)
    if not expected_cache.is_file():
        raise RuntimeError(
            "Exact validation dataset cache is missing; refusing to create or modify caches: "
            f"{expected_cache}"
        )

    dataset = trainer.dataset_input.DataProcessingUnion(
        source,
        opts.data,
        opts.regions_val,
        opts.comps,
        opts.stain_norm,
        classes,
        genes,
        device,
        str(RUN_DIR),
        False,
        FOLD_ID,
        mode="val",
        immune_sampler_boost=1.0,
        immune_class_multipliers=None,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": int(opts.training.batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "drop_last": False,
        "pin_memory": bool(getattr(opts.data, "pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        prefetch_factor = getattr(opts.data, "prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs), expected_cache


def _graph_settings(opts):
    ecrm = opts.model.ecrm
    graph_k = max(int(getattr(ecrm, "graph_k", getattr(ecrm, "k_target", 8))), 2)
    cross_patch = bool(getattr(ecrm, "cross_patch", False))
    cross_patch_k = max(
        int(getattr(ecrm, "cross_patch_k", getattr(ecrm, "graph_k", graph_k))), 1
    )
    radius = getattr(ecrm, "cross_patch_radius", None)
    if radius is not None:
        radius = float(radius)
        if radius <= 0:
            radius = None
    return graph_k, cross_patch, cross_patch_k, radius


def _flatten_cell_ids(patch_ids, n_cells):
    values = []
    n_cells_flat = n_cells.view(-1)
    for batch_index in range(patch_ids.shape[0]):
        n_valid = int(n_cells_flat[batch_index].item())
        if n_valid > 0:
            values.append(patch_ids[batch_index, :n_valid])
    return torch.cat(values, dim=0) if values else None


def _complete_panel(model, expr_cached, expr_mask, out_expr, ref_base):
    mask_obs = (expr_mask > 0.5).float()
    expr_observed = mask_obs * expr_cached + (1.0 - mask_obs) * ref_base
    delta_obs = (expr_observed - ref_base) * mask_obs
    delta_morph = out_expr - ref_base
    delta_hat = model.completion_head(delta_obs, mask_obs, delta_morph)
    completed = torch.relu(ref_base + delta_hat)
    return mask_obs * expr_observed + (1.0 - mask_obs) * completed


def _training_completion_formula(model, expr_cached, expr_mask, out_expr, ref_base):
    mask_obs = (expr_mask > 0.5).float()
    delta_obs = (expr_cached - ref_base) * mask_obs
    delta_morph = out_expr - ref_base
    delta_hat = model.completion_head(delta_obs, mask_obs, delta_morph)
    completed = F.relu(ref_base + delta_hat)
    return mask_obs * expr_cached + (1.0 - mask_obs) * completed


def _collect_slide(
    trainer,
    model,
    loader,
    slide_id,
    target_indices,
    ref_tensor,
    coord_map,
    graph_cfg,
    device,
    pin_memory,
    log_every,
    preflight_results=None,
):
    graph_k, cross_patch, cross_patch_k, radius = graph_cfg
    all_ids = []
    all_predictions = []
    tested_first_batch = False
    started = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            (
                batch_nuclei,
                _,
                batch_he,
                batch_expr,
                batch_n_cells,
                batch_ct,
                patch_ids,
                batch_expr_mask,
                batch_slide_id,
            ) = batch
            if torch.unique(batch_slide_id).tolist() != [slide_id]:
                raise RuntimeError(f"Mixed or incorrect slide IDs in slide {slide_id} loader.")

            batch_nuclei = batch_nuclei.to(device, non_blocking=pin_memory)
            batch_he = batch_he.to(device, non_blocking=pin_memory)
            batch_expr = batch_expr.to(device, non_blocking=pin_memory)
            batch_expr_mask = batch_expr_mask.to(device, non_blocking=pin_memory)
            batch_n_cells = batch_n_cells.to(device, non_blocking=pin_memory)
            batch_ct = batch_ct.to(device, non_blocking=pin_memory)
            patch_ids = patch_ids.to(device, non_blocking=pin_memory)

            holdout_mask = np.zeros(ref_tensor.shape[1], dtype=np.float32)
            holdout_mask[np.asarray(target_indices, dtype=np.int64)] = 1.0
            batch_expr_for_model = (
                trainer.hurdle_evaluation.sanitize_hurdle_panel_holdout_inputs(
                    batch_expr,
                    batch_n_cells,
                    batch_ct,
                    ref_tensor,
                    holdout_mask,
                )
            )

            graph = trainer.graph_utils.build_cell_graph(
                batch_nuclei,
                patch_ids,
                k_neighbors=graph_k,
                coords_batch=None,
                cell_coord_map=coord_map,
                cross_patch=cross_patch,
                cross_patch_k=cross_patch_k,
                cross_patch_radius=radius,
            )
            outputs = model(
                batch_he,
                batch_nuclei,
                batch_n_cells,
                ref_tensor,
                batch_ct,
                batch_expr_for_model,
                patch_ids=patch_ids,
                coords_cells=graph.coords,
                cell_edge_index=graph.edge_index,
                cell_patch_ids=graph.patch_index,
            )
            out_expr = outputs[3]
            cell_ids_pc = outputs[13]
            if out_expr is None or out_expr.numel() == 0:
                continue

            expr_cached = trainer.tensor_utils.flatten_expr(batch_expr, batch_n_cells)
            expr_mask = trainer.tensor_utils.flatten_expr_mask(
                batch_expr_mask, batch_n_cells
            )
            ref_base = (model.last_aux_losses or {}).get("expr_ref_base")
            if (
                expr_cached is None
                or expr_mask is None
                or ref_base is None
                or expr_cached.shape != out_expr.shape
                or expr_mask.shape != out_expr.shape
                or ref_base.shape != out_expr.shape
            ):
                raise RuntimeError("Completion inputs are absent or misaligned.")

            completed = _complete_panel(
                model, expr_cached, expr_mask, out_expr, ref_base
            )
            if not torch.isfinite(completed).all():
                raise RuntimeError("Non-finite panel-completion prediction.")

            if preflight_results is not None and not tested_first_batch:
                training_formula = _training_completion_formula(
                    model, expr_cached, expr_mask, out_expr, ref_base
                )
                formula_error = float(
                    torch.max(torch.abs(completed - training_formula)).item()
                )
                _record_test(
                    preflight_results,
                    "07_one_batch_formula",
                    formula_error <= 1e-6,
                    {"slide_id": slide_id, "max_abs_error": formula_error, "tolerance": 1e-6},
                )

                if not torch.all(expr_mask[:, target_indices] <= 0.5):
                    raise RuntimeError("Target genes are not masked in the preflight batch.")
                perturbed = expr_cached.clone()
                perturbed[:, target_indices] += TARGET_PERTURBATION
                perturbed_prediction = _complete_panel(
                    model, perturbed, expr_mask, out_expr, ref_base
                )
                bitwise_equal = torch.equal(completed, perturbed_prediction)
                _record_test(
                    preflight_results,
                    "05_target_perturbation",
                    bitwise_equal,
                    {
                        "slide_id": slide_id,
                        "perturbation": TARGET_PERTURBATION,
                        "bitwise_equal": bitwise_equal,
                    },
                )
                tested_first_batch = True

            if cell_ids_pc is None:
                cell_ids_pc = _flatten_cell_ids(patch_ids, batch_n_cells)
            if cell_ids_pc is None or cell_ids_pc.shape[0] != completed.shape[0]:
                raise RuntimeError("Cell IDs do not align with completion predictions.")
            cell_ids = cell_ids_pc.detach().cpu().numpy().astype(np.int64)
            valid = cell_ids > 0
            all_ids.append(cell_ids[valid])
            all_predictions.append(
                completed.detach().cpu().numpy().astype(np.float32, copy=False)[valid]
            )

            if (
                batch_index == 0
                or (batch_index + 1) % max(int(log_every), 1) == 0
                or batch_index + 1 == len(loader)
            ):
                _log(
                    f"[INFO] slide={slide_id} batches={batch_index + 1}/{len(loader)} "
                    f"elapsed={time.time() - started:.1f}s"
                )

    if preflight_results is not None and not tested_first_batch:
        raise RuntimeError("No non-empty batch was available for one-batch preflight tests.")
    if not all_ids:
        raise RuntimeError(f"No predictions were produced for slide {slide_id}.")
    return np.concatenate(all_ids), np.concatenate(all_predictions)


def _aggregate_unique(cell_ids, predictions):
    order = np.argsort(cell_ids, kind="stable")
    ids_sorted = cell_ids[order]
    predictions_sorted = predictions[order]
    unique_ids, starts, occurrence_counts = np.unique(
        ids_sorted, return_index=True, return_counts=True
    )
    sums = np.add.reduceat(predictions_sorted, starts, axis=0)
    means = sums / occurrence_counts[:, None].astype(np.float32)
    return unique_ids, means.astype(np.float32, copy=False), occurrence_counts


def _load_scaled_target_gt(raw_source, target_genes, cell_ids, expr_scale):
    header = pd.read_csv(raw_source.fp_expr, nrows=0).columns.tolist()
    index_column = header[0]
    df = pd.read_csv(
        raw_source.fp_expr,
        usecols=[index_column, *target_genes],
        index_col=index_column,
    )
    try:
        df.index = df.index.astype(np.int64)
    except Exception:
        pass
    missing_ids = np.setdiff1d(np.unique(cell_ids), df.index.to_numpy())
    if missing_ids.size:
        raise RuntimeError(f"Raw GT is missing {missing_ids.size} predicted cell IDs.")
    counts = df.reindex(cell_ids)[target_genes].to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all():
        raise RuntimeError("Raw target GT contains non-finite values.")
    return np.log1p(np.clip(counts, 0.0, None)) * float(expr_scale)


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size <= 1:
        return float("nan")
    sum_x = x.sum()
    sum_y = y.sum()
    numerator = (x * y).sum() - (sum_x * sum_y) / x.size
    denom_x = (x * x).sum() - (sum_x * sum_x) / x.size
    denom_y = (y * y).sum() - (sum_y * sum_y) / y.size
    denominator = np.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
    return float(numerator / max(denominator, 1e-8))


def _score_slide(
    slide_id,
    target_genes,
    target_indices,
    occurrence_cell_ids,
    occurrence_predictions,
    unique_ids,
    unique_predictions,
    raw_source,
    expr_scale,
):
    unique_gt = _load_scaled_target_gt(
        raw_source, target_genes, unique_ids, expr_scale
    )
    patch_positions = np.searchsorted(unique_ids, occurrence_cell_ids)
    if bool((patch_positions >= unique_ids.size).any()) or not np.array_equal(
        unique_ids[patch_positions], occurrence_cell_ids
    ):
        raise RuntimeError("Patch IDs do not map back to the finalized unique-cell IDs.")
    patch_gt = unique_gt[patch_positions]
    rows = []
    for target_offset, (gene, gene_index) in enumerate(zip(target_genes, target_indices)):
        pcc_unique = _pearson(unique_predictions[:, gene_index], unique_gt[:, target_offset])
        pcc_patch = _pearson(
            occurrence_predictions[:, gene_index], patch_gt[:, target_offset]
        )
        if not np.isfinite(pcc_unique) or not np.isfinite(pcc_patch):
            raise RuntimeError(f"Non-finite target PCC for slide {slide_id}, gene {gene}.")
        rows.append(
            {
                "slide_id": slide_id,
                "gene": gene,
                "gene_index": gene_index,
                "pcc_unique_cell": pcc_unique,
                "pcc_patch_occurrence": pcc_patch,
                "n_unique_cells": int(unique_ids.size),
                "n_patch_occurrences": int(occurrence_cell_ids.size),
            }
        )
    return rows, unique_gt


def _test_scale(
    loader,
    target_genes,
    unique_ids,
    unique_predictions,
    unique_gt,
    expr_scale,
):
    if float(expr_scale) != 2.0:
        return False, {"expr_scale": float(expr_scale), "expected": 2.0}
    cached_scaled = (
        loader.dataset.df_expr.reindex(unique_ids)[target_genes].to_numpy(dtype=np.float64)
    )
    gt_error = float(np.max(np.abs(unique_gt - cached_scaled)))
    predictions_float64 = unique_predictions.astype(np.float64, copy=False)
    round_trip = (
        np.log1p(np.expm1(np.maximum(predictions_float64, 0.0) / expr_scale))
        * expr_scale
    )
    prediction_error = float(
        np.max(np.abs(round_trip - np.maximum(predictions_float64, 0.0)))
    )
    passed = gt_error <= 1e-6 and prediction_error <= 1e-5
    return passed, {
        "expr_scale": float(expr_scale),
        "formula": "2 * log1p(counts)",
        "inverse_formula": "expm1(max(scaled, 0) / 2)",
        "max_gt_scale_error": gt_error,
        "max_prediction_roundtrip_error": prediction_error,
    }


def _run_training_parity(
    trainer,
    model,
    loader,
    slide_id,
    target_genes,
    score_rows,
    ref_mean,
    ref_map,
    holdout_masks,
    coord_maps,
    graph_cfg,
    genes,
    checkpoint_epoch,
    n_classes,
    device,
    tolerance,
):
    graph_k, cross_patch, cross_patch_k, radius = graph_cfg
    metrics = trainer.hurdle_evaluation.evaluate_hurdle_validation(
        model,
        loader,
        ref_mean,
        device,
        n_classes,
        expr_scale=2.0,
        graph_k=graph_k,
        graph_cross_patch=cross_patch,
        graph_cross_patch_k=cross_patch_k,
        graph_cross_patch_radius=radius,
        slide_coord_map_by_slide=coord_maps,
        expr_ref_torch_map=ref_map,
        holdout_mask_by_slide=holdout_masks,
        panel_completion_enabled=True,
        gene_names=genes,
        epoch=checkpoint_epoch,
        per_gene_dir=None,
        svg_rank_gene_indices_by_slide=None,
        svg_topk=(20, 50),
    )
    model.eval()  # evaluate_validation() restores train mode before returning.
    parity_map_all = metrics.get("holdout_pearson_per_gene", {})
    parity_map = parity_map_all.get(slide_id, parity_map_all.get(str(slide_id), {}))
    ours = {row["gene"]: row["pcc_patch_occurrence"] for row in score_rows}
    errors = {
        gene: abs(float(ours[gene]) - float(parity_map[gene]))
        for gene in target_genes
        if gene in parity_map
    }
    complete = len(errors) == EXPECTED_N_TARGETS
    max_error = max(errors.values()) if errors else float("inf")
    return complete and max_error <= tolerance, {
        "slide_id": slide_id,
        "genes_compared": len(errors),
        "max_abs_pcc_error": max_error,
        "tolerance": tolerance,
    }


def _output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _scaled_to_counts(values):
    counts = np.expm1(
        np.maximum(np.asarray(values, dtype=np.float64), 0.0) / 2.0
    )
    if not np.isfinite(counts).all():
        raise RuntimeError("Scaled-to-count inversion produced a non-finite value.")
    return counts


def _write_slide_outputs(
    out_dir,
    slide_id,
    epoch,
    genes,
    target_genes,
    target_indices,
    unique_ids,
    unique_predictions,
    occurrence_counts,
    score_rows,
    metadata,
    save_completed_panel,
    save_counts,
    overwrite,
):
    prefix = f"slide{slide_id}_epoch{epoch}"
    outputs = {}

    target_path = out_dir / f"{prefix}_imputed10_scaled.csv"
    _output_path(target_path, overwrite)
    target_df = pd.DataFrame(
        unique_predictions[:, target_indices], index=unique_ids, columns=target_genes
    )
    target_df.index.name = "c_id"
    target_df.to_csv(target_path)
    outputs["imputed10_scaled"] = str(target_path)

    occurrence_path = out_dir / f"{prefix}_occurrence_counts.csv"
    _output_path(occurrence_path, overwrite)
    pd.DataFrame(
        {"c_id": unique_ids.astype(np.int64), "n_patch_occurrences": occurrence_counts}
    ).to_csv(occurrence_path, index=False)
    outputs["occurrence_counts"] = str(occurrence_path)

    score_path = out_dir / f"{prefix}_validation_pcc.csv"
    _output_path(score_path, overwrite)
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    outputs["validation_pcc"] = str(score_path)

    if save_completed_panel:
        panel_path = out_dir / f"{prefix}_completed413_scaled.csv"
        _output_path(panel_path, overwrite)
        panel_df = pd.DataFrame(unique_predictions, index=unique_ids, columns=genes)
        panel_df.index.name = "c_id"
        panel_df.to_csv(panel_path)
        outputs["completed413_scaled"] = str(panel_path)

    if save_counts:
        target_counts_path = out_dir / f"{prefix}_imputed10_counts.csv"
        _output_path(target_counts_path, overwrite)
        target_counts = _scaled_to_counts(unique_predictions[:, target_indices])
        target_counts_df = pd.DataFrame(
            target_counts, index=unique_ids, columns=target_genes
        )
        target_counts_df.index.name = "c_id"
        target_counts_df.to_csv(target_counts_path)
        outputs["imputed10_counts"] = str(target_counts_path)

        if save_completed_panel:
            panel_counts_path = out_dir / f"{prefix}_completed413_counts.csv"
            _output_path(panel_counts_path, overwrite)
            panel_counts = _scaled_to_counts(unique_predictions)
            panel_counts_df = pd.DataFrame(
                panel_counts, index=unique_ids, columns=genes
            )
            panel_counts_df.index.name = "c_id"
            panel_counts_df.to_csv(panel_counts_path)
            outputs["completed413_counts"] = str(panel_counts_path)

    meta_path = out_dir / f"{prefix}_metadata.json"
    slide_metadata = {**metadata, "outputs": outputs}
    _write_json(meta_path, slide_metadata, overwrite=overwrite)
    outputs["metadata"] = str(meta_path)
    return outputs


def _verify_unique_export(outputs, expected_occurrences, target_genes):
    target_df = pd.read_csv(outputs["imputed10_scaled"], index_col="c_id")
    occurrence_df = pd.read_csv(outputs["occurrence_counts"])
    passed = (
        target_df.index.is_unique
        and occurrence_df["c_id"].is_unique
        and target_df.columns.tolist() == target_genes
        and bool((occurrence_df["n_patch_occurrences"] > 0).all())
        and np.array_equal(
            np.sort(target_df.index.to_numpy(dtype=np.int64)),
            np.sort(occurrence_df["c_id"].to_numpy(dtype=np.int64)),
        )
        and int(occurrence_df["n_patch_occurrences"].sum()) == int(expected_occurrences)
    )
    return passed, {
        "unique_exported_c_ids": int(target_df.shape[0]),
        "occurrence_count_rows": int(occurrence_df.shape[0]),
        "total_patch_occurrences": int(occurrence_df["n_patch_occurrences"].sum()),
        "aggregation": "mean_by_c_id",
    }


def _runtime_overrides(epoch_progress):
    return {
        "foundation_model.pretrained": False,
        "model.avgexp_residual_scale": 1.0,
        "model.use_gt_ct_ref_weights": False,
        "model.ecrm.use_gt_ct": False,
        "model.epoch_progress": epoch_progress,
        "completion_head": {
            "n_genes": EXPECTED_N_GENES,
            "hidden_dim": 256,
            "dropout": 0.0,
            "use_morph": True,
            "morph_gate_init": -2.0,
        },
    }


def _slide_metadata(
    slide_id,
    epoch,
    checkpoint_path,
    checkpoint_sha256,
    config_dict,
    manifest_hash,
    genes_by_slide,
    cache_source,
    dataset_cache,
    unique_ids,
    occurrence_cell_ids,
    occurrence_counts,
    epoch_progress,
    selection_rule,
):
    return {
        "validation_label": "validation inference (not an independent test)",
        "checkpoint_selection_rule": selection_rule,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": epoch,
        "model_epoch_progress": epoch_progress,
        "runtime_overrides": _runtime_overrides(epoch_progress),
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "config_snapshot": config_dict,
        "genes_path": str(GENES_PATH),
        "genes_sha256": EXPECTED_GENES_SHA256,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_hash": manifest_hash,
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "slide_id": slide_id,
        "target_genes": genes_by_slide[slide_id],
        "n_genes": EXPECTED_N_GENES,
        "n_targets": EXPECTED_N_TARGETS,
        "expr_scale": 2.0,
        "scale_formula": "2 * log1p(counts)",
        "counts_inverse": "expm1(max(scaled, 0) / 2)",
        "prior_mode": "training-region leave-one-slide-out targets",
        "prior_fold": FOLD_ID,
        "prior_domain_specific": False,
        "use_gt_ct_ref_weights": False,
        "ecrm_use_gt_ct": False,
        "strict_design_caveat": STRICT_DESIGN_CAVEAT,
        "design_qualifications": list(DESIGN_QUALIFICATIONS),
        "cached_expression_path": str(cache_source.fp_expr),
        "cached_mask_path": str(cache_source.fp_mask),
        "dataset_cache_path": str(dataset_cache),
        "n_unique_cells": int(unique_ids.size),
        "n_patch_occurrences": int(occurrence_cell_ids.size),
        "aggregation_mode": "mean_by_c_id",
        "occurrence_count_min": int(occurrence_counts.min()),
        "occurrence_count_max": int(occurrence_counts.max()),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exact GHIST+ fixed-10-Giotto-SVG validation inference. "
            "Checkpoint choice is always explicit and user-controlled."
        )
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--epoch", type=int)
    checkpoint_group.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Logical index within the GPUs visible to this process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="-1 uses the exact run setting (32); 0 is useful for debugging.",
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--save_completed_panel", action="store_true")
    parser.add_argument("--save_counts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.epoch is not None and args.epoch <= 0:
        raise ValueError("--epoch must be positive.")
    if args.num_workers < -1:
        raise ValueError("--num_workers must be -1 or non-negative.")
    if not DATASET_CACHE_ROOT.is_dir():
        raise FileNotFoundError(DATASET_CACHE_ROOT)
    os.environ["CACHE_ROOT"] = str(DATASET_CACHE_ROOT)

    sys.path.insert(0, str(REPO_ROOT))
    import train as trainer

    tests = {}
    (
        opts,
        config_dict,
        genes,
        genes_by_slide,
        manifest_hash,
        manifest_ok,
    ) = _load_run_state(trainer)
    _prepare_options(opts)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trained model.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"--gpu_id={args.gpu_id} is outside the {torch.cuda.device_count()} visible GPU(s)."
        )
    device = torch.device(f"cuda:{args.gpu_id}")
    checkpoint_path, path_epoch = _resolve_checkpoint(args)
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else OUTPUT_ROOT
        / (
            f"epoch_{path_epoch}_preflight"
            if args.preflight_only
            else f"epoch_{path_epoch}_unique_cell"
        )
    ).resolve()
    if not _inside(out_dir, OUTPUT_ROOT.resolve()):
        raise RuntimeError(f"Output directory must be under {OUTPUT_ROOT}.")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Use a new directory or --overwrite."
        )

    _record_test(
        tests,
        "01_manifest",
        manifest_ok,
        {
            "slides": list(sorted(genes_by_slide)),
            "targets_per_slide": {
                str(slide_id): len(genes_by_slide[slide_id]) for slide_id in EXPECTED_SLIDES
            },
            "manifest_hash": manifest_hash,
            "expected_hash": EXPECTED_MANIFEST_HASH,
        },
    )

    classes = list(opts.data.cell_types)
    if len(classes) != 9:
        raise RuntimeError("The completed run expects nine cell-type reference classes.")
    (
        raw_sources,
        raw_by_slide,
        cached_by_slide,
        holdout_masks,
        cache_details,
    ) = _raw_and_cached_sources(trainer, opts, genes, genes_by_slide)
    dataset_cache_paths = {
        slide_id: _expected_dataset_cache(opts, cached_by_slide[slide_id], genes)
        for slide_id in EXPECTED_SLIDES
    }
    dataset_caches_ok = all(path.is_file() for path in dataset_cache_paths.values())
    _record_test(
        tests,
        "02_cache_mask",
        dataset_caches_ok,
        {
            "slides_checked": list(EXPECTED_SLIDES),
            "mask_semantics": "1 iff measured in raw slide and not one of its ten targets",
            "cache_details": cache_details,
            "dataset_caches": {
                str(slide_id): str(path) for slide_id, path in dataset_cache_paths.items()
            },
        },
    )

    refs, train_regions = _build_references(
        trainer, opts, raw_sources, genes, classes, holdout_masks
    )
    prior_ok, prior_details = _test_prior_invariance(
        trainer,
        raw_sources,
        train_regions,
        genes,
        classes,
        refs,
        genes_by_slide,
        holdout_masks,
    )
    _record_test(tests, "04_prior_invariance", prior_ok, prior_details)

    checkpoint_sha256 = _sha256(checkpoint_path)
    model, checkpoint_epoch, epoch_progress, checkpoint_details = _build_and_load_model(
        trainer,
        opts,
        checkpoint_path,
        path_epoch,
        len(classes),
        len(genes),
        len(classes),
        device,
    )
    selection_rule = (
        f"strict_best:{STRICT_BEST_METRIC}={STRICT_BEST_VALUE}"
        if checkpoint_epoch == STRICT_BEST_EPOCH
        else "user_selected_explicitly"
    )
    _record_test(tests, "03_checkpoint", True, checkpoint_details)

    no_gt_ct = (
        opts.model.use_gt_ct_ref_weights is False
        and opts.model.ecrm.use_gt_ct is False
        and getattr(model, "use_gt_ct_ref_weights", None) is False
        and getattr(model, "ecrm_use_gt_ct", None) is False
    )
    _record_test(
        tests,
        "06_no_gt_ct",
        no_gt_ct,
        {
            "opts.model.use_gt_ct_ref_weights": bool(opts.model.use_gt_ct_ref_weights),
            "opts.model.ecrm.use_gt_ct": bool(opts.model.ecrm.use_gt_ct),
            "model.use_gt_ct_ref_weights": bool(model.use_gt_ct_ref_weights),
            "model.ecrm_use_gt_ct": bool(model.ecrm_use_gt_ct),
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    num_workers = (
        int(opts.data.num_workers) if int(args.num_workers) < 0 else int(args.num_workers)
    )
    if num_workers < 0:
        raise ValueError("--num_workers cannot be negative except for -1.")

    ref_np_map = {
        slide_id: refs[slide_id].to_numpy(dtype=np.float32, copy=False)
        for slide_id in EXPECTED_SLIDES
    }
    ref_map = {
        slide_id: torch.from_numpy(values).float().to(device)
        for slide_id, values in ref_np_map.items()
    }
    ref_mean = torch.from_numpy(
        np.mean(np.stack(list(ref_np_map.values()), axis=0), axis=0)
    ).float().to(device)
    coord_maps = {
        slide_id: trainer.spatial_utils.load_histology_coord_map_from_source(
            cached_by_slide[slide_id]
        )
        for slide_id in EXPECTED_SLIDES
    }
    if any(not coord_maps[slide_id] for slide_id in EXPECTED_SLIDES):
        raise RuntimeError("A global coordinate map is missing.")
    graph_cfg = _graph_settings(opts)
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    pin_memory = bool(getattr(opts.data, "pin_memory", True))
    parity_tolerance = float(getattr(opts.evaluation, "parity_tolerance_abs", 0.001))

    # One complete slide is finalized first.  The other four remain blocked until
    # every mandatory check, including read-back of this export, has passed.
    target_genes = genes_by_slide[PARITY_SLIDE]
    target_indices = [gene_to_index[gene] for gene in target_genes]
    parity_loader, parity_dataset_cache = _make_loader(
        trainer,
        opts,
        cached_by_slide[PARITY_SLIDE],
        genes,
        classes,
        device,
        num_workers,
    )
    occurrence_cell_ids, occurrence_predictions = _collect_slide(
        trainer,
        model,
        parity_loader,
        PARITY_SLIDE,
        target_indices,
        ref_map[PARITY_SLIDE],
        coord_maps[PARITY_SLIDE],
        graph_cfg,
        device,
        pin_memory,
        args.log_every,
        preflight_results=tests,
    )
    unique_ids, unique_predictions, occurrence_counts = _aggregate_unique(
        occurrence_cell_ids, occurrence_predictions
    )

    # Raw target columns are first selected and used for scoring here, after
    # predictions are detached and unique-cell aggregation is complete.
    parity_rows, parity_unique_gt = _score_slide(
        PARITY_SLIDE,
        target_genes,
        target_indices,
        occurrence_cell_ids,
        occurrence_predictions,
        unique_ids,
        unique_predictions,
        raw_by_slide[PARITY_SLIDE],
        float(opts.data.expr_scale),
    )
    scale_ok, scale_details = _test_scale(
        parity_loader,
        target_genes,
        unique_ids,
        unique_predictions,
        parity_unique_gt,
        float(opts.data.expr_scale),
    )
    _record_test(tests, "10_scale", scale_ok, scale_details)

    parity_ok, parity_details = _run_training_parity(
        trainer,
        model,
        parity_loader,
        PARITY_SLIDE,
        target_genes,
        parity_rows,
        ref_mean,
        ref_map,
        holdout_masks,
        coord_maps,
        graph_cfg,
        genes,
        checkpoint_epoch,
        len(classes),
        device,
        parity_tolerance,
    )
    _record_test(tests, "08_one_slide_parity", parity_ok, parity_details)

    if set(tests) != PRE_EXPORT_TESTS or not all(
        result["passed"] for result in tests.values()
    ):
        raise RuntimeError("Preflight state is incomplete before unique-cell export.")

    parity_metadata = _slide_metadata(
        PARITY_SLIDE,
        checkpoint_epoch,
        checkpoint_path,
        checkpoint_sha256,
        config_dict,
        manifest_hash,
        genes_by_slide,
        cached_by_slide[PARITY_SLIDE],
        parity_dataset_cache,
        unique_ids,
        occurrence_cell_ids,
        occurrence_counts,
        epoch_progress,
        selection_rule,
    )
    parity_outputs = _write_slide_outputs(
        out_dir,
        PARITY_SLIDE,
        checkpoint_epoch,
        genes,
        target_genes,
        target_indices,
        unique_ids,
        unique_predictions,
        occurrence_counts,
        parity_rows,
        parity_metadata,
        args.save_completed_panel,
        args.save_counts,
        args.overwrite,
    )
    unique_ok, unique_details = _verify_unique_export(
        parity_outputs,
        expected_occurrences=occurrence_cell_ids.size,
        target_genes=target_genes,
    )
    _record_test(tests, "09_unique_cell_export", unique_ok, unique_details)

    all_tests_passed = set(tests) == set(REQUIRED_TESTS) and all(
        result["passed"] for result in tests.values()
    )
    if not all_tests_passed:
        raise RuntimeError("The ten-check preflight gate did not pass.")

    preflight_path = out_dir / f"epoch{checkpoint_epoch}_preflight_tests.json"
    _write_json(
        preflight_path,
        {
            "all_passed": True,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "tests": tests,
        },
        overwrite=args.overwrite,
    )
    _log(f"[PREFLIGHT] all ten mandatory checks passed: {preflight_path}")

    score_rows_all = list(parity_rows)
    slide_outputs = {str(PARITY_SLIDE): parity_outputs}
    unique_cells_by_slide = {str(PARITY_SLIDE): int(unique_ids.size)}
    occurrences_by_slide = {
        str(PARITY_SLIDE): int(occurrence_cell_ids.size)
    }

    del (
        parity_loader,
        occurrence_predictions,
        unique_predictions,
        parity_unique_gt,
    )
    gc.collect()
    if args.preflight_only:
        _log("[DONE] preflight-only run complete; remaining four slides were not run.")
        return

    for slide_id in EXPECTED_SLIDES:
        if slide_id == PARITY_SLIDE:
            continue
        target_genes = genes_by_slide[slide_id]
        target_indices = [gene_to_index[gene] for gene in target_genes]
        loader, dataset_cache = _make_loader(
            trainer,
            opts,
            cached_by_slide[slide_id],
            genes,
            classes,
            device,
            num_workers,
        )
        occurrence_cell_ids, occurrence_predictions = _collect_slide(
            trainer,
            model,
            loader,
            slide_id,
            target_indices,
            ref_map[slide_id],
            coord_maps[slide_id],
            graph_cfg,
            device,
            pin_memory,
            args.log_every,
        )
        unique_ids, unique_predictions, occurrence_counts = _aggregate_unique(
            occurrence_cell_ids, occurrence_predictions
        )
        rows, _ = _score_slide(
            slide_id,
            target_genes,
            target_indices,
            occurrence_cell_ids,
            occurrence_predictions,
            unique_ids,
            unique_predictions,
            raw_by_slide[slide_id],
            float(opts.data.expr_scale),
        )
        metadata = _slide_metadata(
            slide_id,
            checkpoint_epoch,
            checkpoint_path,
            checkpoint_sha256,
            config_dict,
            manifest_hash,
            genes_by_slide,
            cached_by_slide[slide_id],
            dataset_cache,
            unique_ids,
            occurrence_cell_ids,
            occurrence_counts,
            epoch_progress,
            selection_rule,
        )
        outputs = _write_slide_outputs(
            out_dir,
            slide_id,
            checkpoint_epoch,
            genes,
            target_genes,
            target_indices,
            unique_ids,
            unique_predictions,
            occurrence_counts,
            rows,
            metadata,
            args.save_completed_panel,
            args.save_counts,
            args.overwrite,
        )
        unique_ok, _ = _verify_unique_export(
            outputs,
            expected_occurrences=occurrence_cell_ids.size,
            target_genes=target_genes,
        )
        if not unique_ok:
            raise RuntimeError(f"Unique-cell output verification failed for slide {slide_id}.")
        score_rows_all.extend(rows)
        slide_outputs[str(slide_id)] = outputs
        unique_cells_by_slide[str(slide_id)] = int(unique_ids.size)
        occurrences_by_slide[str(slide_id)] = int(occurrence_cell_ids.size)
        del loader, occurrence_predictions, unique_predictions
        gc.collect()

    scores = pd.DataFrame(score_rows_all).sort_values(["slide_id", "gene"])
    if scores.shape[0] != 50 or not np.isfinite(scores["pcc_unique_cell"]).all():
        raise RuntimeError("Expected exactly 50 finite slide-gene validation PCC values.")
    scores_path = out_dir / f"epoch{checkpoint_epoch}_validation_pcc_50.csv"
    _output_path(scores_path, args.overwrite)
    scores.to_csv(scores_path, index=False)

    unique_values = scores["pcc_unique_cell"].to_numpy(dtype=np.float64)
    patch_values = scores["pcc_patch_occurrence"].to_numpy(dtype=np.float64)
    summary = {
        "validation_label": "validation inference (not an independent test)",
        "checkpoint_selection_rule": selection_rule,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "runtime_overrides": _runtime_overrides(epoch_progress),
        "manifest_hash": manifest_hash,
        "prior_mode": "training-region leave-one-slide-out targets",
        "expr_scale": 2.0,
        "aggregation_mode": "mean_by_c_id",
        "strict_design_caveat": STRICT_DESIGN_CAVEAT,
        "design_qualifications": list(DESIGN_QUALIFICATIONS),
        "unique_cell_slide_gene_summary": {
            "median": float(np.median(unique_values)),
            "mean": float(np.mean(unique_values)),
            "min": float(np.min(unique_values)),
            "max": float(np.max(unique_values)),
            "n": 50,
        },
        "patch_occurrence_parity_summary": {
            "label": "training-validator parity diagnostic, not a second inference method",
            "median": float(np.median(patch_values)),
            "mean": float(np.mean(patch_values)),
            "min": float(np.min(patch_values)),
            "max": float(np.max(patch_values)),
            "n": 50,
        },
        "unique_cells_by_slide": unique_cells_by_slide,
        "patch_occurrences_by_slide": occurrences_by_slide,
        "preflight_tests": str(preflight_path),
        "per_gene_scores": str(scores_path),
        "slide_outputs": slide_outputs,
    }
    summary_path = out_dir / f"epoch{checkpoint_epoch}_validation_summary.json"
    _write_json(summary_path, summary, overwrite=args.overwrite)
    _log(f"[DONE] five-slide validation inference complete: {summary_path}")


if __name__ == "__main__":
    main()

"""Training entry point and evaluation utilities for GHIST+."""

import argparse
import logging
import os
import sys
import shutil
import hashlib
from types import SimpleNamespace
import inspect
import json
import random
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort

import dataio.dataset_input as dataset_input_base
import dataio.dataset_input_union as dataset_input
import dataio.references as reference_utils
import dataio.samplers as sampler_utils
import dataio.spatial as spatial_utils
import dataio.tensors as tensor_utils
import model.framework as model_framework
import model.hurdle_framework as hurdle_framework
import model.hurdle_distribution as hurdle_distribution
import model.graph as graph_utils
import model.panel_completion as panel_completion
import utils.evaluation as evaluation_utils
import utils.hurdle_evaluation as hurdle_evaluation
import utils.metrics as metric_utils
import utils.checkpoint_selection as checkpoint_selection
import utils.gene_mask_imputer as imputer_task
import utils.tma_select as tma_select
import utils.utils as utils


class _ConciseTrainingLogFilter(logging.Filter):
    """Keep the user-facing stream limited to run essentials."""

    _INFO_PREFIXES = (
        "Using visible GPU(s):",
        "Cell types:",
        "Num cell types:",
        "Train cells:",
        "Val cells:",
        "VAL epoch=",
        "VAL HoldoutSVG",
    )

    def filter(self, record):
        if record.levelno == logging.WARNING:
            return False
        if record.levelno >= logging.ERROR:
            return True
        if record.levelno != logging.INFO:
            return False
        message = record.getMessage()
        return message.startswith(self._INFO_PREFIXES) or (
            message.endswith(" genes (union)")
            and message[: -len(" genes (union)")].isdigit()
        )


class _FatalLogErrorHandler(logging.Handler):
    """Turn explicit ERROR/CRITICAL log records into failed training runs."""

    def __init__(self):
        super().__init__(level=logging.ERROR)

    def emit(self, record):
        raise RuntimeError(f"fatal logged error: {record.getMessage()}")


def _configure_training_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(_ConciseTrainingLogFilter())
    if os.environ.get("GHIST_FATAL_LOG_ERRORS", "0") == "1":
        root_logger.addHandler(_FatalLogErrorHandler())


def _to_namespace(obj):
    if obj is None:
        return None
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, "_asdict"):
            return _to_namespace(obj._asdict())
        return tuple(_to_namespace(v) for v in obj)
    return obj


def _to_serialisable(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_serialisable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _leaf_datasets(dataloader):
    dataset = dataloader.dataset
    return list(dataset.datasets) if isinstance(dataset, torch.utils.data.ConcatDataset) else [dataset]


def _unique_dataset_cell_count(dataset):
    datasets = list(dataset.datasets) if isinstance(dataset, torch.utils.data.ConcatDataset) else [dataset]
    keys = set()
    for item in datasets:
        slide_id = int(getattr(item, "slide_idx", -1))
        keys.update((slide_id, int(cell_id)) for cell_id in item.all_intersect)
    return len(keys)


def _figure3_validation_cell_ids(dataset):
    """Return the exact unique cell IDs emitted by all deterministic VAL patches."""

    emitted = set()
    allowed = set(int(value) for value in dataset.all_intersect)
    for hs, ws in dataset.coords_starts:
        patch = dataset.nuclei[hs : hs + dataset.hsize, ws : ws + dataset.wsize]
        patch_ids = np.unique(patch)
        patch_ids = patch_ids[(patch_ids != 0) & np.isin(patch_ids, list(allowed))]
        if patch_ids.size > int(dataset.max_cells_per_patch):
            patch_ids = patch_ids[: int(dataset.max_cells_per_patch)]
        emitted.update(int(value) for value in patch_ids)
    return emitted


def _hash_array(hasher, array, dtype):
    values = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    hasher.update(str(values.shape).encode("ascii"))
    hasher.update(values.view(np.uint8))


def _build_fixed_figure3_svg_cohort(
    val_dataloader,
    raw_sources,
    gene_names,
    slide_coord_map_by_slide,
    *,
    k_neighbors=8,
):
    """Freeze Figure3's GT/coordinate/actual-VAL cohort before epoch one."""

    source_by_slide = {
        int(getattr(source, "slide_idx", -1)): source for source in raw_sources
    }
    model_gene_index = {str(gene): index for index, gene in enumerate(gene_names)}
    cohorts = {}
    audit_slides = {}
    combined = hashlib.sha256(b"Figure3-fixed-GT-SVG-v1")

    for dataset in _leaf_datasets(val_dataloader):
        slide_id = int(getattr(dataset, "slide_idx", -1))
        if slide_id in cohorts:
            raise RuntimeError(f"Duplicate validation dataset for slide {slide_id}")
        source = source_by_slide.get(slide_id)
        if source is None:
            raise RuntimeError(f"Missing raw GT source for validation slide {slide_id}")
        raw_path = getattr(source, "fp_expr", None)
        if raw_path is None:
            raise RuntimeError(f"Missing raw GT expression path for validation slide {slide_id}")
        raw = pd.read_csv(raw_path, index_col=0)
        numeric_ids = pd.to_numeric(raw.index, errors="coerce")
        if numeric_ids.isna().any():
            raise RuntimeError(f"Raw GT has nonnumeric cell IDs for validation slide {slide_id}")
        raw.index = numeric_ids.astype(np.int64)
        if raw.index.has_duplicates:
            raise RuntimeError(f"Raw GT has duplicate cell IDs for validation slide {slide_id}")
        raw.columns = raw.columns.astype(str)
        if raw.columns.has_duplicates:
            raise RuntimeError(f"Raw GT has duplicate genes for validation slide {slide_id}")

        emitted = _figure3_validation_cell_ids(dataset)
        cell_ids = np.asarray(
            [int(cell_id) for cell_id in raw.index if int(cell_id) in emitted],
            dtype=np.int64,
        )
        if cell_ids.size != len(emitted):
            missing = sorted(emitted.difference(cell_ids.tolist()))
            raise RuntimeError(
                f"Raw GT is missing {len(missing)} emitted validation cells on slide {slide_id}: {missing[:10]}"
            )
        if cell_ids.size <= int(k_neighbors):
            raise RuntimeError(
                f"Validation slide {slide_id} has {cell_ids.size} cells; Figure3 k={int(k_neighbors)} needs more"
            )

        gt_genes = [gene for gene in raw.columns if gene in model_gene_index]
        if len(gt_genes) < 50:
            raise RuntimeError(
                f"Validation slide {slide_id} has only {len(gt_genes)} shared GT/model genes; Top50 is undefined"
            )
        model_indices = np.asarray(
            [model_gene_index[gene] for gene in gt_genes], dtype=np.int64
        )
        target_raw = raw.loc[cell_ids, gt_genes].to_numpy(np.float64)
        if not np.isfinite(target_raw).all():
            raise RuntimeError(f"Raw GT contains nonfinite values on validation slide {slide_id}")
        if np.any(target_raw < 0):
            raise RuntimeError(f"Raw GT contains negative values on validation slide {slide_id}")
        target_log1p = np.log1p(target_raw)

        coordinate_map = slide_coord_map_by_slide.get(slide_id)
        if coordinate_map is None:
            raise RuntimeError(f"Missing coordinate map for validation slide {slide_id}")
        missing_coordinates = [
            int(cell_id) for cell_id in cell_ids if int(cell_id) not in coordinate_map
        ]
        if missing_coordinates:
            raise RuntimeError(
                f"Coordinates missing for {len(missing_coordinates)} validation cells on slide {slide_id}: "
                f"{missing_coordinates[:10]}"
            )
        coordinates_xy = np.asarray(
            [
                (float(coordinate_map[int(cell_id)][1]), float(coordinate_map[int(cell_id)][0]))
                for cell_id in cell_ids
            ],
            dtype=np.float64,
        )
        if not np.isfinite(coordinates_xy).all():
            raise RuntimeError(f"Nonfinite coordinates on validation slide {slide_id}")

        scores, order = metric_utils.figure3_giotto_scores_and_order(
            target_log1p, coordinates_xy, k=int(k_neighbors)
        )
        slide_hash = hashlib.sha256(b"Figure3-fixed-GT-SVG-slide-v1")
        slide_hash.update(str(slide_id).encode("ascii"))
        _hash_array(slide_hash, cell_ids, "<i8")
        for gene in gt_genes:
            encoded = gene.encode("utf-8")
            slide_hash.update(len(encoded).to_bytes(4, "little"))
            slide_hash.update(encoded)
        _hash_array(slide_hash, target_raw, "<f8")
        _hash_array(slide_hash, target_log1p, "<f8")
        _hash_array(slide_hash, coordinates_xy, "<f8")
        _hash_array(slide_hash, scores, "<f4")
        _hash_array(slide_hash, order, "<i8")
        frozen_sha = slide_hash.hexdigest()
        combined.update(bytes.fromhex(frozen_sha))

        cohorts[slide_id] = {
            "slide_id": slide_id,
            "cell_ids": cell_ids,
            "gene_names_gt_order": gt_genes,
            "model_gene_indices_gt_order": model_indices,
            "target_log1p_gt_order": target_log1p,
            "coordinates_xy": coordinates_xy,
            "giotto_scores": scores,
            "giotto_order_gt_positions": order,
            "frozen_sha256": frozen_sha,
        }
        audit_slides[str(slide_id)] = {
            "frozen_sha256": frozen_sha,
            "raw_gt_source": os.path.realpath(raw_path),
            "cell_count": int(cell_ids.size),
            "cell_ids": cell_ids.tolist(),
            "gene_count": int(len(gt_genes)),
            "genes_gt_order": gt_genes,
            "giotto_scores": scores.astype(float).tolist(),
            "giotto_order_gt_positions": order.tolist(),
            "giotto_ranked_genes": [gt_genes[index] for index in order],
        }

    return cohorts, {
        "protocol": "Figure3.ipynb exact GT-only Giotto SVG ranking",
        "ground_truth_transform": "reject negative/nonfinite raw counts, then numpy.log1p(float64); rank helper casts expression to float32",
        "cohort": "actual deterministic deduplicated VAL IDs in raw-GT row order; raw-GT genes in raw-GT column order",
        "coordinates": "corrected histology map converted from (y,x) to float64 (x,y)",
        "k_neighbors": int(k_neighbors),
        "sampling": None,
        "combined_frozen_sha256": combined.hexdigest(),
        "slides": audit_slides,
    }


def _lock_svg_cohort_manifest(payload, run_metrics_dir):
    """Persist the per-run audit and optionally enforce a shared cross-arm hash."""

    _write_json(os.path.join(run_metrics_dir, "fixed_gt_svg_cohort.json"), payload)
    shared_path = os.environ.get("GHIST_SVG_COHORT_MANIFEST")
    if not shared_path:
        return
    shared_path = os.path.abspath(os.path.expanduser(shared_path))
    os.makedirs(os.path.dirname(shared_path), exist_ok=True)
    temporary = f"{shared_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    try:
        os.link(temporary, shared_path)
    except FileExistsError:
        pass
    finally:
        os.unlink(temporary)
    with open(shared_path, "r", encoding="utf-8") as handle:
        shared = json.load(handle)
    expected = shared.get("combined_frozen_sha256")
    observed = payload.get("combined_frozen_sha256")
    if expected != observed:
        raise RuntimeError(
            "Cross-arm fixed SVG validation cohort mismatch: "
            f"shared={expected} current={observed} manifest={shared_path}"
        )


def _unique_cell_rows_for_epoch(cell_ids, slide_id, seen):
    """Select each valid cell ID once per epoch for calibrated hurdle NLL."""

    if cell_ids is None or not isinstance(cell_ids, torch.Tensor):
        raise RuntimeError("Hurdle unique-cell supervision requires patch cell IDs")
    flat = cell_ids.detach().view(-1).cpu().tolist()
    keep = []
    for cell_id in flat:
        key = (int(slide_id), int(cell_id))
        valid = int(cell_id) > 0 and key not in seen
        keep.append(valid)
        if valid:
            seen.add(key)
    return torch.tensor(keep, dtype=torch.bool, device=cell_ids.device)


class TrainingVariant(str, Enum):
    """Small task switches around the one canonical training pipeline."""

    BASE = "base"
    GENE_MASK_IMPUTER = "gene_mask_imputer"
    TMA_SELECT = "tma_select"


def _coerce_training_variant(value):
    if isinstance(value, TrainingVariant):
        return value
    return TrainingVariant(str(value))


def main(config, variant=TrainingVariant.BASE):
    variant = _coerce_training_variant(variant)
    gene_mask_variant = variant is TrainingVariant.GENE_MASK_IMPUTER
    tma_variant = variant is TrainingVariant.TMA_SELECT
    opts = _to_namespace(utils.json_file_to_pyobj(config.config_file))
    imputer_cfg_resolved = None
    if gene_mask_variant:
        imputer_cfg_resolved = imputer_task.resolve_config(
            getattr(opts, "gene_mask_imputer", None),
            repo_root=os.path.dirname(os.path.abspath(__file__)),
        )
        if not imputer_cfg_resolved["enabled"]:
            raise ValueError("gene-mask variant requires gene_mask_imputer.enabled=true")
    if tma_variant and hasattr(opts, "data") and opts.data is not None:
        if not hasattr(opts.data, "punch_select_enabled"):
            opts.data.punch_select_enabled = True
        if not hasattr(opts.data, "punch_filter_splits"):
            opts.data.punch_filter_splits = "train"
        if not hasattr(opts.data, "roi_size_um") and not hasattr(
            opts.data, "broadcast_window_um"
        ):
            opts.data.roi_size_um = 1000.0
        if not hasattr(opts.data, "pixel_size_um"):
            opts.data.pixel_size_um = 0.2125
    training_seed = int(getattr(opts.training, "seed", 20260807))
    random.seed(training_seed)
    np.random.seed(training_seed)

    _configure_training_logging()
    # get_device sets CUDA_VISIBLE_DEVICES when the launcher did not. This must
    # happen before any CUDA availability/seeding call to avoid binding GPU 0.
    device = utils.get_device(config.gpu_id)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(False)
    logging.info("Reproducibility seed: %d (Python/NumPy/Torch/CUDA)", training_seed)
    if gene_mask_variant:
        imputer_task.log_config(imputer_cfg_resolved)

    # Runtime configuration
    eval_cfg = _to_namespace(getattr(opts, "evaluation", None)) or SimpleNamespace()

    if not hasattr(opts, "model") or opts.model is None:
        opts.model = SimpleNamespace()
    if not hasattr(opts.model, "ecrm") or opts.model.ecrm is None:
        opts.model.ecrm = SimpleNamespace()

    histology_normalization = (
        dataset_input_base.validate_foundation_model_input_normalization(
            opts.data,
            opts.model,
        )
    )
    logging.info(
        "Foundation-model H&E input normalization: %s",
        histology_normalization,
    )

    hurdle_cfg = _to_namespace(getattr(opts.model, "hurdle", None)) or SimpleNamespace()
    hurdle_enabled = bool(getattr(hurdle_cfg, "enabled", False))
    if gene_mask_variant and not hurdle_enabled:
        raise ValueError("gene_mask_imputer requires model.hurdle.enabled=true")
    uniform_sampler = bool(getattr(opts.training, "uniform_sampler", False))
    if hurdle_enabled and not uniform_sampler:
        raise ValueError(
            "Hurdle training requires training.uniform_sampler=true so q is calibrated "
            "to the unboosted training cohort"
        )
    if uniform_sampler:
        # This disables weighted multi-slide interleave as well as the
        # single-slide WeightedRandomSampler below.
        opts.training.weighted_interleave_slide_batches = False

    if getattr(opts, "strict_method", None) is not None:
        logging.warning(
            "Config field 'strict_method' is deprecated and ignored; leak guards are always enabled."
        )

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
    opts.model.ecrm.message_dropout = float(getattr(opts.model.ecrm, "message_dropout", 0.05))
    opts.model.ecrm.residual_gate_init = float(
        getattr(opts.model.ecrm, "residual_gate_init", -1.4)
    )

    # Output and cache paths
    this_dir = os.path.abspath(os.path.dirname(__file__))
    nature_root = this_dir
    output_root_override = os.environ.get("OUTPUT_ROOT")
    if output_root_override:
        output_root = os.path.abspath(os.path.expanduser(output_root_override))
    else:
        output_root = nature_root
    default_results_dir = os.path.join(output_root, "results")
    default_metrics_dir = os.path.join(output_root, "metrics")
    os.makedirs(default_results_dir, exist_ok=True)
    os.makedirs(default_metrics_dir, exist_ok=True)

    if not hasattr(opts, "experiment_dirs") or opts.experiment_dirs is None:
        opts.experiment_dirs = SimpleNamespace()
    cfg_load_dir = getattr(opts.experiment_dirs, "load_dir", None)
    cfg_metrics_dir = getattr(opts.experiment_dirs, "metrics_dir", None)
    if not getattr(opts.experiment_dirs, "load_dir", None):
        opts.experiment_dirs.load_dir = default_results_dir
    if not getattr(opts.experiment_dirs, "model_dir", None):
        opts.experiment_dirs.model_dir = "models"
    if cfg_load_dir and os.path.abspath(cfg_load_dir) != default_results_dir:
        logging.warning(
            "Ignoring experiment_dirs.load_dir=%s; outputs are pinned to %s",
            cfg_load_dir,
            default_results_dir,
        )
    if cfg_metrics_dir and os.path.abspath(cfg_metrics_dir) != default_metrics_dir:
        logging.warning(
            "Ignoring experiment_dirs.metrics_dir=%s; metrics are pinned to %s",
            cfg_metrics_dir,
            default_metrics_dir,
        )
    opts.experiment_dirs.load_dir = default_results_dir
    opts.experiment_dirs.metrics_dir = default_metrics_dir
    metrics_dir = default_metrics_dir
    os.makedirs(metrics_dir, exist_ok=True)

    cache_root_override = os.environ.get("CACHE_ROOT")
    if cache_root_override:
        cache_root = os.path.abspath(os.path.expanduser(cache_root_override))
    else:
        cache_root = default_results_dir
    os.environ["CACHE_ROOT"] = cache_root
    logging.info(
        "Path policy: forcing outputs under %s and caches under %s",
        output_root,
        cache_root,
    )
    os.makedirs(cache_root, exist_ok=True)

    if config.resume_epoch != 0:
        make_new = False
    else:
        make_new = True

    timestamp = utils.get_experiment_id(make_new, opts.experiment_dirs.load_dir, config.fold_id)
    timestamp_override = os.environ.get("RUN_ID")
    if timestamp_override:
        if os.path.isabs(timestamp_override):
            timestamp = os.path.basename(os.path.normpath(timestamp_override))
            logging.warning(
                "Ignoring absolute RUN_ID=%s; using run name %s under %s",
                timestamp_override,
                timestamp,
                default_results_dir,
            )
        else:
            timestamp = timestamp_override

    if os.path.isabs(timestamp):
        experiment_path_abs = os.path.abspath(timestamp)
        try:
            inside_results = (
                os.path.commonpath([experiment_path_abs, default_results_dir]) == default_results_dir
            )
        except ValueError:
            inside_results = False
        if inside_results:
            experiment_path = experiment_path_abs
        else:
            experiment_path = os.path.join(
                default_results_dir, os.path.basename(os.path.normpath(experiment_path_abs))
            )
            logging.warning(
                "Ignoring absolute experiment path outside %s; using %s",
                default_results_dir,
                experiment_path,
            )
    else:
        experiment_path = os.path.join(default_results_dir, timestamp)

    per_gene_dir = os.path.join(experiment_path, "per_gene_pearson")
    os.makedirs(experiment_path + "/" + opts.experiment_dirs.model_dir, exist_ok=True)
    os.makedirs(per_gene_dir, exist_ok=True)

    shutil.copyfile(
        config.config_file, experiment_path + "/" + os.path.basename(config.config_file)
    )

    run_meta = {
        "config_file": os.path.abspath(config.config_file),
        "fold_id": int(config.fold_id),
        "gpu_id": int(config.gpu_id),
        "resume_epoch": int(config.resume_epoch),
        "experiment_path": os.path.abspath(experiment_path),
        "metrics_dir": metrics_dir,
        "data_policy": {
            "stats_fit_sources": "trainval_only",
            "gene_union_sources": "all_sources",
            "use_gt_ct_ref_weights": False,
            "ecrm_use_gt_ct": False,
        },
        "evaluation": _to_serialisable(eval_cfg),
    }
    if gene_mask_variant:
        run_meta.update(
            {
                "task_name": imputer_task.TASK_NAME,
                "task_description": imputer_task.TASK_DESCRIPTION,
                "entrypoint": "train_gene_mask_imputer.py",
            }
        )
        run_meta["gene_mask_imputer"] = _to_serialisable(imputer_cfg_resolved)
    _write_json(os.path.join(metrics_dir, "run_meta.json"), run_meta)

    # Model and source setup
    logging.info("Initialising model")

    use_avgexp = opts.comps.avgexp
    use_celltype = opts.comps.celltype
    use_neighb = opts.comps.neighb if use_celltype else False
    avgexp_domain_specific = bool(
        getattr(getattr(opts, "model", SimpleNamespace()), "avgexp_domain_specific", False)
    )

    immune_class_indices = []
    immune_label_whitelist = {
        "b",
        "t",
        "plasma",
        "macrophage",
        "myeloid (excluding macrophage)",
        "myeloid",
    }
    if use_celltype:
        classes = opts.data.cell_types
        n_classes = len(classes)
        class_weights_np = None
        immune_class_indices = [
            idx
            for idx, name in enumerate(classes)
            if str(name).strip().lower() in immune_label_whitelist
        ]
        logging.info("Cell types: %s", classes)
        logging.info("Num cell types: %d", n_classes)
    else:
        n_classes = 0
        classes = []
        class_weights_np = None

    def _ensure_list(sources):
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        return [
            _to_namespace(utils.json_file_to_pyobj(src))
            if isinstance(src, str)
            else _to_namespace(src)
            for src in sources
        ]

    sources_trainval = _ensure_list(getattr(opts, "data_sources_train_val", []))
    sources_test = _ensure_list(getattr(opts, "data_sources_test", []))
    all_sources = sources_trainval + sources_test
    stats_sources = sources_trainval
    logging.info(
        "Source policy: fit_sources=trainval_only (fit_sources=%d, impute_targets=%d, test_sources=%d)",
        len(stats_sources),
        len(all_sources),
        len(sources_test),
    )

    # Gene panel and training statistics
    gene_union = set()
    expr_per_source = {}
    for src in all_sources:
        df_expr_tmp = pd.read_csv(src.fp_expr, index_col=0)
        gene_union.update(df_expr_tmp.columns.tolist())
    for src in stats_sources:
        df_expr_tmp = pd.read_csv(src.fp_expr, index_col=0)
        expr_per_source[src.fp_expr] = df_expr_tmp
    gene_names = natsort.natsorted(gene_union)
    excluded_paths = {str(getattr(src, "fp_expr", "")) for src in sources_test}
    overlap = excluded_paths.intersection(set(expr_per_source.keys()))
    assert len(overlap) == 0, "Leakage guard failed: test source used for stats fitting."
    logging.info(
        "Leakage guard active: excluded %d test expression source(s) from union/statistics fit.",
        len(excluded_paths),
    )

    # Panel completion is a named training variant, never an implicit config
    # side path. BASE and TMA therefore execute the identical core trainer.
    holdout_n_genes = int(
        getattr(opts.training, "holdout_n_genes", 20 if gene_mask_variant else 0)
    )
    if holdout_n_genes < 0:
        holdout_n_genes = 0
    if not gene_mask_variant:
        holdout_n_genes = 0

    holdout_n_genes_eval = (
        int(imputer_cfg_resolved["mask_n_genes"])
        if gene_mask_variant else 0
    )
    if holdout_n_genes_eval:
        logging.info("Gene-mask evaluation: %d fixed genes per train/val slide", holdout_n_genes_eval)
    else:
        logging.info("Holdout eval disabled; training uses all measured genes.")

    panel_hide_default = (
        float(imputer_cfg_resolved["random_mask_frac"])
        if gene_mask_variant
        else 0.0
    )
    panel_hide_frac = (
        float(getattr(opts.training, "panel_hide_frac", panel_hide_default))
        if gene_mask_variant else 0.0
    )
    panel_use_natural_missing = (
        bool(getattr(opts.training, "panel_use_natural_missing", False))
        if gene_mask_variant else False
    )
    panel_completion_enabled = bool(
        gene_mask_variant
        and (
            imputer_cfg_resolved["enabled"]
            or (holdout_n_genes > 0)
            or (panel_hide_frac > 0.0)
            or panel_use_natural_missing
        )
    )
    panel_completion_loss_weight = (
        float(
            getattr(
                opts.training,
                "panel_completion_loss_weight",
                1.0 if panel_completion_enabled else 0.0,
            )
        )
        if gene_mask_variant else 0.0
    )
    panel_hidden_dim = 256
    panel_dropout = 0.0
    panel_use_morph = bool(imputer_cfg_resolved["use_morph"]) if gene_mask_variant else False
    panel_detach_morph = False
    panel_copy_observed = (
        bool(imputer_cfg_resolved["copy_observed"])
        if gene_mask_variant else True
    )
    panel_train_on_holdout = False
    panel_hide_in_forward = False
    panel_morph_gate_init = -2.0
    logging.info(
        "Panel completion: enabled=%s natural_missing=%s loss_w=%.3f "
        "hide_frac=%.2f hidden=%d use_morph=%s",
        panel_completion_enabled,
        panel_use_natural_missing,
        panel_completion_loss_weight,
        panel_hide_frac,
        panel_hidden_dim,
        panel_use_morph,
    )

    regions_train = getattr(opts, "regions_train", None)
    train_regions = regions_train if regions_train is not None else getattr(opts, "regions_val", None)
    if train_regions is None:
        raise ValueError(
            "No regions specified for training (expected regions_train or regions_val in config)."
        )

    # Expression baselines and imputation statistics
    gene_means_series, ct_means_fallback = reference_utils.build_train_region_expression_fallbacks(
        sources_trainval,
        train_regions,
        config.fold_id,
        gene_names,
        classes if use_celltype else None,
        expr_per_source=expr_per_source,
    )
    gene_means_series = gene_means_series.reindex(gene_names).fillna(0.0)
    gene_means_vec = gene_means_series.to_numpy()
    logging.info("Built train-region expression fallback means for %d genes", len(gene_names))
    use_expr_baseline = bool(getattr(opts.training, "use_expr_baseline", False))
    if hurdle_enabled and use_expr_baseline:
        logging.info("Hurdle mode bypasses training.use_expr_baseline")
        use_expr_baseline = False
    if use_expr_baseline:
        baseline_torch = torch.from_numpy(gene_means_vec.astype(np.float32)).float().to(device)
        logging.info("Using per-gene baseline for delta training (breast_all)")
    else:
        baseline_torch = None

    holdout_genes_by_slide = {}
    holdout_mask_by_slide = {}
    holdout_hash = "none"
    if gene_mask_variant:
        holdout_genes_by_slide, holdout_mask_by_slide, holdout_hash = (
            imputer_task.prepare_holdout_masks(
                sources_trainval=sources_trainval,
                expr_per_source=expr_per_source,
                gene_names=gene_names,
                mask_n=holdout_n_genes_eval,
                cfg=imputer_cfg_resolved,
                fold_id=config.fold_id,
                experiment_path=experiment_path,
                metrics_dir=metrics_dir,
                spatial_utils=spatial_utils,
            )
        )

    ct_series_map = {}
    if use_celltype:
        for src in all_sources:
            ct_series_tmp = reference_utils.load_ct_series_for_classes(
                getattr(src, "fp_cell_type", None), classes
            )
            if ct_series_tmp is None:
                continue
            ct_series_map[getattr(src, "fp_expr", "")] = ct_series_tmp

    avgexp_df_by_slide = {}
    avgexp_holdout_fill_strategy = (
        imputer_task.avgexp_holdout_fill_strategy(imputer_cfg_resolved)
        if gene_mask_variant else "leave_one_slide_out"
    )
    if use_avgexp and use_celltype and classes:
        avgexp_df_by_slide = reference_utils.build_train_region_avgexp_df_by_slide(
            sources_trainval,
            train_regions,
            config.fold_id,
            gene_names,
            classes,
            float(opts.data.expr_scale),
            holdout_mask_by_slide=holdout_mask_by_slide,
            domain_specific=avgexp_domain_specific,
            holdout_fill_strategy=avgexp_holdout_fill_strategy,
        )
        missing_trainval_refs = [
            int(getattr(src, "slide_idx", -1))
            for src in sources_trainval
            if int(getattr(src, "slide_idx", -1)) not in avgexp_df_by_slide
        ]
        if missing_trainval_refs:
            raise RuntimeError(
                "Train-region avgexp refs missing for trainval slide(s): "
                + ", ".join(map(str, missing_trainval_refs))
            )

        logging.info(
            "Built %strain-region avgexp priors for %d trainval slide(s) "
            "(shape %dx%d)",
            "domain-specific " if avgexp_domain_specific else "global ",
            len(avgexp_df_by_slide),
            len(classes),
            len(gene_names),
        )

    # Imputation cache
    gene_union_hash = hashlib.md5(",".join(gene_names).encode("utf-8")).hexdigest()[:8]
    impute_tag = f"imputed_trainregionfb_f{config.fold_id}_{gene_union_hash}"
    if gene_mask_variant:
        impute_tag += f"_topgiotto{holdout_n_genes_eval}_{holdout_hash}"
    impute_dir = os.path.join(cache_root, impute_tag)
    os.makedirs(impute_dir, exist_ok=True)
    force_reimpute = str(os.environ.get("FORCE_REIMPUTE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    def _impute_and_save(src_obj, kind="trainval"):
        if isinstance(src_obj, SimpleNamespace):
            src = SimpleNamespace(**src_obj.__dict__)
        elif isinstance(src_obj, dict):
            src = SimpleNamespace(**src_obj)
        else:
            src = SimpleNamespace(
                slide_idx=getattr(src_obj, "slide_idx", -1),
                domain_id=getattr(src_obj, "domain_id", 0),
                fp_avgexp=getattr(src_obj, "fp_avgexp", None),
                fp_expr=getattr(src_obj, "fp_expr", None),
                fp_cell_type=getattr(src_obj, "fp_cell_type", None),
                fp_nuc_seg=getattr(src_obj, "fp_nuc_seg", None),
                fp_hist=getattr(src_obj, "fp_hist", None),
                fp_nuc_sizes=getattr(src_obj, "fp_nuc_sizes", None),
            )
        if not hasattr(src, "domain_id"):
            src.domain_id = 0
        slide_id_local = int(getattr(src, "slide_idx", -1))

        expr_out = os.path.join(
            impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_expr.csv"
        )
        mask_out = os.path.join(
            impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_mask.npy"
        )
        can_reuse = (not force_reimpute) and os.path.isfile(expr_out) and os.path.isfile(mask_out)
        if can_reuse:
            src.fp_expr = expr_out
            src.fp_mask = mask_out
            df_ref_cached = (
                avgexp_df_by_slide.get(slide_id_local)
                if use_avgexp and use_celltype
                else None
            )
            logging.info(
                "Reusing imputed cache for slide=%s kind=%s: %s",
                slide_id_local,
                str(kind),
                expr_out,
            )
            return src, df_ref_cached

        src_expr_key = src.fp_expr
        df_expr = pd.read_csv(src_expr_key, index_col=0)
        missing_expr = [g for g in gene_names if g not in df_expr.columns]
        mask_vec = np.ones(len(gene_names), dtype=np.float32)
        for g in missing_expr:
            mask_vec[gene_names.index(g)] = 0.0
        if gene_mask_variant and kind == "trainval":
            for gene in holdout_genes_by_slide.get(slide_id_local, []):
                mask_vec[gene_names.index(gene)] = 0.0
        df_expr = df_expr.reindex(columns=gene_names)
        if use_celltype and ct_means_fallback is not None:
            ct_series = ct_series_map.get(src_expr_key)
            df_expr_np = df_expr.to_numpy(dtype=np.float32)
            missing_mask_np = ~np.isfinite(df_expr_np)
            if ct_series is not None:
                ct_aligned = ct_series.reindex(df_expr.index)
                ct_arr = ct_aligned.to_numpy(dtype=np.float32)
                fill_vals = np.broadcast_to(gene_means_vec, df_expr_np.shape).copy()
                if ct_means_fallback is not None:
                    for ct_val in np.unique(ct_arr[np.isfinite(ct_arr)]):
                        ct_int = int(ct_val)
                        if 0 <= ct_int < ct_means_fallback.shape[0]:
                            rows = ct_arr == ct_int
                            if rows.any():
                                fill_vals[rows] = ct_means_fallback[ct_int]
                df_expr_np[missing_mask_np] = fill_vals[missing_mask_np]
                df_expr = pd.DataFrame(df_expr_np, index=df_expr.index, columns=df_expr.columns)
            else:
                df_expr = df_expr.fillna(gene_means_series).copy()
        else:
            df_expr = df_expr.fillna(gene_means_series).copy()
        present_frac = 1.0 - (mask_vec == 0).mean()
        logging.debug(
            "Slide %s kind %s: %d/%d genes present (%.2f%%)",
            getattr(src, "slide_idx", "na"),
            kind,
            int(mask_vec.sum()),
            len(mask_vec),
            present_frac * 100,
        )
        try:
            df_expr.index = df_expr.index.astype(int)
        except Exception:
            pass
        df_expr.to_csv(expr_out)
        src.fp_expr = expr_out
        np.save(mask_out, mask_vec)
        src.fp_mask = mask_out

        if kind == "trainval":
            holdout_genes = holdout_genes_by_slide.get(slide_id_local)
            if holdout_genes:
                fp_hold = os.path.join(
                    impute_dir, f"{kind}_slide{src.slide_idx}_domain{src.domain_id}_holdout_genes.txt"
                )
                with open(fp_hold, "w") as handle:
                    for g in holdout_genes:
                        handle.write(f"{g}\n")

        df_ref = None
        if use_avgexp and use_celltype:
            df_ref = avgexp_df_by_slide.get(slide_id_local)
        return src, df_ref

    imputed_trainval = []
    imputed_refs = []
    expr_ref_map = {}
    for src in sources_trainval:
        src_out, df_ref = _impute_and_save(src, kind="trainval")
        imputed_trainval.append(src_out)
        if df_ref is not None:
            imputed_refs.append(df_ref)
            expr_ref_map[src_out.slide_idx] = df_ref

    imputed_test = []
    for src in sources_test:
        src_out, df_ref = _impute_and_save(src, kind="test")
        imputed_test.append(src_out)
        if df_ref is not None:
            expr_ref_map[src_out.slide_idx] = df_ref

    train_sources = imputed_trainval
    test_sources = imputed_test

    # Spatial graph support
    ecrm_cfg_for_graph = getattr(getattr(opts, "model", None), "ecrm", None)
    ecrm_cross_patch_enabled = (
        ecrm_cfg_for_graph is not None
        and bool(getattr(ecrm_cfg_for_graph, "enabled", True))
        and bool(getattr(ecrm_cfg_for_graph, "cross_patch", False))
    )
    slide_coord_map_by_slide = {}
    if ecrm_cross_patch_enabled:
        for src in (train_sources + test_sources):
            sid = int(getattr(src, "slide_idx", -1))
            if sid in slide_coord_map_by_slide:
                continue
            cmap = spatial_utils.load_histology_coord_map_from_source(src)
            if cmap:
                slide_coord_map_by_slide[sid] = cmap
        logging.info(
            "Loaded global cell-coordinate maps for %d slide(s).",
            len(slide_coord_map_by_slide),
        )
    if ecrm_cross_patch_enabled and len(slide_coord_map_by_slide) == 0:
        logging.warning(
            "ECRM cross-patch requested but no global coordinate maps were found; "
            "graph will fall back to within-patch connectivity."
        )

    # Reference priors
    expr_ref_torch_map = {}
    if use_avgexp and imputed_refs:
        ref_counts = []
        ref_stack = []
        for slide_id, df_ref_tmp in expr_ref_map.items():
            df_aligned = df_ref_tmp.reindex(columns=gene_names)
            ref_counts.append(df_aligned.shape[0])
            ref_np = df_aligned.to_numpy(dtype=np.float32)
            expr_ref_torch_map[slide_id] = torch.from_numpy(ref_np).float().to(device)
            ref_stack.append(ref_np)

        unique_counts = set(ref_counts)
        if len(unique_counts) != 1:
            raise ValueError(
                f"Avgexp references per slide differ: {unique_counts}. "
                "All slides must have the same number of refs for a shared model."
            )
        n_ref = ref_counts[0]
        if n_ref <= 0:
            raise ValueError("Avgexp references found but none valid (n_ref <= 0).")

        ref_stack_arr = np.stack(ref_stack, axis=0)
        expr_ref_mean = np.nanmean(ref_stack_arr, axis=0)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        logging.info("Using avgexp with %d reference(s) per slide", n_ref)
    elif use_avgexp:
        expr_ref_mean = np.zeros((1, len(gene_names)), dtype=np.float32)
        expr_ref_torch = torch.from_numpy(expr_ref_mean).float().to(device)
        n_ref = 1
        logging.warning("Avgexp enabled but no references loaded; falling back to zeros.")
    else:
        n_ref = None
        expr_ref_torch = None

    n_genes = len(gene_names)
    logging.info("%d genes (union)", n_genes)

    fp_out = os.path.join(experiment_path, "genes.txt")
    with open(fp_out, "w") as f:
        for line in gene_names:
            f.write(f"{line}\n")

    framework_cls = hurdle_framework.HurdleFramework if hurdle_enabled else model_framework.Framework
    framework_name = f"{framework_cls.__module__}.{framework_cls.__name__}"
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
        logging.info("Using %s (with model_cfg)", framework_name)
    except TypeError:
        if hurdle_enabled:
            raise
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
        logging.info("Using %s (no model_cfg)", framework_name)
    if hurdle_enabled:
        logging.info(
            "Hurdle expression policy: one shared absolute-log1p NLL%s; "
            "no zero-aware MSE/PCC/ECRM-edge/residual/variance/immune/invasive expression losses",
            " plus one masked-panel hurdle NLL" if gene_mask_variant else "",
        )

    if panel_completion_enabled:
        model.completion_head = panel_completion.PanelCompletionHead(
            n_genes,
            hidden_dim=panel_hidden_dim,
            dropout=panel_dropout,
            use_morph=panel_use_morph,
            morph_gate_init=panel_morph_gate_init,
        )

    try:
        fwd_params = inspect.signature(model.forward).parameters
        supports_cell_graph = all(
            k in fwd_params for k in ("coords_cells", "cell_edge_index", "cell_patch_ids")
        )
    except Exception:
        supports_cell_graph = False
    if hurdle_enabled:
        # HurdleFramework forwards the full base signature through *args/**kwargs.
        supports_cell_graph = True
    logging.info("Cell-graph support: %s", supports_cell_graph)

    # Datasets and loaders
    logging.info("Preparing data")

    expr_ref_torch_val = expr_ref_torch
    expr_ref_torch_val_map = expr_ref_torch_map
    if use_avgexp and use_celltype and classes:
        logging.info("Validation avgexp refs reuse train-region-only training refs.")

    immune_sampler_boost = (
        1.0 if uniform_sampler else float(getattr(opts.training, "immune_sampler_boost", 1.0))
    )
    if (not uniform_sampler) and immune_sampler_boost <= 1.0 and use_celltype:
        try:
            counts = np.zeros(n_classes, dtype=np.int64)
            ct_to_idx = {name: idx for idx, name in enumerate(classes)}
            for src in train_sources:
                df_ct_counts = pd.read_csv(
                    src.fp_cell_type, index_col="c_id"
                )["ct"].astype(str)
                ct_indices = df_ct_counts.map(lambda x: ct_to_idx.get(x, None))
                counts += np.bincount(
                    np.array([c for c in ct_indices if c is not None], dtype=int),
                    minlength=n_classes,
                )
            immune_counts = (
                counts[immune_class_indices] if immune_class_indices else np.array([])
            )
            if immune_counts.size > 0 and immune_counts.max() > 0:
                max_boost = float(getattr(opts.training, "sampler_weight_cap", 3.0))
                rare_ratio = immune_counts.max() / max(immune_counts.min(), 1)
                immune_sampler_boost = min(max_boost, max(1.0, rare_ratio))
                logging.info(
                    "Auto immune_sampler_boost=%.2f (rare_ratio=%.2f, cap=%.2f)",
                    immune_sampler_boost,
                    rare_ratio,
                    max_boost,
                )
        except Exception as exc:
            logging.warning("Failed to derive immune_sampler_boost automatically: %s", exc)

    train_datasets = []
    for src in train_sources:
        src_ns = src if isinstance(src, SimpleNamespace) else SimpleNamespace(**src)
        ds = dataset_input.DataProcessingUnion(
            src_ns,
            opts.data,
            train_regions,
            opts.comps,
            opts.stain_norm,
            classes,
            gene_names,
            device,
            experiment_path,
            opts.training.stain_aug,
            config.fold_id,
            mode="train",
            immune_sampler_boost=immune_sampler_boost,
            immune_class_multipliers=None,
        )
        train_datasets.append(ds)

    punch_enabled = bool(
        getattr(opts.data, "punch_select_enabled", False)
        or getattr(opts.data, "tma_select_enabled", False)
    )
    if tma_variant and punch_enabled:
        model.to(device)
        selector_graph_k = max(
            int(
                getattr(
                    opts.model.ecrm,
                    "graph_k",
                    getattr(opts.model.ecrm, "k_target", 8),
                )
            ),
            2,
        )
        selected_datasets = []
        for ds in train_datasets:
            cache_path = tma_select.punch_cache_path(
                experiment_path, int(ds.slide_idx)
            )
            try:
                tma_select.preselect_tma_punch_with_vq(
                    model,
                    ds,
                    opts,
                    device,
                    expr_ref_torch,
                    expr_ref_torch_map,
                    classes,
                    graph_k=selector_graph_k,
                    cache_path=cache_path,
                )
            except Exception as exc:
                logging.warning(
                    "[punch] VQ TMA preselection failed for slide=%s: %s",
                    getattr(ds, "slide_idx", "unknown"),
                    exc,
                )
            selected_datasets.append(
                tma_select.apply_cached_punch_filter(
                    ds,
                    opts.data,
                    cache_path,
                    immune_sampler_boost=immune_sampler_boost,
                )
            )
        train_datasets = selected_datasets

    def _auto_immune_multipliers(ds, immune_idx, classes_all):
        if not immune_idx or not hasattr(ds, "df_ct") or "ct" not in ds.df_ct.columns:
            return {}
        try:
            ct_series = ds.df_ct["ct"].astype(int) - 1
            counts = np.bincount(
                ct_series.clip(lower=0), minlength=len(classes_all)
            ).astype(float)
            immune_counts = counts[immune_idx]
            if immune_counts.sum() <= 0:
                return {}
            props = immune_counts / immune_counts.sum()
            target = np.ones_like(props) / len(props)
            beta = float(getattr(opts.training, "sampler_multiplier_beta", 0.5))
            m_min = float(getattr(opts.training, "sampler_multiplier_min", 0.7))
            m_max = float(getattr(opts.training, "sampler_multiplier_max", 1.5))
            ratio = target / np.maximum(props, 1e-8)
            mult = np.power(ratio, beta)
            mult = np.clip(mult, m_min, m_max)
            keys = [idx + 1 for idx in immune_idx]
            multipliers = {k: float(v) for k, v in zip(keys, mult)}
            logging.info(
                "Auto immune multipliers for slide=%s (beta=%.2f, min=%.2f, max=%.2f): %s",
                getattr(ds, "slide_idx", "unknown"),
                beta,
                m_min,
                m_max,
                multipliers,
            )
            return multipliers
        except Exception as exc:
            logging.warning(
                "Failed to derive immune multipliers for slide=%s: %s",
                getattr(ds, "slide_idx", "unknown"),
                exc,
            )
            return {}

    for ds in train_datasets:
        if uniform_sampler:
            logging.info(
                "Uniform sampler: slide=%s auto immune boost/multipliers disabled",
                getattr(ds, "slide_idx", "unknown"),
            )
            continue
        immune_multipliers = _auto_immune_multipliers(
            ds, immune_class_indices, classes
        )
        if immune_multipliers:
            ds.set_immune_sampling_multipliers(immune_multipliers)
        ds.refresh_patch_sampling_weights(immune_sampler_boost)

    if len(train_datasets) == 0:
        raise ValueError("No training slides could be loaded (all skipped).")
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        use_batch_sampler = False
        batches = None
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        batches = sampler_utils.slide_batch_sampler(
            train_datasets,
            opts.training.batch_size,
            opts.training,
            interleave=bool(getattr(opts.training, "interleave_slide_batches", True)),
        )
        use_batch_sampler = True

    sampler = None
    patch_weights_all = None
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        pw_list = []
        for ds in train_dataset.datasets:
            if getattr(ds, "patch_weights", None) is not None:
                pw_list.extend(ds.patch_weights)
        if pw_list:
            patch_weights_all = pw_list
    else:
        if getattr(train_dataset, "patch_weights", None) is not None:
            patch_weights_all = train_dataset.patch_weights

    if patch_weights_all is not None and not use_batch_sampler and not uniform_sampler:
        try:
            weights = torch.as_tensor(patch_weights_all, dtype=torch.double)
            weight_cap = float(getattr(opts.training, "sampler_weight_cap", 3.0))
            if weight_cap > 0:
                weights = torch.clamp(weights, max=weight_cap)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )
            logging.info(
                "Using WeightedRandomSampler with cap %.2f (min %.4f, max %.4f)",
                weight_cap,
                float(weights.min()),
                float(weights.max()),
            )
        except Exception as exc:
            logging.warning("Falling back to shuffle dataloader (sampler init failed): %s", exc)
            sampler = None

    if use_batch_sampler:
        train_loader_kwargs = {
            "batch_sampler": batches,
            "num_workers": opts.data.num_workers,
            "drop_last": False,
            "pin_memory": getattr(opts.data, "pin_memory", False),
        }
    else:
        train_loader_kwargs = {
            "batch_size": opts.training.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "num_workers": opts.data.num_workers,
            "drop_last": not uniform_sampler,
            "pin_memory": getattr(opts.data, "pin_memory", False),
        }
    if train_loader_kwargs["num_workers"] and train_loader_kwargs["num_workers"] > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = getattr(opts.data, "prefetch_factor", 2)

    dataloader = DataLoader(
        dataset=train_dataset,
        **train_loader_kwargs,
    )
    if uniform_sampler:
        if sampler is not None:
            raise RuntimeError("uniform_sampler invariant failed: weighted sampler is active")
        logging.info("Uniform patch sampling active; no replacement weights or auto immune boost")

    n_train_examples = len(dataloader)
    logging.info("Train cells: %d", _unique_dataset_cell_count(train_dataset))
    logging.info("Total number of training batches: %d" % n_train_examples)

    if use_celltype and class_weights_np is None:
        try:
            counts_total = np.zeros(n_classes, dtype=np.int64)
            datasets_iter = (
                train_dataset.datasets
                if isinstance(train_dataset, torch.utils.data.ConcatDataset)
                else [train_dataset]
            )
            for ds in datasets_iter:
                if hasattr(ds, "df_ct") and "ct" in ds.df_ct.columns:
                    ct_series = ds.df_ct["ct"].astype(int) - 1
                    counts = np.bincount(ct_series.clip(lower=0), minlength=n_classes)
                    counts_total += counts
            counts_total = np.maximum(counts_total, 1)
            inv = 1.0 / counts_total
            inv = inv / inv.mean()
            weight_cap = float(getattr(opts.training, "class_weight_cap", 5.0))
            inv = np.clip(inv, 0.0, weight_cap)
            class_weights_np = inv
            logging.info(
                "CT class weights (cap %.2f): %s",
                weight_cap,
                class_weights_np.tolist(),
            )
        except Exception as exc:
            logging.warning("Failed to compute class weights: %s", exc)

    ecrm_cfg = getattr(opts.model, "ecrm", None)
    if ecrm_cfg is None:
        graph_k = 8
        graph_cross_patch = False
        graph_cross_patch_k = 0
        graph_cross_patch_radius = None
    else:
        graph_k = int(getattr(ecrm_cfg, "graph_k", getattr(ecrm_cfg, "k_target", 8)))
        graph_cross_patch = bool(getattr(ecrm_cfg, "cross_patch", False))
        graph_cross_patch_k = int(
            getattr(ecrm_cfg, "cross_patch_k", getattr(ecrm_cfg, "graph_k", graph_k))
        )
        graph_cross_patch_radius = getattr(ecrm_cfg, "cross_patch_radius", None)
        if graph_cross_patch_radius is not None:
            graph_cross_patch_radius = float(graph_cross_patch_radius)
            if graph_cross_patch_radius <= 0:
                graph_cross_patch_radius = None
    graph_k = max(graph_k, 2)
    graph_cross_patch_k = max(graph_cross_patch_k, 1)

    slide_comp_target = None
    if use_celltype:
        try:
            if hasattr(train_dataset, "df_ct") and "ct" in train_dataset.df_ct.columns:
                slide_series = train_dataset.df_ct["ct"].astype(int) - 1
                counts = np.bincount(slide_series, minlength=n_classes)
                counts = np.maximum(counts, 0)
                total = counts.sum()
                if total > 0:
                    comp = counts / total
                    slide_comp_target = (
                        torch.from_numpy(comp).float().to(device)
                    )
                    logging.info(
                        "Slide GT composition: %s",
                        ", ".join(
                            f"{classes[idx]}:{comp[idx]:.4f}"
                            for idx in range(n_classes)
                    ),
                )
        except Exception as exc:
            logging.warning("Failed to read slide composition: %s", exc)

    def _make_eval_loader(src_list, regions, mode_name):
        datasets = []
        for src in src_list:
            src_ns = src if isinstance(src, SimpleNamespace) else SimpleNamespace(**src)
            ds = dataset_input.DataProcessingUnion(
                src_ns,
                opts.data,
                regions,
                opts.comps,
                opts.stain_norm,
                classes,
                gene_names,
                device,
                experiment_path,
                False,
                config.fold_id,
                mode=mode_name,
                immune_sampler_boost=1.0,
                immune_class_multipliers=None,
            )
            if tma_variant:
                ds = tma_select.apply_cached_punch_filter(
                    ds,
                    opts.data,
                    tma_select.punch_cache_path(
                        experiment_path, int(ds.slide_idx)
                    ),
                    immune_sampler_boost=1.0,
                )
            datasets.append(ds)
        if not datasets:
            return None
        if len(datasets) == 1:
            dataset = datasets[0]
            kwargs = {
                "dataset": dataset,
                "batch_size": opts.training.batch_size,
                "shuffle": False,
                "num_workers": opts.data.num_workers,
                "drop_last": False,
                "pin_memory": getattr(opts.data, "pin_memory", False),
            }
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)
            kwargs = {
                "dataset": dataset,
                "num_workers": opts.data.num_workers,
                "batch_sampler": sampler_utils.slide_batch_sampler(
                    datasets,
                    opts.training.batch_size,
                    opts.training,
                    interleave=False,
                ),
                "drop_last": False,
                "pin_memory": getattr(opts.data, "pin_memory", False),
            }
        if opts.data.num_workers and opts.data.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = getattr(opts.data, "prefetch_factor", 2)
        return DataLoader(**kwargs)

    try:
        val_dataloader = _make_eval_loader(train_sources, opts.regions_val, mode_name="val")
    except Exception as exc:
        logging.warning("Validation loader creation failed: %s", exc)
        val_dataloader = None
    if val_dataloader is not None:
        logging.info("Val cells: %d", _unique_dataset_cell_count(val_dataloader.dataset))

    ext_regions = getattr(opts, "regions_test", None) or getattr(opts, "regions_val", None)
    try:
        external_dataloader = _make_eval_loader(test_sources, ext_regions, mode_name="val")
    except Exception as exc:
        logging.warning("External loader creation failed: %s", exc)
        external_dataloader = None

    svg_topk = (20, 50)
    svg_knn_k = 8
    fixed_svg_cohort_by_slide = {}
    fixed_svg_cohort_audit = None
    if hurdle_enabled:
        if val_dataloader is None:
            raise RuntimeError(
                "Hurdle SVG checkpoint metrics require the validation loader"
            )
        fixed_svg_cohort_by_slide, fixed_svg_cohort_audit = (
            _build_fixed_figure3_svg_cohort(
                val_dataloader,
                sources_trainval,
                gene_names,
                slide_coord_map_by_slide,
                k_neighbors=svg_knn_k,
            )
        )
        _lock_svg_cohort_manifest(fixed_svg_cohort_audit, metrics_dir)
        logging.info(
            "Frozen Figure3 VAL SVG cohort: sha256=%s slides=%d kNN=%d sample_cap=none",
            fixed_svg_cohort_audit["combined_frozen_sha256"],
            len(fixed_svg_cohort_by_slide),
            svg_knn_k,
        )

    svg_sample_cap = 3000
    train_svg_rank_indices_by_slide = spatial_utils.compute_svg_rank_gene_indices_by_slide(
        sources_trainval,
        train_regions,
        config.fold_id,
        mode_name="train",
        gene_names=gene_names,
        k_neighbors=svg_knn_k,
        sample_cap=svg_sample_cap,
    )
    val_svg_rank_indices_by_slide = (
        {}
        if hurdle_enabled
        else spatial_utils.compute_svg_rank_gene_indices_by_slide(
            sources_trainval,
            opts.regions_val,
            config.fold_id,
            mode_name="val",
            gene_names=gene_names,
            k_neighbors=svg_knn_k,
            sample_cap=svg_sample_cap,
        )
    )
    ext_svg_rank_indices_by_slide = spatial_utils.compute_svg_rank_gene_indices_by_slide(
        sources_test,
        ext_regions,
        config.fold_id,
        mode_name="val",
        gene_names=gene_names,
        k_neighbors=svg_knn_k,
        sample_cap=svg_sample_cap,
    )
    logging.info(
        "Precomputed Giotto SVG ranks: train_slides=%d val_slides=%d ext_slides=%d topk=%s kNN=%d sample_cap=%d",
        len(train_svg_rank_indices_by_slide),
        len(val_svg_rank_indices_by_slide),
        len(ext_svg_rank_indices_by_slide),
        list(svg_topk),
        svg_knn_k,
        svg_sample_cap,
    )

    # Optimizer, resume, and losses
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training.learning_rate,
        betas=(opts.training.beta1, opts.training.beta2),
        weight_decay=opts.training.weight_decay,
        eps=opts.training.eps,
    )

    if config.resume_epoch != 0:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    if config.resume_epoch != 0:
        logging.info("Resume training")

        load_path = (
            experiment_path
            + "/"
            + opts.experiment_dirs.model_dir
            + "/epoch_%d_model.pth" % (config.resume_epoch)
        )
        checkpoint = torch.load(load_path)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            logging.warning(
                "Strict state_dict load failed (%s); retrying with strict=False",
                exc,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info("Loaded %s", load_path)

        model.to(device)

        try:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/epoch_%d_optim.pth" % (config.resume_epoch)
            )
            optimizer_checkpoint = torch.load(load_path)
            optimizer.load_state_dict(optimizer_checkpoint["optimizer_state_dict"])
            logging.info("Loaded %s", load_path)
        except Exception:
            logging.info("Optimizer state dict not found")

    else:
        model.to(device)

    logging.info("Begin training")

    if class_weights_np is not None:
        class_weights_torch = torch.from_numpy(class_weights_np).float().to(device)
        logging.info("Using class weights (immune boost applied): %s", class_weights_np.tolist())
    else:
        class_weights_torch = None

    loss_map = nn.CrossEntropyLoss(reduction="mean")
    loss_ct_hist = nn.CrossEntropyLoss(
        weight=class_weights_torch, reduction="mean"
    )
    loss_expr_ct = nn.CrossEntropyLoss(
        weight=class_weights_torch, reduction="mean"
    )
    loss_expr_ct_embed = nn.CosineEmbeddingLoss(reduction="mean")
    loss_logits = nn.MSELoss(reduction="mean")

    zero_weight = float(getattr(opts.training, "zero_weight", 0.1))
    zero_threshold = float(getattr(opts.training, "zero_threshold", 0.0))
    expr_ct_embed_loss_weight = float(
        getattr(opts.training, "expr_ct_embed_loss_weight", 1.0)
    )
    logits_loss_weight = float(getattr(opts.training, "logits_loss_weight", 1.0))
    logging.info(
        (
            "Loss setup: expr_ct_embed_w=%.3f "
            "logits_w=%.3f expr_ct_embed_internal=100 "
            "interleave_slide_batches=%s"
        ),
        expr_ct_embed_loss_weight,
        logits_loss_weight,
        str(bool(getattr(opts.training, "interleave_slide_batches", True))),
    )

    def expr_loss_weighted(pred, target, mask=None):
        """
        Expression loss = zero-aware weighted MSE with existing missing-gene masking.
        """
        w_zero = pred.new_tensor(zero_weight)
        w_one = pred.new_tensor(1.0)
        w = torch.where(target > zero_threshold, w_one, w_zero)
        if mask is not None:
            w = w * mask
        mse_num = ((pred - target) ** 2 * w).sum()
        mse_den = w.sum().clamp_min(1e-8)
        loss_mse_val = mse_num / mse_den

        return loss_mse_val

    def masked_mse(pred, target, mask):
        return expr_loss_weighted(pred, target, mask)

    def masked_var(x, mask):
        if mask is None:
            return torch.var(x, unbiased=False)
        m = mask.bool()
        if not m.any():
            return torch.tensor(0.0, device=x.device)
        vals = x[m]
        return torch.var(vals, unbiased=False)

    total_epochs = opts.training.total_epochs
    validation_function = (
        hurdle_evaluation.evaluate_hurdle_validation
        if hurdle_enabled
        else evaluation_utils.evaluate_validation
    )
    validation_extra_kwargs = (
        {"expr_scale": float(opts.data.expr_scale)} if hurdle_enabled else {}
    )
    val_validation_extra_kwargs = dict(validation_extra_kwargs)
    if hurdle_enabled:
        val_validation_extra_kwargs["fixed_svg_cohort_by_slide"] = (
            fixed_svg_cohort_by_slide
        )
    ext_validation_extra_kwargs = dict(validation_extra_kwargs)
    if gene_mask_variant:
        val_validation_extra_kwargs["panel_completion_enabled"] = True
        ext_validation_extra_kwargs["panel_completion_enabled"] = True
    ext_eval_every_epochs = max(1, int(getattr(eval_cfg, "external_every_epochs", 5)))
    ext_eval_final_epoch = bool(getattr(eval_cfg, "external_final_epoch", True))
    grad_clip_norm = float(getattr(opts.training, "grad_clip_norm", 1.0))
    var_ratio_limit = float(getattr(opts.training, "expr_var_ratio_limit", 30.0))
    parity_tolerance_abs = float(getattr(eval_cfg, "parity_tolerance_abs", 0.001))
    best_metric_eps = float(getattr(eval_cfg, "best_metric_eps", 1e-8))
    external_source_tag = str(getattr(eval_cfg, "external_source", "data_sources_test"))
    svg_run_role = str(os.environ.get("GHIST_RUN_ROLE", "FULL")).strip().upper()
    if svg_run_role not in {"FULL", "ABLATION"}:
        raise ValueError("GHIST_RUN_ROLE must be FULL or ABLATION")
    svg_joint_selector_enabled = bool(hurdle_enabled and svg_run_role == "FULL")
    if hurdle_enabled:
        logging.info(
            "SVG checkpoint policy: role=%s selector=%s external_selection=false",
            svg_run_role,
            "joint_val_rank" if svg_joint_selector_enabled else "deferred_to_full",
        )

    epoch_records = []
    best_val_gene_pooled = -float("inf")
    best_val_ct_macro = -float("inf")
    best_epoch = None
    best_ckpt_path = None
    best_selection_metric_name = (
        "holdout_gene_pooled_median"
        if gene_mask_variant and holdout_n_genes_eval > 0
        else "pearson_gene_pooled_mean"
    )
    best_val_selection_metric = -float("inf")
    kpi_csv_path = os.path.join(default_results_dir, "main_kpi_summary.csv")
    epoch_jsonl_path = os.path.join(metrics_dir, "epoch_metrics.jsonl")
    if initial_epoch == 0:
        for fp in (kpi_csv_path, epoch_jsonl_path):
            if os.path.isfile(fp):
                os.remove(fp)
    elif os.path.isfile(epoch_jsonl_path):
        prior_by_epoch = {}
        with open(epoch_jsonl_path, encoding="utf-8") as history_handle:
            for line in history_handle:
                try:
                    row = json.loads(line)
                    row_epoch = int(row["epoch"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if row_epoch <= initial_epoch:
                    prior_by_epoch[row_epoch] = row
        epoch_records = [prior_by_epoch[key] for key in sorted(prior_by_epoch)]
        logging.info(
            "Resume metric history: loaded %d completed epoch record(s)",
            len(epoch_records),
        )
        if not hurdle_enabled:
            for row in epoch_records:
                score = float(row.get("val_pearson_gene_pooled_mean", -float("inf")))
                ct_score = float(row.get("val_ct_accuracy_macro", -float("inf")))
                if score > best_val_gene_pooled and (
                    best_epoch is None or ct_score >= best_val_ct_macro
                ):
                    best_val_gene_pooled = score
                    best_val_ct_macro = ct_score
                    best_epoch = int(row["epoch"])
                    best_ckpt_path = row.get("checkpoint")

    # Training loop
    for epoch in range(initial_epoch, total_epochs):
        logging.debug("Starting epoch %d", epoch + 1)
        model.train()
        if hasattr(model, "set_epoch_progress"):
            total_eps = max(total_epochs - 1, 1)
            model.set_epoch_progress(epoch / total_eps)

        optimizer.param_groups[0]["lr"] = opts.training.learning_rate * (
            1 - epoch / total_epochs
        )

        loss_epoch = 0
        if use_celltype:
            running_pred_counts = torch.zeros(n_classes, device=device)
            running_gt_counts = torch.zeros(n_classes, device=device)

        epoch_class_counts = (
            np.zeros(n_classes, dtype=np.int64) if n_classes > 0 else None
        )
        pbar = tqdm(
            dataloader,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc=f"Train epoch {epoch+1}/{total_epochs}",
            dynamic_ncols=True,
        )
        loss_total = None
        hurdle_seen_cell_ids = set()

        for (
            batch_nuclei,
            batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            patch_ids,
            batch_expr_mask,
            batch_slide_id,
        ) in pbar:
            optimizer.zero_grad()

            batch_nuclei = batch_nuclei.to(device)
            batch_type_patch = batch_type_patch.to(device)
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
            if supports_cell_graph and getattr(model, "use_ecrm", False):
                coord_map_slide = slide_coord_map_by_slide.get(slide_id_val)
                graph = graph_utils.build_cell_graph(
                    batch_nuclei,
                    patch_ids,
                    k_neighbors=graph_k,
                    coords_batch=None,
                    cell_coord_map=coord_map_slide,
                    cross_patch=graph_cross_patch,
                    cross_patch_k=graph_cross_patch_k,
                    cross_patch_radius=graph_cross_patch_radius,
                )
                model_extra_kwargs = {
                    "coords_cells": graph.coords,
                    "cell_edge_index": graph.edge_index,
                    "cell_patch_ids": graph.patch_index,
                }

            batch_expr_mask_pc = tensor_utils.flatten_expr_mask(batch_expr_mask, batch_n_cells)
            expr_true_pc = tensor_utils.flatten_expr(batch_expr, batch_n_cells)

            mask_panel_pc = (
                batch_expr_mask_pc > 0.5
                if batch_expr_mask_pc is not None and batch_expr_mask_pc.numel() > 0
                else None
            )
            natural_missing_pc = (
                (~mask_panel_pc)
                if panel_completion_enabled
                and panel_use_natural_missing
                and mask_panel_pc is not None
                else None
            )
            mask_hide_pc = None
            if (
                panel_completion_enabled
                and panel_completion_loss_weight > 0
                and mask_panel_pc is not None
                and panel_hide_frac > 0
            ):
                mask_hide_pc = (
                    torch.rand_like(batch_expr_mask_pc) < panel_hide_frac
                ) & mask_panel_pc

            holdout_mask_vec = holdout_mask_by_slide.get(slide_id_val)
            batch_expr_for_model = batch_expr
            need_clone = False
            if holdout_mask_vec is not None and np.any(holdout_mask_vec > 0):
                need_clone = True
            if natural_missing_pc is not None and bool(natural_missing_pc.any()):
                need_clone = True
            if mask_hide_pc is not None and panel_hide_in_forward:
                need_clone = True
            if need_clone:
                batch_expr_for_model = batch_expr.clone()

            if holdout_mask_vec is not None and np.any(holdout_mask_vec > 0):
                holdout_idx = torch.from_numpy(
                    np.where(holdout_mask_vec > 0.5)[0].astype(np.int64)
                ).to(device)
                if holdout_idx.numel() > 0:
                    n_ref_local = (
                        expr_ref_batch.shape[0]
                        if expr_ref_batch is not None else 0
                    )
                    pc_off = 0
                    for b in range(batch_expr_for_model.shape[0]):
                        n_valid = int(batch_n_cells[b].item())
                        if n_valid <= 0:
                            continue
                        ct_b = batch_ct[b, :n_valid].long().clamp(min=0)
                        if n_ref_local > 0:
                            ct_b = ct_b.clamp(max=n_ref_local - 1)
                            baseline_all = expr_ref_batch[ct_b]
                            batch_expr_for_model[
                                b, :n_valid, holdout_idx
                            ] = baseline_all[:, holdout_idx]
                            if mask_hide_pc is not None and panel_hide_in_forward:
                                mask_hide_b = mask_hide_pc[
                                    pc_off : pc_off + n_valid
                                ]
                                if mask_hide_b.any():
                                    expr_b = batch_expr_for_model[b, :n_valid, :]
                                    expr_b[mask_hide_b] = baseline_all[mask_hide_b]
                        else:
                            batch_expr_for_model[
                                b, :n_valid, holdout_idx
                            ] = 0.0
                        pc_off += n_valid
            else:
                n_ref_local = (
                    expr_ref_batch.shape[0] if expr_ref_batch is not None else 0
                )
                pc_off = 0
                for b in range(batch_expr_for_model.shape[0]):
                    n_valid = int(batch_n_cells[b].item())
                    if n_valid <= 0:
                        continue
                    mask_forward_b = None
                    if natural_missing_pc is not None:
                        mask_missing_b = natural_missing_pc[
                            pc_off : pc_off + n_valid
                        ]
                        if mask_missing_b.any():
                            mask_forward_b = mask_missing_b
                    if mask_hide_pc is not None and panel_hide_in_forward:
                        mask_hide_b = mask_hide_pc[pc_off : pc_off + n_valid]
                        if mask_hide_b.any():
                            mask_forward_b = (
                                mask_hide_b
                                if mask_forward_b is None
                                else (mask_forward_b | mask_hide_b)
                            )
                    if mask_forward_b is None or not mask_forward_b.any():
                        pc_off += n_valid
                        continue
                    if n_ref_local > 0:
                        ct_b = (
                            batch_ct[b, :n_valid]
                            .long()
                            .clamp(min=0)
                            .clamp(max=n_ref_local - 1)
                        )
                        baseline_all = expr_ref_batch[ct_b]
                        expr_b = batch_expr_for_model[b, :n_valid, :]
                        expr_b[mask_forward_b] = baseline_all[mask_forward_b]
                    pc_off += n_valid

            (
                out_cell_type,
                out_map,
                batch_ct_pc,
                out_expr,
                out_expr_immune,
                out_expr_invasive,
                out_cell_type_expr,
                fv_cell_type_expr,
                out_cell_type_gt_expr,
                fv_cell_type_gt_expr,
                batch_expr_pc,
                comp_estimated,
                _,
                patch_ids_pc,
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

            if batch_ct_pc.shape[0] == 0:
                continue

            if hurdle_enabled:
                unique_rows = _unique_cell_rows_for_epoch(
                    patch_ids_pc, slide_id_val, hurdle_seen_cell_ids
                )
                if unique_rows.numel() != batch_expr_mask_pc.shape[0]:
                    raise RuntimeError("Hurdle cell-ID/mask row mismatch")
                batch_expr_mask_pc = (
                    batch_expr_mask_pc * unique_rows.to(batch_expr_mask_pc.dtype).unsqueeze(1)
                )

            fixed_holdout_t = torch.zeros_like(batch_expr_mask_pc, dtype=torch.bool)
            if gene_mask_variant and holdout_mask_vec is not None:
                fixed_gene_mask = torch.as_tensor(
                    np.asarray(holdout_mask_vec) > 0,
                    dtype=torch.bool,
                    device=device,
                ).view(1, -1)
                fixed_holdout_t = fixed_gene_mask.expand_as(fixed_holdout_t)
                batch_expr_mask_pc = batch_expr_mask_pc * (~fixed_holdout_t).to(
                    batch_expr_mask_pc.dtype
                )

            if (
                epoch_class_counts is not None
                and immune_class_indices
                and batch_ct_pc.numel() > 0
            ):
                class_counts_batch = (
                    torch.bincount(
                        batch_ct_pc.detach().cpu(), minlength=n_classes
                    )
                    .to(torch.int64)
                    .cpu()
                    .numpy()
                )
                epoch_class_counts += class_counts_batch

            aux_main = getattr(model, "last_aux_losses", {}) or {}
            ref_base_main = aux_main.get("expr_ref_base")
            if hurdle_enabled:
                # Absolute model-scale log1p target and signed Eq. 5 location;
                # the helper converts both to unscaled log1p exactly once.
                target_expr_pc = expr_true_pc if gene_mask_variant else batch_expr_pc
                loss_expr_val = hurdle_distribution.hurdle_reconstruction_loss_from_model(
                    model,
                    target_expr_pc,
                    batch_expr_mask_pc,
                    opts.data.expr_scale,
                )
                loss_ecrm_val = torch.tensor(0.0, device=device)
            else:
                graph_residual_delta = aux_main.get("ecrm_graph_residual_delta")
                graph_residual_base = aux_main.get("ecrm_graph_residual_base")
                pred_expr_for_loss = out_expr
                target_expr_for_loss = batch_expr_pc
                if use_expr_baseline and baseline_torch is not None:
                    pred_expr_for_loss = pred_expr_for_loss - baseline_torch
                    target_expr_for_loss = target_expr_for_loss - baseline_torch

                loss_expr_val = masked_mse(
                    pred_expr_for_loss, target_expr_for_loss, batch_expr_mask_pc
                )
                (
                    ecrm_residual_term_val,
                    ecrm_gene_corr_term_val,
                    ecrm_edge_contrast_term_val,
                ) = metric_utils.graph_residual_loss_terms(
                    graph_residual_delta,
                    graph_residual_base,
                    batch_expr_pc,
                    batch_expr_mask_pc,
                    edge_index=model_extra_kwargs.get("cell_edge_index"),
                    svg_gene_order=train_svg_rank_indices_by_slide.get(int(slide_id_val)),
                    zero_threshold=zero_threshold,
                    zero_weight=zero_weight,
                )
                loss_ecrm_val = (
                    ecrm_residual_term_val
                    + ecrm_gene_corr_term_val
                    + ecrm_edge_contrast_term_val
                )
            loss_map_val = loss_map(out_map, batch_type_patch)

            loss_panel_completion_val = torch.tensor(0.0, device=device)
            if (
                panel_completion_enabled
                and panel_completion_loss_weight > 0
                and hasattr(model, "completion_head")
                and model.completion_head is not None
            ):
                occurrence_logits = aux_main.get("hurdle_occurrence_logits")
                if occurrence_logits is None:
                    raise RuntimeError("Gene-mask variant requires hurdle auxiliaries")
                scale = float(opts.data.expr_scale)
                ref_base_model = (
                    ref_base_main
                    if ref_base_main is not None and ref_base_main.shape == expr_true_pc.shape
                    else torch.zeros_like(expr_true_pc)
                )
                mask_hide = (
                    mask_hide_pc
                    if mask_hide_pc is not None
                    else torch.zeros_like(mask_panel_pc, dtype=torch.bool)
                )
                mask_obs = mask_panel_pc & (~mask_hide)
                mask_obs_f = mask_obs.to(expr_true_pc.dtype)
                mask_target = mask_hide.to(expr_true_pc.dtype)
                if panel_use_natural_missing and natural_missing_pc is not None:
                    mask_target = torch.maximum(
                        mask_target,
                        natural_missing_pc.to(expr_true_pc.dtype),
                    )
                if panel_train_on_holdout:
                    mask_target = torch.maximum(
                        mask_target,
                        fixed_holdout_t.to(expr_true_pc.dtype),
                    )

                if bool((mask_target > 0).any()):
                    delta_obs = (expr_true_pc - ref_base_model) * mask_obs_f
                    delta_morph = out_expr - ref_base_model
                    if panel_detach_morph:
                        delta_morph = delta_morph.detach()
                    delta_hat = model.completion_head(
                        delta_obs,
                        mask_obs_f,
                        delta_morph if panel_use_morph else None,
                    )
                    # Preserve the original B reconstruction exactly; only
                    # its masked MSE is replaced by the hurdle likelihood.
                    pred_completed_model = F.relu(ref_base_model + delta_hat)
                    if panel_copy_observed:
                        pred_completed_model = (
                            mask_obs_f * expr_true_pc
                            + (1.0 - mask_obs_f) * pred_completed_model
                        )
                    loss_panel_completion_val = (
                        hurdle_distribution.masked_hurdle_truncated_gaussian_nll(
                            pred_completed_model / scale,
                            occurrence_logits,
                            expr_true_pc / scale,
                            model.hurdle_sigma(),
                            mask_target,
                        )
                    )

            if use_celltype:
                loss_ct_hist_val = loss_ct_hist(out_cell_type.clone(), batch_ct_pc)
                if hurdle_enabled:
                    # The hurdle expression path has exactly one likelihood.
                    # Retain morphology CT/map supervision, but do not let
                    # expression-conditioned auxiliaries reshape mu or q.
                    loss_expr_ct_val = torch.tensor(0.0, device=device)
                    loss_expr_ct_embed_val = torch.tensor(0.0, device=device)
                    loss_logits_val = torch.tensor(0.0, device=device)
                else:
                    loss_expr_ct_val = loss_expr_ct(
                        out_cell_type_expr.clone(), batch_ct_pc
                    )
                    loss_expr_ct_embed_val = 100 * loss_expr_ct_embed(
                        fv_cell_type_expr,
                        fv_cell_type_gt_expr,
                        target=torch.ones(batch_ct_pc.size(0)).to(device),
                    )
                    loss_logits_val = loss_logits(
                        out_cell_type_expr.clone(), out_cell_type_gt_expr.clone()
                    )
            else:
                loss_ct_hist_val = torch.tensor(0.0).to(device)
                loss_expr_ct_val = torch.tensor(0.0).to(device)
                loss_expr_ct_embed_val = torch.tensor(0.0).to(device)
                loss_logits_val = torch.tensor(0.0).to(device)

            if use_neighb:

                def _find_class_idx(candidates):
                    candidates = {c.strip().lower() for c in candidates}
                    for idx, name in enumerate(classes):
                        lname = name.strip().lower()
                        if lname in candidates:
                            return idx
                    raise ValueError(f"No class found matching {candidates}")

                inv_ct_idx = _find_class_idx(["malignant"])

                if immune_class_indices:
                    imm_mask = torch.isin(
                        batch_ct_pc,
                        torch.tensor(immune_class_indices, device=device),
                    )
                    imm_idx = torch.where(imm_mask)[0]
                else:
                    imm_idx = torch.tensor([], device=device, dtype=torch.long)

                if (not hurdle_enabled) and imm_idx.shape[0] > 0:
                    imm_mask_expr = (
                        batch_expr_mask_pc[imm_idx, :] if batch_expr_mask_pc is not None else None
                    )
                    ref_base_immune = aux_main.get("expr_ref_base_immune")
                    pred_imm = out_expr_immune[imm_idx, :]
                    targ_imm = batch_expr_pc[imm_idx, :]
                    if ref_base_immune is not None and ref_base_immune.shape == out_expr_immune.shape:
                        pred_imm = pred_imm - ref_base_immune[imm_idx, :]
                        targ_imm = targ_imm - ref_base_immune[imm_idx, :]
                    if use_expr_baseline and baseline_torch is not None:
                        pred_imm = pred_imm - baseline_torch
                        targ_imm = targ_imm - baseline_torch
                    loss_expr_immune_val = (1 / n_classes) * masked_mse(
                        pred_imm, targ_imm, imm_mask_expr
                    )
                else:
                    loss_expr_immune_val = torch.tensor(0.0).to(device)

                inv_idx = torch.where(batch_ct_pc == inv_ct_idx)[0]
                if (not hurdle_enabled) and inv_idx.shape[0] > 0:
                    inv_mask_expr = (
                        batch_expr_mask_pc[inv_idx, :] if batch_expr_mask_pc is not None else None
                    )
                    ref_base_inv = aux_main.get("expr_ref_base_invasive")
                    pred_inv = out_expr_invasive[inv_idx, :]
                    targ_inv = batch_expr_pc[inv_idx, :]
                    if ref_base_inv is not None and ref_base_inv.shape == out_expr_invasive.shape:
                        pred_inv = pred_inv - ref_base_inv[inv_idx, :]
                        targ_inv = targ_inv - ref_base_inv[inv_idx, :]
                    if use_expr_baseline and baseline_torch is not None:
                        pred_inv = pred_inv - baseline_torch
                        targ_inv = targ_inv - baseline_torch
                    loss_expr_invasive_val = (1 / n_classes) * masked_mse(
                        pred_inv, targ_inv, inv_mask_expr
                    )
                else:
                    loss_expr_invasive_val = torch.tensor(0.0).to(device)

                comp_cells = aux_main.get("comp_cells")
                comp_source = (
                    comp_estimated
                    if comp_estimated is not None and comp_estimated.shape[0] > 0
                    else comp_cells
                )
                comp_losses_ready = comp_source is not None and comp_source.shape[0] > 0
                if comp_losses_ready:
                    comp_estimated_vals = comp_source.clone()
                    n_cells = batch_n_cells.squeeze(-1).float()
                    valid_mask = n_cells > 0
                    if valid_mask.any():
                        weights = n_cells[valid_mask]
                        weights = weights / weights.sum()
                        comp_estimated_sum = torch.sum(
                            comp_estimated_vals[valid_mask] * weights.unsqueeze(1), dim=0
                        )
                    else:
                        comp_estimated_sum = comp_estimated_vals.mean(dim=0)

                    comp_gt = F.one_hot(batch_ct_pc, num_classes=n_classes).float()
                    comp_gt = torch.mean(comp_gt, dim=0)
                    comp_gt = comp_gt.clamp_min(1e-8)
                    comp_gt = comp_gt / comp_gt.sum()

                    kl_eps = 1e-8
                    comp_estimated_log = torch.log(
                        comp_estimated_sum.clamp_min(kl_eps)
                    )

                    comp_logits = out_cell_type.clone()
                    logits_logsum = torch.logsumexp(comp_logits, dim=1, keepdim=True)
                    cell_log_probs = comp_logits - logits_logsum
                    comp_out = torch.exp(cell_log_probs).mean(dim=0)
                    comp_out = comp_out.clamp_min(kl_eps)
                    comp_out = comp_out / comp_out.sum()
                    comp_out_log = torch.log(comp_out.clamp_min(kl_eps))

                    kl_est_vec = F.kl_div(
                        comp_estimated_log, comp_gt, reduction="none"
                    )
                    kl_out_vec = F.kl_div(
                        comp_out_log, comp_gt, reduction="none"
                    )

                    pred_probs_cells = torch.softmax(out_cell_type.detach(), dim=1)
                    running_pred_counts += pred_probs_cells.sum(dim=0)
                    gt_onehot = F.one_hot(batch_ct_pc, num_classes=n_classes).float()
                    running_gt_counts += gt_onehot.sum(dim=0)

                    if slide_comp_target is not None:
                        target_dist = slide_comp_target
                    else:
                        target_dist = running_gt_counts / running_gt_counts.sum().clamp_min(1.0)

                    pred_dist = running_pred_counts / running_pred_counts.sum().clamp_min(1.0)
                    class_error = torch.abs(pred_dist - target_dist).detach()
                    class_weights = class_error + 1e-6
                    class_weights = class_weights / class_weights.sum().clamp_min(1e-6)

                    loss_comp_est_val = torch.sum(class_weights * kl_est_vec)
                    loss_comp_gt_val = torch.sum(class_weights * kl_out_vec)
                else:
                    loss_comp_est_val = torch.tensor(0.0, device=device)
                    loss_comp_gt_val = torch.tensor(0.0, device=device)
            else:
                loss_comp_est_val = torch.tensor(0.0).to(device)
                loss_comp_gt_val = torch.tensor(0.0).to(device)
                loss_expr_immune_val = torch.tensor(0.0).to(device)
                loss_expr_invasive_val = torch.tensor(0.0).to(device)

            var_pred = masked_var(out_expr, batch_expr_mask_pc)
            var_label = masked_var(batch_expr_pc, batch_expr_mask_pc)
            var_w_cfg = float(getattr(opts.training, "expr_var_penalty_weight", 0.0))
            if hurdle_enabled:
                # These branches are diagnostics only in hurdle mode.
                loss_expr_immune_val = torch.tensor(0.0, device=device)
                loss_expr_invasive_val = torch.tensor(0.0, device=device)
                loss_var_val = torch.tensor(0.0, device=device)
            elif var_w_cfg > 0:
                loss_var_val = var_w_cfg * torch.abs(var_pred - var_label)
            else:
                loss_var_val = torch.tensor(0.0, device=device)

            aux_losses = getattr(model, "last_aux_losses", {})
            loss_vq_val = aux_losses.get(
                "vq_patch", torch.tensor(0.0, device=device)
            )

            if hurdle_enabled:
                loss_expression_val = loss_expr_val
            else:
                loss_expression_val = (
                    loss_expr_val
                    + loss_ecrm_val
                    + loss_expr_immune_val
                    + loss_expr_invasive_val
                    + loss_var_val
                )
            loss_celltype_val = (
                loss_map_val
                + loss_ct_hist_val
                + loss_expr_ct_val
                + expr_ct_embed_loss_weight * loss_expr_ct_embed_val
                + logits_loss_weight * loss_logits_val
            )
            loss_composition_val = loss_comp_est_val + loss_comp_gt_val
            loss_auxiliary_val = (
                panel_completion_loss_weight * loss_panel_completion_val
                + loss_vq_val
            )
            loss = (
                loss_expression_val
                + loss_celltype_val
                + loss_composition_val
                + loss_auxiliary_val
            )

            loss_total = loss.detach()

            var_ratio = (
                1.0
                if hurdle_enabled
                else float(
                    (var_pred.detach() / var_label.detach().clamp_min(1e-6)).item()
                )
            )
            if not torch.isfinite(loss_total):
                logging.warning(
                    "Skip batch (non-finite loss) slide_id=%s var_pred=%.4f var_label=%.4f",
                    slide_id_val,
                    float(var_pred.detach().item()),
                    float(var_label.detach().item()),
                )
                continue
            if (
                not hurdle_enabled
                and torch.isfinite(var_pred)
                and torch.isfinite(var_label)
                and var_ratio > var_ratio_limit
            ):
                logging.warning(
                    "Skip batch (expr_var/label_var=%.2f > %.1f) slide_id=%s",
                    var_ratio,
                    var_ratio_limit,
                    slide_id_val,
                )
                continue

            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            loss_epoch += loss.mean().item()

            if hasattr(pbar, "set_description"):
                pbar.set_description(f"loss: {loss_total:.4f}")

            optimizer.step()

        if hurdle_enabled:
            logging.debug(
                "Hurdle unique-cell reconstruction observations this epoch: %d",
                len(hurdle_seen_cell_ids),
            )
        logging.info(
            "Epoch[%d/%d], Loss:%.4f",
            epoch + 1,
            opts.training.total_epochs,
            loss_epoch,
        )
        if use_celltype:
            pred_dist_epoch = running_pred_counts / running_pred_counts.sum().clamp_min(1.0)
            gt_dist_epoch = running_gt_counts / running_gt_counts.sum().clamp_min(1.0)
            logging.debug(
                "Epoch %d running composition pred %s | gt %s",
                epoch + 1,
                ", ".join(
                    f"{classes[idx]}:{pred_dist_epoch[idx]:.4f}"
                    for idx in range(n_classes)
                ),
                ", ".join(
                    f"{classes[idx]}:{gt_dist_epoch[idx]:.4f}"
                    for idx in range(n_classes)
                ),
            )
        ckpt_model_path = None
        if (epoch % opts.save_freqs.model_freq) == 0:
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            ckpt_model_path = save_path
            optim_save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_optim.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                optim_save_path,
            )
            logging.info(
                "Checkpoint epoch=%d model=%s optimiser=%s",
                epoch + 1,
                ckpt_model_path,
                optim_save_path,
            )

        # Evaluation and checkpoint selection
        val_metrics = None
        if val_dataloader is not None:
            val_metrics = validation_function(
                model,
                val_dataloader,
                expr_ref_torch_val,
                device,
                n_classes,
                graph_k=graph_k,
                graph_cross_patch=graph_cross_patch,
                graph_cross_patch_k=graph_cross_patch_k,
                graph_cross_patch_radius=graph_cross_patch_radius,
                slide_coord_map_by_slide=slide_coord_map_by_slide,
                expr_ref_torch_map=expr_ref_torch_val_map,
                holdout_mask_by_slide=(
                    holdout_mask_by_slide if gene_mask_variant else None
                ),
                gene_names=gene_names,
                epoch=epoch + 1,
                per_gene_dir=None,
                svg_rank_gene_indices_by_slide=val_svg_rank_indices_by_slide,
                svg_topk=svg_topk,
                **val_validation_extra_kwargs,
            )
            if hurdle_enabled:
                metric_utils.log_svg_validation_epoch(
                    val_metrics, epoch=epoch + 1, svg_topk=svg_topk
                )
                if gene_mask_variant:
                    imputer_task.log_holdout_svg_pearson(
                        val_metrics, split_tag="VAL", epoch=epoch + 1
                    )
            else:
                metric_utils.log_gene_pcc_epoch(
                    val_metrics,
                    split_tag="VAL",
                    epoch=epoch + 1,
                    svg_topk=svg_topk,
                )

        ext_metrics = None
        run_ext_eval = False
        if external_dataloader is not None:
            run_ext_eval = ((epoch + 1) % ext_eval_every_epochs == 0) or (
                ext_eval_final_epoch and (epoch + 1 == total_epochs)
            )
        if run_ext_eval and external_dataloader is not None:
            ext_metrics = validation_function(
                model,
                external_dataloader,
                expr_ref_torch,
                device,
                n_classes,
                graph_k=graph_k,
                graph_cross_patch=graph_cross_patch,
                graph_cross_patch_k=graph_cross_patch_k,
                graph_cross_patch_radius=graph_cross_patch_radius,
                slide_coord_map_by_slide=slide_coord_map_by_slide,
                expr_ref_torch_map=expr_ref_torch_map,
                holdout_mask_by_slide=(
                    holdout_mask_by_slide if gene_mask_variant else None
                ),
                gene_names=gene_names,
                epoch=epoch + 1,
                per_gene_dir=None,
                svg_rank_gene_indices_by_slide=ext_svg_rank_indices_by_slide,
                svg_topk=svg_topk,
                **ext_validation_extra_kwargs,
            )
            if gene_mask_variant:
                imputer_task.log_holdout_svg_pearson(
                    ext_metrics, split_tag="EXT", epoch=epoch + 1
                )
            elif not hurdle_enabled:
                metric_utils.log_gene_pcc_epoch(
                    ext_metrics,
                    split_tag="EXT",
                    epoch=epoch + 1,
                    svg_topk=svg_topk,
                )
        elif external_dataloader is not None:
            logging.info(
                "EXT evaluation skipped at epoch %d (every %d epochs)",
                epoch + 1,
                ext_eval_every_epochs,
            )

        val_gene_pooled_mean = (
            float(val_metrics.get("pearson_gene_pooled_mean", 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        val_ct_macro = (
            float(val_metrics.get("ct_accuracy_macro", 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        val_selection_metric = (
            float(val_metrics.get(best_selection_metric_name, 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        val_holdout_pooled_median = (
            float(val_metrics.get("holdout_gene_pooled_median", 0.0))
            if isinstance(val_metrics, dict)
            else 0.0
        )
        if isinstance(ext_metrics, dict):
            ext_gene_pooled_record = float(
                ext_metrics.get("pearson_gene_pooled_mean", 0.0)
            )
        else:
            ext_gene_pooled_record = None
        svg_validation = (
            val_metrics.get("fixed_gt_svg_validation", {})
            if hurdle_enabled and isinstance(val_metrics, dict)
            else {}
        )
        record = {
            "epoch": int(epoch + 1),
            "checkpoint": ckpt_model_path,
            "train_loss_total": float(loss_epoch),
            "val_pearson_gene_pooled_mean": val_gene_pooled_mean,
            "val_ct_accuracy_macro": val_ct_macro,
            "val_gene_pcc_valid_coverage": (
                float(val_metrics.get("pearson_gene_valid_coverage", 0.0))
                if isinstance(val_metrics, dict) else 0.0
            ),
            "val_mean_per_gene_w1": (
                float(val_metrics.get("hurdle_mean_per_gene_w1", float("inf")))
                if hurdle_enabled and isinstance(val_metrics, dict) else None
            ),
            "val_mean_per_gene_zero_gap": (
                float(val_metrics.get("hurdle_mean_per_gene_zero_gap", float("inf")))
                if hurdle_enabled and isinstance(val_metrics, dict) else None
            ),
            "val_positive_w1": (
                float(val_metrics.get("hurdle_positive_w1_mean", float("inf")))
                if hurdle_enabled and isinstance(val_metrics, dict) else None
            ),
            "ext_pearson_gene_pooled_mean": ext_gene_pooled_record,
            "external_source": external_source_tag,
            "parity_tolerance_abs": parity_tolerance_abs,
        }
        if gene_mask_variant and isinstance(val_metrics, dict):
            record.update(
                {
                    "val_holdout_gene_pooled_mean": val_metrics.get(
                        "holdout_gene_pooled_mean"
                    ),
                    "val_holdout_gene_pooled_median": val_metrics.get(
                        "holdout_gene_pooled_median"
                    ),
                }
            )
        for k_value in svg_topk:
            top_metrics = svg_validation.get(f"top{int(k_value)}", {})
            prefix = f"val_svg{int(k_value)}"
            record[f"{prefix}_pcc_median"] = top_metrics.get("pcc_median")
            record[f"{prefix}_ssim_median"] = top_metrics.get("ssim_median")
            record[f"{prefix}_cmd"] = top_metrics.get("cmd")
            record[f"{prefix}_pcc_valid_gene_count"] = top_metrics.get(
                "pcc_valid_gene_count", 0
            )
            record[f"{prefix}_ssim_valid_gene_count"] = top_metrics.get(
                "ssim_valid_gene_count", 0
            )
            record[f"{prefix}_requested_gene_count"] = top_metrics.get(
                "requested_gene_count", 0
            )
            record[f"{prefix}_full_k_of_k"] = bool(
                top_metrics.get("full_k_of_k", False)
            )
        epoch_records.append(record)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(epoch_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(record) + "\n")
        df_epochs = pd.DataFrame(epoch_records)
        df_epochs.to_csv(kpi_csv_path, index=False)
        df_epochs.to_csv(os.path.join(metrics_dir, "epoch_metrics.csv"), index=False)

        if val_metrics is not None:
            metrics_payload = {
                "epoch": int(epoch + 1),
                "checkpoint": ckpt_model_path,
                "val": val_metrics,
                "external": ext_metrics if ext_metrics is not None else {},
            }
            _write_json(
                os.path.join(metrics_dir, f"epoch_{epoch + 1:03d}_metrics.json"),
                metrics_payload,
            )

        if gene_mask_variant:
            primary_improved = val_selection_metric > (
                best_val_selection_metric + best_metric_eps
            )
            primary_tied = abs(
                val_selection_metric - best_val_selection_metric
            ) <= best_metric_eps
            if holdout_n_genes_eval > 0:
                should_update_best = primary_improved
            else:
                ct_constraint_ok = (
                    best_epoch is None
                    or val_ct_macro >= (best_val_ct_macro - best_metric_eps)
                )
                should_update_best = (
                    primary_improved and ct_constraint_ok
                ) or (
                    primary_tied
                    and val_ct_macro > (best_val_ct_macro + best_metric_eps)
                )
            if should_update_best:
                best_val_selection_metric = val_selection_metric
                best_val_ct_macro = val_ct_macro
                best_epoch = int(epoch + 1)
                best_ckpt_path = ckpt_model_path
            elif (
                holdout_n_genes_eval <= 0
                and primary_improved
                and not ct_constraint_ok
            ):
                logging.info(
                    "Best-checkpoint update rejected at epoch %d: %s %.6f > %.6f but ct_macro %.6f < %.6f",
                    epoch + 1,
                    best_selection_metric_name,
                    val_selection_metric,
                    best_val_selection_metric,
                    val_ct_macro,
                    best_val_ct_macro,
                )
        elif not hurdle_enabled:
            primary_improved = val_gene_pooled_mean > (best_val_gene_pooled + best_metric_eps)
            primary_tied = abs(val_gene_pooled_mean - best_val_gene_pooled) <= best_metric_eps
            ct_constraint_ok = (
                best_epoch is None
                or val_ct_macro >= (best_val_ct_macro - best_metric_eps)
            )
            if (primary_improved and ct_constraint_ok) or (
                primary_tied and val_ct_macro > (best_val_ct_macro + best_metric_eps)
            ):
                best_val_gene_pooled = val_gene_pooled_mean
                best_val_ct_macro = val_ct_macro
                best_epoch = int(epoch + 1)
                best_ckpt_path = ckpt_model_path
            elif primary_improved and not ct_constraint_ok:
                logging.info(
                    "Best-checkpoint update rejected at epoch %d: pooled_gene_pearson %.6f > %.6f but ct_macro %.6f < %.6f",
                    epoch + 1,
                    val_gene_pooled_mean,
                    best_val_gene_pooled,
                    val_ct_macro,
                    best_val_ct_macro,
                )

    # Best-checkpoint summary
    if gene_mask_variant:
        strict_best = {
            "task_name": imputer_task.TASK_NAME,
            "task_description": imputer_task.TASK_DESCRIPTION,
            "gene_mask_imputer": _to_serialisable(imputer_cfg_resolved),
            "selection_metric": best_selection_metric_name,
            "selection_constraint": (
                "none"
                if holdout_n_genes_eval > 0
                else "non_decreasing_ct_accuracy_macro"
            ),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_selection_metric": (
                float(best_val_selection_metric)
                if np.isfinite(best_val_selection_metric)
                else None
            ),
            "last_val_pearson_gene_pooled_mean": (
                val_gene_pooled_mean
                if "val_gene_pooled_mean" in locals()
                else None
            ),
            "last_val_holdout_gene_pooled_median": (
                val_holdout_pooled_median
                if "val_holdout_pooled_median" in locals()
                else None
            ),
            "best_val_ct_accuracy_macro": (
                float(best_val_ct_macro)
                if np.isfinite(best_val_ct_macro)
                else None
            ),
            "best_checkpoint": best_ckpt_path,
            "parity_tolerance_abs": parity_tolerance_abs,
            "external_source": external_source_tag,
        }
    elif hurdle_enabled:
        if svg_joint_selector_enabled:
            strict_best = checkpoint_selection.select_svg_joint_rank_checkpoint(
                epoch_records, topk=svg_topk
            )
            strict_best["selection_status"] = (
                "selected" if strict_best["best_epoch"] is not None else "no_eligible_epoch"
            )
            if strict_best["best_epoch"] is None:
                logging.warning(
                    "FULL has no eligible SVG checkpoint; no fallback checkpoint was selected"
                )
        else:
            strict_best = {
                "selection_metric": "deferred_to_full_svg_joint_rank_epoch",
                "selection_scope": "validation_metrics_logged_but_not_ranked_for_this_arm",
                "selection_status": "deferred_to_full",
                "fallback": None,
                "best_epoch": None,
                "best_checkpoint": None,
                "selected_metrics": None,
            }
        strict_best["run_role"] = svg_run_role
        strict_best["external_metrics_used_for_selection"] = False
        strict_best["fixed_svg_cohort_sha256"] = fixed_svg_cohort_audit[
            "combined_frozen_sha256"
        ]
        strict_best["parity_tolerance_abs"] = parity_tolerance_abs
        strict_best["external_source"] = external_source_tag
        _write_json(os.path.join(metrics_dir, "distribution_best.json"), strict_best)
    else:
        strict_best = {
            "selection_metric": "pearson_gene_pooled_mean",
            "selection_constraint": "non_decreasing_ct_accuracy_macro",
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_pearson_gene_pooled_mean": (
                float(best_val_gene_pooled) if np.isfinite(best_val_gene_pooled) else None
            ),
            "best_val_ct_accuracy_macro": (
                float(best_val_ct_macro) if np.isfinite(best_val_ct_macro) else None
            ),
            "best_checkpoint": best_ckpt_path,
            "parity_tolerance_abs": parity_tolerance_abs,
            "external_source": external_source_tag,
        }
    _write_json(os.path.join(metrics_dir, "strict_best.json"), strict_best)
    _write_json(os.path.join(experiment_path, "strict_best.json"), strict_best)
    logging.info("Training finished")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=0,
        type=int,
        help="resume training from this epoch, set to 0 for new training",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    return parser


def run_cli(variant=TrainingVariant.BASE):
    main(build_parser().parse_args(), variant=variant)


if __name__ == "__main__":
    run_cli(TrainingVariant.BASE)

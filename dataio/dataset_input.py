"""Patch-level histology dataset construction for GHIST+."""

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import json
import hashlib
import tifffile
import natsort
import h5py
from pathlib import Path
from tqdm import tqdm
import torchvision
import imageio
import torchstain
from torchvision import transforms
import cv2

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import torch.nn.functional as F

from .utils import load_image

from stainlib.augmentation.augmenter import HedLighterColorAugmenter


RUNTIME_MIN_MAX_CELLS_PER_PATCH = 2048


def _as_hwc_uint8(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}")

    image = np.nan_to_num(image, nan=255.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def _has_sufficient_tissue(image_hwc, white_threshold=245, min_tissue_fraction=0.02):
    gray = image_hwc.mean(axis=2)
    tissue_fraction = float(np.mean(gray < white_threshold))
    return tissue_fraction >= min_tissue_fraction


def _has_stable_stain_stats(
    image_hwc,
    white_threshold=245,
    min_tissue_pixels=256,
    min_gray_std=5.0,
    min_channel_std=2.0,
):
    gray = image_hwc.mean(axis=2)
    tissue_mask = gray < white_threshold
    tissue_pixels = image_hwc[tissue_mask]
    if tissue_pixels.shape[0] < min_tissue_pixels:
        return False

    tissue_pixels = tissue_pixels.astype(np.float32, copy=False)
    tissue_gray = gray[tissue_mask].astype(np.float32, copy=False)
    if float(tissue_gray.std()) < min_gray_std:
        return False
    if float(tissue_pixels.std(axis=0).mean()) < min_channel_std:
        return False
    return True


def _fit_stain_normalizer_bundle(target_hwc):
    target_chw = torch.from_numpy(
        np.moveaxis(target_hwc, -1, 0).astype(np.float32)
    )
    bundle = {}
    errors = {}

    try:
        macenko = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        macenko.fit(target_chw)
        bundle["macenko"] = macenko
    except Exception as exc:
        errors["macenko"] = exc

    try:
        reinhard = torchstain.normalizers.ReinhardNormalizer(backend="torch")
        reinhard.fit(target_chw)
        bundle["reinhard"] = reinhard
    except Exception as exc:
        errors["reinhard"] = exc

    if not bundle:
        raise RuntimeError(
            "Failed to fit stain normalizers: "
            + ", ".join(f"{name}={type(exc).__name__}: {exc}" for name, exc in errors.items())
        )
    return bundle


def norm_stain(target, to_transform, device, normalizer=None):
    _ = device  # keep signature; normalization happens on CPU

    patch_hwc = _as_hwc_uint8(to_transform)
    if not _has_sufficient_tissue(patch_hwc):
        return patch_hwc
    if not _has_stable_stain_stats(patch_hwc):
        return patch_hwc

    if normalizer is None:
        target_hwc = _as_hwc_uint8(target)
        normalizer = _fit_stain_normalizer_bundle(target_hwc)

    patch_chw = torch.from_numpy(np.moveaxis(patch_hwc, -1, 0).astype(np.float32))
    if isinstance(normalizer, dict):
        normalizer_chain = [
            ("macenko", normalizer.get("macenko")),
            ("reinhard", normalizer.get("reinhard")),
        ]
    else:
        normalizer_chain = [("primary", normalizer)]

    last_exc = None
    for name, norm_obj in normalizer_chain:
        if norm_obj is None:
            continue
        try:
            if name == "reinhard":
                norm = norm_obj.normalize(I=patch_chw)
            else:
                result = norm_obj.normalize(I=patch_chw, stains=True)
                norm = result[0] if isinstance(result, tuple) else result
            norm = norm.detach().cpu().numpy()
            if norm.ndim == 3 and norm.shape[0] == 3:
                norm = np.moveaxis(norm, 0, -1)
            norm = np.nan_to_num(norm, nan=255.0, posinf=255.0, neginf=0.0)
            return np.clip(norm, 0, 255).astype(np.uint8, copy=False)
        except TypeError:
            try:
                norm = norm_obj.normalize(I=patch_chw)
                norm = norm.detach().cpu().numpy()
                if norm.ndim == 3 and norm.shape[0] == 3:
                    norm = np.moveaxis(norm, 0, -1)
                norm = np.nan_to_num(norm, nan=255.0, posinf=255.0, neginf=0.0)
                return np.clip(norm, 0, 255).astype(np.uint8, copy=False)
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    return patch_hwc


def _df_to_payload(df):
    if df is None:
        return None
    return {
        "index": df.index.tolist(),
        "columns": df.columns.tolist(),
        "data": df.to_numpy().tolist(),
    }


def _df_from_payload(payload):
    if payload is None:
        return None
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    df.index = payload["index"]
    return df


def _build_cache_path(experiment_path, meta):
    _ = experiment_path
    cache_root = os.environ.get("CACHE_ROOT")
    if cache_root:
        cache_dir = Path(cache_root).expanduser().resolve() / "cache"
    else:
        nature_root = Path(__file__).resolve().parent.parent
        cache_dir = nature_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, sort_keys=True).encode("utf-8")
    digest = hashlib.md5(meta_json).hexdigest()
    return cache_dir / f"dataset_{digest}.pt"


def _resolve_max_cells_per_patch(configured_value):
    configured_value = int(configured_value)
    if configured_value < RUNTIME_MIN_MAX_CELLS_PER_PATCH:
        print(
            "Raising max_cells_per_patch from "
            f"{configured_value} to {RUNTIME_MIN_MAX_CELLS_PER_PATCH} "
            "for dense-slide stability."
        )
    return max(configured_value, RUNTIME_MIN_MAX_CELLS_PER_PATCH)


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


def get_region_spacing(size, mode, divisions_fold):
    """
    size = size of whole H&E image in pixels
    mode = train/val/test
    divisions_fold = list of 2 elements
        fraction of size indicating start and end of val/test region

    returns array of valid coordinates along vertical
    """
    div_a = int(round(divisions_fold[0] * size))
    div_b = int(round(divisions_fold[1] * size))

    wp_test = np.arange(div_a, div_b)

    if mode == "train":
        # remove val points to get train points
        wp_train = np.arange(size)
        mask = np.isin(wp_train, wp_test, invert=True)
        wp_train = wp_train[mask]
        return wp_train
    else:
        return wp_test


def find_patch_coordinates(w1, w2, patch_width=256, overlap=30):
    coordinates = []
    step_size = patch_width - overlap
    current_coord = w1

    while current_coord < w2:
        coordinates.append(min(current_coord, w2 - patch_width))
        current_coord += step_size

    return coordinates


def get_input_data(
    fp_nuc_seg,
    fp_hist,
    fp_nuc_sizes,
    mode,
    opts_data,
    fold_id,
    hsize,
    wsize,
    overlap,
    gene_names,
    divisions_fold,
    fp_expr,
    fp_cell_type,
    cell_types,
    experiment_path,
):

    # cell gene expressions
    if fp_expr is not None:
        df_expr = pd.read_csv(fp_expr, index_col=0)
        df_expr = df_expr[gene_names]
    else:
        df_expr = None

    if fp_cell_type is not None:
        df_ct = pd.read_csv(fp_cell_type, index_col="c_id")
        is_all_numbers = pd.to_numeric(df_ct["ct"], errors="coerce").notna().all()
        if not is_all_numbers:
            ct_dict = dict(zip(cell_types, list(range(len(cell_types)))))
            df_ct["ct"] = df_ct["ct"].map(ct_dict).astype(int)
            print(f"Cell type data shape, {df_ct.shape}")
        df_ct["ct"] = df_ct["ct"] + 1
    else:
        df_ct = None

    nuclei = load_image(fp_nuc_seg)
    hist = load_image(fp_hist)

    cache_meta = {
        "fp_hist": fp_hist,
        "fp_nuc_seg": fp_nuc_seg,
        "fp_expr": fp_expr,
        "fp_cell_type": fp_cell_type,
        "fp_nuc_sizes": fp_nuc_sizes,
        "mode": mode,
        "fold_id": fold_id,
        "hsize": hsize,
        "wsize": wsize,
        "overlap": overlap,
        "divisions": divisions_fold,
        "gene_names": gene_names,
        "hist_mtime": os.path.getmtime(fp_hist) if fp_hist else None,
        "nuc_mtime": os.path.getmtime(fp_nuc_seg) if fp_nuc_seg else None,
        "expr_mtime": os.path.getmtime(fp_expr) if fp_expr else None,
        "celltype_mtime": os.path.getmtime(fp_cell_type) if fp_cell_type else None,
        "nuc_sizes_mtime": os.path.getmtime(fp_nuc_sizes) if fp_nuc_sizes else None,
    }
    cache_path = _build_cache_path(experiment_path, cache_meta)
    if cache_path.exists():
        payload = torch.load(cache_path, weights_only=False)
        coords_starts_valid = [tuple(c) for c in payload["coords"]]
        all_intersect = payload["all_intersect"]
        df_ct = _df_from_payload(payload["df_ct"])
        df_expr = _df_from_payload(payload["df_expr"])
        norms_hist = np.array(payload["norms"])
        print(f"[cache] Loaded dataset metadata from {cache_path}")
        return (
            coords_starts_valid,
            hist,
            nuclei,
            all_intersect,
            df_ct,
            df_expr,
            norms_hist,
        )

    # nuclei = nuclei[2000:4000, 2000:4000]
    # hist = hist[2000:4000, 2000:4000]

    whole_h = hist.shape[0]
    whole_w = hist.shape[1]

    print(f"Histology image {hist.shape}, Nuclei {nuclei.shape}")

    # valid region of whole image
    wp = get_region_spacing(whole_h, mode, divisions_fold)
    nuclei_fold = nuclei[wp, :]

    # final cells = those in segmentation, meets min size, expr data, cell type data
    ids_seg = np.unique(nuclei_fold)
    ids_seg = ids_seg[ids_seg != 0]

    # meets min size req
    if fp_nuc_sizes is not False:
        df_sizes = pd.read_csv(fp_nuc_sizes, index_col=0)
        min_nuc_size = opts_data.min_nuc_area
        df_sizes = df_sizes[df_sizes["size_pix_histology"] >= min_nuc_size]

        ids_meet_min = df_sizes.index.tolist()

        all_intersect = list(set(ids_seg) & set(list(ids_meet_min)))
    else:
        all_intersect = list(set(ids_seg))

    if fp_expr is not None:
        all_intersect = list(set(all_intersect) & set(df_expr.index.tolist()))
        # get expr of the cells
        df_expr = df_expr[df_expr.index.isin(all_intersect)]
        assert list(df_expr.index) == df_expr.index.tolist()
        df_expr = opts_data.expr_scale * np.log1p(df_expr)
    else:
        df_expr = None

    if fp_cell_type is not None:
        all_intersect = list(set(all_intersect) & set(list(df_ct.index)))
        df_ct = df_ct.loc[all_intersect, :]
    else:
        df_ct = None

    all_intersect = natsort.natsorted(all_intersect)

    n_cells = len(all_intersect)
    print(f"{n_cells} cells")

    # overlapping patches
    w_starts = list(np.arange(0, whole_w - wsize, wsize - overlap))
    w_starts.append(whole_w - wsize)

    coord_idx = find_patch_coordinates(0, len(wp), patch_width=hsize, overlap=overlap)
    h_starts = wp[coord_idx]
    print("Patches min/max coords", h_starts.min(), h_starts.max() + hsize)

    # check there are cells in the patches
    print("Getting valid patches")
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    coords_starts_valid = []

    # # save coords_starts_valid to file
    # if mode == "train":
    #     fp_coords = "coords_train_%d.txt" % (fold_id)
    # elif mode == "test":
    #     fp_coords = "coords_test_%d.txt" % (fold_id)
    # else:
    #     fp_coords = "coords_val_%d.txt" % (fold_id)

    # if os.path.exists(fp_coords):
    #     coords_starts_valid = []
    #     with open(fp_coords, 'r') as f:
    #         for line in f:
    #             # Split the line by comma and convert to integers, then convert to tuple
    #             x, y = map(int, line.strip().split(','))
    #             coords_starts_valid.append((x, y))
    # else:
    for hs, ws in tqdm(coords_starts):
        nuclei_p = nuclei[hs : hs + hsize, ws : ws + wsize]

        ids_seg = np.unique(nuclei_p)
        ids_seg = ids_seg[ids_seg != 0]
        valid_ids = list(set(ids_seg) & set(all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_p)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_p == k] = v

        if np.sum(nuclei_valid) > 0:
            coords_starts_valid.append((hs, ws))

    # with open(fp_coords, "w") as f:
    #     for hs, ws in coords_starts_valid:
    #         # Write each tuple as a line in the file, formatted as 'hs, ws'
    #         f.write(f"{hs},{ws}\n")

    # Initialize min and max values with the first element of the list
    min_hs, min_ws = coords_starts_valid[0]
    max_hs, max_ws = coords_starts_valid[0]

    # Iterate through the list to find min and max values
    for hs, ws in coords_starts_valid:
        if hs < min_hs:
            min_hs = hs
        if hs > max_hs:
            max_hs = hs
        if ws < min_ws:
            min_ws = ws
        if ws > max_ws:
            max_ws = ws

    # Standardisation of RGB.
    print("Standardisation")
    fp_norms = f"{experiment_path}/standardisation_hist_fold_{fold_id}.npy"

    if mode == "train":
        if not os.path.exists(fp_norms):
            hist_means = np.zeros(3)
            hist_stds = np.zeros(3)
            for hs, ws in tqdm(coords_starts):

                hist_p = hist[hs : hs + hsize, ws : ws + wsize]

                hist_means += np.mean(hist_p, (0, 1))
                hist_stds += np.std(hist_p, (0, 1))

            hist_means = hist_means / len(coords_starts)
            hist_stds = hist_stds / len(coords_starts)

            norms_hist = np.vstack((hist_means, hist_stds))
            np.save(fp_norms, norms_hist)

    norms_hist = np.load(fp_norms)

    payload = {
        "coords": coords_starts_valid,
        "all_intersect": all_intersect,
        "df_ct": _df_to_payload(df_ct),
        "df_expr": _df_to_payload(df_expr),
        "norms": norms_hist.tolist(),
    }
    torch.save(payload, cache_path)
    print(f"[cache] Saved dataset metadata to {cache_path}")

    return coords_starts_valid, hist, nuclei, all_intersect, df_ct, df_expr, norms_hist


class DataProcessing(data.Dataset):
    def __init__(
        self,
        opts_data_sources,
        opts_data,
        opts_regions,
        opts_comps,
        opts_stain_norm,
        classes,
        gene_names,
        device,
        experiment_path,
        stain_aug,
        fold_id=1,
        mode="train",
        immune_sampler_boost=1.0,
        immune_class_multipliers=None,
    ):

        # check all the files to load
        check_path(opts_data_sources.fp_nuc_seg)
        check_path(opts_data_sources.fp_hist)
        check_path(opts_data_sources.fp_nuc_sizes)

        if mode != "test":
            check_path(opts_data_sources.fp_expr)
            fp_expr = opts_data_sources.fp_expr
        else:
            fp_expr = None

        if opts_comps.celltype and mode != "test":
            check_path(opts_data_sources.fp_cell_type)
            fp_cell_type = opts_data_sources.fp_cell_type
            self.cell_types = opts_data.cell_types
            self.comps_celltype = True
        else:
            fp_cell_type = None
            self.cell_types = None
            self.comps_celltype = False

        self.normstain = (
            (opts_stain_norm.norm_train * (mode == "train"))
            or (opts_stain_norm.norm_val * (mode == "val"))
            or (opts_stain_norm.norm_test * (mode == "test"))
        )
        if self.normstain:
            check_path(opts_stain_norm.fp_norm_ref)

            stain_ref = load_image(opts_stain_norm.fp_norm_ref)
            resized_h = max(1, int(stain_ref.shape[0] * opts_stain_norm.resized_scale))
            resized_w = max(1, int(stain_ref.shape[1] * opts_stain_norm.resized_scale))
            stain_ref = cv2.resize(
                stain_ref, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
            )
            self.stain_ref = _as_hwc_uint8(stain_ref)
            self._stain_normalizer = None
        print("Do stain normalisation:", self.normstain)

        self.classes = classes
        self.mode = mode
        self.fold_id = fold_id
        self.gene_names = gene_names
        self.max_cells_per_patch = _resolve_max_cells_per_patch(
            opts_data.max_cells_per_patch
        )
        self.hsize = opts_data.hsize
        self.wsize = opts_data.wsize
        self.opts_data = opts_data
        self.device = device
        self.experiment_path = experiment_path
        self.stain_aug = stain_aug
        self._immune_class_multipliers = self._sanitize_immune_multipliers(
            immune_class_multipliers
        )

        # fraction of image size: height start to height end
        divisions_fold = opts_regions.divisions[self.fold_id - 1]

        # overlap between tiles (pixels)
        if mode == "train":
            overlap = 0
        else:
            overlap = opts_data.overlap

        (
            coords_starts_valid,
            self.hist,
            self.nuclei,
            self.all_intersect,
            self.df_ct,
            self.df_expr,
            norms_hist,
        ) = get_input_data(
            opts_data_sources.fp_nuc_seg,
            opts_data_sources.fp_hist,
            opts_data_sources.fp_nuc_sizes,
            self.mode,
            opts_data,
            fold_id,
            self.hsize,
            self.wsize,
            overlap,
            gene_names,
            divisions_fold,
            fp_expr,
            fp_cell_type,
            self.cell_types,
            experiment_path,
        )

        self.norms_hist = norms_hist.copy()
        self.coords_starts = coords_starts_valid
        self._norm_failures = 0
        self._norm_failures_max_log = 5

        self.n_patches = len(self.coords_starts)
        self._all_intersect_set = set(self.all_intersect)
        self.patch_weights = self._compute_patch_sampling_weights(
            immune_sampler_boost
        )

        # Augmentation
        self.tfs = v2.Compose(
            [
                v2.ToImage(),
                # v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomApply([v2.RandomRotation((90, 90))], p=0.25),
                v2.RandomApply([v2.RandomRotation((180, 180))], p=0.25),
                v2.RandomApply([v2.RandomRotation((270, 270))], p=0.25),
                v2.ToDtype(torch.float32),
            ]
        )

        self.tfs_test = v2.Compose(
            [
                v2.ToImage(),
                # v2.ToDtype(torch.float32, scale=True),
                v2.ToDtype(torch.float32),
            ]
        )

        self.hed_lighter_aug = HedLighterColorAugmenter()

    def _get_stain_normalizer(self):
        if not self.normstain:
            return None
        if self._stain_normalizer is None:
            self._stain_normalizer = _fit_stain_normalizer_bundle(
                _as_hwc_uint8(self.stain_ref)
            )
        return self._stain_normalizer

    def _truncate_patch_ids(self, patch_ids, max_cells_per_patch, hs, ws):
        if len(patch_ids) <= max_cells_per_patch:
            return patch_ids
        print(
            f"patch coords {hs}, {ws} have {len(patch_ids)} cells; "
            f"truncating to {max_cells_per_patch}"
        )
        random_subsample = bool(
            getattr(self.opts_data, "random_cell_subsample_train", True)
        )
        if self.mode == "train" and random_subsample:
            selected = np.random.choice(
                patch_ids,
                size=max_cells_per_patch,
                replace=False,
            )
            return np.sort(selected)
        return patch_ids[:max_cells_per_patch]

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_patches

    def __getitem__(self, index):
        "Generates one sample of data"

        hs, ws = self.coords_starts[index]

        nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
        hist_patch = self.hist[hs : hs + self.hsize, ws : ws + self.wsize]

        if self.normstain:
            try:
                hist_patch = norm_stain(
                    None,
                    hist_patch,
                    self.device,
                    normalizer=self._get_stain_normalizer(),
                )
            except Exception as exc:
                if self._norm_failures < self._norm_failures_max_log:
                    print(
                        f"norm stain failed for patch coords {hs}, {ws}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                elif self._norm_failures == self._norm_failures_max_log:
                    print("Further stain normalization failures muted.")
                self._norm_failures += 1

        if self.mode == "train" and self.stain_aug:
            # https://github.com/sebastianffx/stainlib/blob/main/stainlib_augmentation.ipynb
            self.hed_lighter_aug.randomize()
            hist_patch = self.hed_lighter_aug.transform(hist_patch)

        hist_patch = np.nan_to_num(hist_patch, nan=0.0, posinf=255.0, neginf=0.0)
        hist_patch = np.clip(hist_patch, 0.0, 255.0)

        ids_seg = np.unique(nuclei_patch)
        ids_seg = ids_seg[ids_seg != 0]

        # make sure cells have valid data
        valid_ids = list(set(ids_seg) & set(self.all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_patch)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_patch == k] = v

        if self.comps_celltype and self.mode != "test":
            # map to cell type nuclei map
            dictionary = dict(zip(valid_ids, self.df_ct.loc[valid_ids, "ct"].tolist()))
            types_patch = np.copy(nuclei_valid)
            for k, v in dictionary.items():
                types_patch[nuclei_valid == k] = v
        else:
            types_patch = np.where(nuclei_valid > 0, 1, 0)

        # standardisation
        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        hist_patch = hist_patch - means
        hist_patch = hist_patch / stds

        patch_ids = np.unique(nuclei_valid)
        patch_ids = patch_ids[patch_ids != 0]

        max_cells_per_patch = self.max_cells_per_patch
        patch_ids = self._truncate_patch_ids(
            patch_ids,
            max_cells_per_patch,
            hs,
            ws,
        )

        n_cells = len(patch_ids)

        expr_pad = np.zeros((max_cells_per_patch, len(self.gene_names)))
        if self.mode != "test":
            expr = self.df_expr.loc[patch_ids, :].to_numpy()
            expr_pad[:n_cells, :] = expr.copy()

        gt_types_pad = np.zeros(max_cells_per_patch)
        if self.comps_celltype and self.mode != "test":
            # cell type labels (previously added 1 to df_ct such that 0 is bkg)
            gt_types_pad[:n_cells] = self.df_ct.loc[patch_ids, "ct"].to_numpy() - 1
        gt_types_torch = torch.from_numpy(gt_types_pad).long()

        # cell IDs in patch
        patch_ids_pad = np.zeros(max_cells_per_patch)
        patch_ids_pad[:n_cells] = patch_ids.copy()
        patch_ids_torch = torch.from_numpy(patch_ids_pad).long()

        # number of cells in patch
        n_cells = np.array([n_cells])
        n_cells_torch = torch.from_numpy(n_cells).long()

        x_input = np.concatenate(
            (
                np.expand_dims(nuclei_valid, -1),
                np.expand_dims(types_patch, -1),
                hist_patch,
            ),
            -1,
        )

        # augmentation
        if self.mode == "train":
            x_input = self.tfs(x_input)
        else:
            x_input = self.tfs_test(x_input)

        nuclei_torch = x_input[0, :, :].type(torch.LongTensor)
        types_patch_torch = x_input[1, :, :].type(torch.LongTensor)
        hist_torch = x_input[2:, :, :]

        expr_torch = torch.from_numpy(expr_pad).float()

        return (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
        )

    def _compute_patch_sampling_weights(self, boost_factor):
        if self.mode != "train" or not self.comps_celltype:
            return [1.0] * max(1, len(self.coords_starts))
        if boost_factor <= 1.0 or not self.cell_types or self.df_ct is None:
            return [1.0] * max(1, len(self.coords_starts))

        immune_whitelist = {
            "b",
            "t",
            "plasma",
            "macrophage",
            "myeloid",
            "myeloid (excluding macrophage)",
        }
        immune_ct_map = {
            idx + 1: name
            for idx, name in enumerate(self.cell_types)
            if name.strip().lower() in immune_whitelist
        }
        immune_ct_indices = set(immune_ct_map.keys())
        if not immune_ct_indices:
            return [1.0] * max(1, len(self.coords_starts))

        ct_counts_series = (
            self.df_ct["ct"].astype(int).value_counts()
            if "ct" in self.df_ct.columns
            else None
        )
        if ct_counts_series is None or ct_counts_series.empty:
            return [1.0] * max(1, len(self.coords_starts))

        immune_totals = {
            idx: int(ct_counts_series.get(idx, 0)) for idx in immune_ct_indices
        }
        if sum(immune_totals.values()) <= 0:
            return [1.0] * max(1, len(self.coords_starts))

        mean_count = np.mean(list(immune_totals.values()))
        class_rarity = {
            idx: mean_count / max(count, 1) for idx, count in immune_totals.items()
        }
        mean_rarity = np.mean(list(class_rarity.values()))
        if mean_rarity <= 0:
            mean_rarity = 1.0
        class_rarity = {idx: val / mean_rarity for idx, val in class_rarity.items()}
        rarity_cap = float(getattr(self.opts_data, "immune_rarity_cap", 1.5))
        rarity_cap = max(rarity_cap, 1.0)

        immune_weights = []
        multiplier_map = self._immune_class_multipliers or {}
        for hs, ws in self.coords_starts:
            nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
            ids_seg = np.unique(nuclei_patch)
            ids_seg = ids_seg[ids_seg != 0]
            valid_ids = [cid for cid in ids_seg if cid in self._all_intersect_set]
            if not valid_ids:
                immune_weights.append(1.0)
                continue
            class_hits = {}
            for cid in valid_ids:
                ct_val = self.df_ct.loc[cid, "ct"] if cid in self.df_ct.index else None
                if ct_val in immune_ct_indices:
                    class_hits[ct_val] = class_hits.get(ct_val, 0) + 1
            if not class_hits:
                immune_weights.append(1.0)
                continue
            total_valid = max(len(valid_ids), 1)
            adaptive_score = 0.0
            for idx, hits in class_hits.items():
                rarity = class_rarity.get(idx, 1.0)
                rarity *= max(multiplier_map.get(idx, 1.0), 0.0)
                adaptive_score += (hits / total_valid) * rarity
            adaptive_score = min(max(adaptive_score, 0.0), rarity_cap)
            weight = 1.0 + (boost_factor - 1.0) * adaptive_score
            immune_weights.append(weight)

        if immune_weights:
            arr = np.array(immune_weights, dtype=np.float32)
            arr = arr * (len(arr) / arr.sum())
            return arr.tolist()
        return [1.0]

    def _sanitize_immune_multipliers(self, values):
        if not values:
            return {}
        clean = {}
        for key, val in values.items():
            try:
                clean[int(key)] = max(float(val), 0.0)
            except (TypeError, ValueError):
                continue
        return clean

    def set_immune_sampling_multipliers(self, multipliers):
        """Update cached per-class multipliers used by the adaptive sampler."""
        self._immune_class_multipliers = self._sanitize_immune_multipliers(
            multipliers
        )

    def refresh_patch_sampling_weights(self, boost_factor):
        """Recompute patch-level weights after the multipliers/boost factor change."""
        self.patch_weights = self._compute_patch_sampling_weights(boost_factor)
        return self.patch_weights

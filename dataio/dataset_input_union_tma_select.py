"""Union-panel dataset wrapper that adds a gene-presence mask per sample."""
import os
import copy
import numpy as np
import torch
from .dataset_input_tma_select import DataProcessing

__all__ = ["DataProcessingUnion"]


class DataProcessingUnion(DataProcessing):
    def __init__(self, *args, **kwargs):
        # Extract mask path if provided on the source object (first positional arg)
        # Drop args unsupported by base class
        kwargs.pop("return_expr_mask", None)

        args = list(args)

        # Normalise stain norm ref: if list/tuple, use the first entry (DataProcessing expects a single path)
        if len(args) > 4:
            opts_stain_norm = args[4]
            if hasattr(opts_stain_norm, "fp_norm_ref") and isinstance(
                opts_stain_norm.fp_norm_ref, (list, tuple)
            ):
                opts_stain_norm = copy.copy(opts_stain_norm)
                opts_stain_norm.fp_norm_ref = opts_stain_norm.fp_norm_ref[0]
                args[4] = opts_stain_norm

        mask_path = None
        # src may be passed positionally or via kwargs (opts_data_sources)
        src = args[0] if args else kwargs.get("opts_data_sources", None)
        if src is not None:
            if hasattr(src, "fp_mask"):
                mask_path = getattr(src, "fp_mask")
            self_slide = getattr(src, "slide_idx", -1)
        else:
            self_slide = -1

        super().__init__(*args, **kwargs)

        if mask_path and os.path.isfile(mask_path):
            mask = np.load(mask_path)
        else:
            mask = np.ones(len(self.gene_names), dtype=np.float32)
        self.gene_mask = torch.from_numpy(mask.astype(np.float32))
        self.slide_idx = self_slide

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
        ) = base[:7]
        extras = base[7:]

        n_cells = int(n_cells_torch.item())
        mask_pad = torch.zeros_like(expr_torch)
        if n_cells > 0:
            mask_pad[:n_cells, :] = self.gene_mask

        # attach slide_id so batches can be kept single-slide for avgexp
        slide_id = torch.tensor(getattr(self, "slide_idx", -1), dtype=torch.long)

        sample = (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
            mask_pad,
            slide_id,
        )
        return sample + extras

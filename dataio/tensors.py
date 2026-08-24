"""Tensor reshaping helpers for GHIST+ batches."""

import torch


def flatten_expr_mask(mask_batch: torch.Tensor, n_cells_batch: torch.Tensor):
    """
    Collapse a per-patch expression mask (B, max_cells, n_genes) to a per-cell
    mask aligned with concatenated outputs shaped (total_cells, n_genes).
    """
    if mask_batch is None:
        return None
    if mask_batch.ndim == 2:
        return mask_batch

    device = mask_batch.device
    n_genes = mask_batch.shape[-1]
    n_cells_flat = n_cells_batch.view(-1)
    masks = []
    for idx in range(mask_batch.shape[0]):
        n_valid = int(n_cells_flat[idx].item()) if idx < n_cells_flat.numel() else 0
        if n_valid <= 0:
            continue
        masks.append(mask_batch[idx, :n_valid, :])
    if masks:
        return torch.cat(masks, dim=0)
    return torch.zeros((0, n_genes), device=device, dtype=mask_batch.dtype)


def flatten_expr(expr_batch: torch.Tensor, n_cells_batch: torch.Tensor):
    """
    Collapse a per-patch expression tensor (B, max_cells, n_genes) to a per-cell
    tensor aligned with concatenated outputs shaped (total_cells, n_genes).
    """
    if expr_batch is None:
        return None
    if expr_batch.ndim == 2:
        return expr_batch

    device = expr_batch.device
    n_genes = expr_batch.shape[-1]
    n_cells_flat = n_cells_batch.view(-1)
    exprs = []
    for idx in range(expr_batch.shape[0]):
        n_valid = int(n_cells_flat[idx].item()) if idx < n_cells_flat.numel() else 0
        if n_valid <= 0:
            continue
        exprs.append(expr_batch[idx, :n_valid, :])
    if exprs:
        return torch.cat(exprs, dim=0)
    return torch.zeros((0, n_genes), device=device, dtype=expr_batch.dtype)

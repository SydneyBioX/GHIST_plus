"""Collate functions for GHIST+ training batches."""

import torch
from torch.utils.data import default_collate

# Positions in the tuples returned by DataProcessing (7 fields) and
# DataProcessingUnion (9 fields): n_cells, then the per-cell tensors whose first
# dimension is the (padded) cell axis: expr, gt_types, patch_ids, expr mask.
_N_CELLS = 4
_PER_CELL = (3, 5, 6, 7)


def collate_compact_batch(batch):
    """Stack a batch, padding the cell axis to the largest cell count in it.

    The default pipeline pads every item to ``max_cells_per_patch`` (2048) rows,
    so a batch of 64 carries ~2 x 64 x 2048 x n_genes float32 of mostly zeros
    through shared memory and over PCIe. Everything downstream reads cell rows
    as ``[b, :n_cells[b]]`` (see dataio/tensors.py), so a cell axis of
    ``max(n_cells)`` rows holds exactly the same data.

    Works on items padded either way: per-cell tensors are cut to ``n_cells``
    rows and zero-padded to the batch maximum (at least one row, so no tensor
    has a zero-sized dimension).
    """
    n_cells = [int(item[_N_CELLS].reshape(-1)[0]) for item in batch]
    rows = max(1, max(n_cells))
    out = []
    for j in range(len(batch[0])):
        column = [item[j] for item in batch]
        if j in _PER_CELL:
            first = column[0]
            padded = torch.zeros((len(column), rows, *first.shape[1:]), dtype=first.dtype)
            for b, (tensor, n) in enumerate(zip(column, n_cells)):
                padded[b, :n] = tensor[:n]
            out.append(padded)
        else:
            out.append(default_collate(column))
    return tuple(out)

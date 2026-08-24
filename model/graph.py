"""Cell graph construction helpers for GHIST+."""

from typing import NamedTuple

import torch


class CellGraph(NamedTuple):
    coords: torch.Tensor
    patch_index: torch.Tensor
    edge_index: torch.Tensor
    cells_per_patch: torch.Tensor


def _build_knn_graph(coords: torch.Tensor, patch_index: torch.Tensor, k: int) -> torch.Tensor:
    if coords.numel() == 0 or patch_index.numel() == 0:
        device = coords.device if coords.numel() else patch_index.device
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edges = []
    unique_patches = torch.unique(patch_index, sorted=True)
    for pid in unique_patches:
        idx = torch.where(patch_index == pid)[0]
        n = idx.numel()
        if n <= 1:
            continue
        coords_patch = coords[idx]
        dist = torch.cdist(coords_patch, coords_patch, p=2)
        dist.fill_diagonal_(float("inf"))
        k_eff = min(max(int(k), 1), n - 1)
        if k_eff <= 0:
            continue
        nbr = dist.topk(k_eff, largest=False).indices
        src = idx.unsqueeze(1).expand(-1, k_eff)
        dst = idx[nbr]
        edges.append(torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0))

    if edges:
        return torch.cat(edges, dim=1)
    device = coords.device if coords.numel() else patch_index.device
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def _build_knn_graph_cross_patch(
    coords: torch.Tensor,
    patch_index: torch.Tensor,
    global_mask: torch.Tensor,
    k: int,
    radius: float | None = None,
) -> torch.Tensor:
    if coords.numel() == 0 or patch_index.numel() == 0 or global_mask.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    idx = torch.where(global_mask.to(device=coords.device, dtype=torch.bool))[0]
    n = int(idx.numel())
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)

    coords_g = coords[idx]
    patch_g = patch_index[idx]
    dist = torch.cdist(coords_g, coords_g, p=2)
    dist.fill_diagonal_(float("inf"))
    same_patch = patch_g.view(-1, 1) == patch_g.view(1, -1)
    dist = dist.masked_fill(same_patch, float("inf"))
    if radius is not None and float(radius) > 0:
        dist = dist.masked_fill(dist > float(radius), float("inf"))

    k_eff = min(max(int(k), 1), n - 1)
    nbr_local = dist.topk(k_eff, largest=False).indices
    nbr_dist = dist.gather(1, nbr_local)
    valid = torch.isfinite(nbr_dist)
    if not valid.any():
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
    src = idx.view(-1, 1).expand(-1, k_eff)[valid]
    dst = idx[nbr_local[valid]]
    return torch.stack([src, dst], dim=0)


def _coalesce_edges(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device if edge_index is not None else "cpu")
    src = edge_index[0].long()
    dst = edge_index[1].long()
    valid = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes)
    src = src[valid]
    dst = dst[valid]
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    key = src * n_nodes + dst
    perm = torch.argsort(key)
    key_sorted = key[perm]
    keep = torch.ones_like(key_sorted, dtype=torch.bool)
    keep[1:] = key_sorted[1:] != key_sorted[:-1]
    uniq_idx = perm[keep]
    return torch.stack([src[uniq_idx], dst[uniq_idx]], dim=0)


def build_cell_graph(
    nuclei_batch: torch.Tensor,
    patch_ids_batch: torch.Tensor,
    k_neighbors: int,
    coords_batch: torch.Tensor | None = None,
    cell_coord_map: dict | None = None,
    cross_patch: bool = False,
    cross_patch_k: int | None = None,
    cross_patch_radius: float | None = None,
) -> CellGraph:
    """
    Construct per-cell centroids and an intra-patch kNN graph so ECRM can mix
    logits at the cell granularity. The order of cells follows the sorted nuclei
    IDs to stay aligned with batch_ct/batch_expr tensors.
    """
    device = nuclei_batch.device
    B, H, W = nuclei_batch.shape
    coords_local_list = []
    coords_global_list = []
    patch_assign_list = []
    global_coord_list = []
    cells_per_patch = []
    coords_batch_valid_global = (
        isinstance(coords_batch, torch.Tensor)
        and coords_batch.ndim == 3
        and coords_batch.shape[0] == B
        and coords_batch.shape[2] >= 2
    )
    has_coord_map = isinstance(cell_coord_map, dict) and len(cell_coord_map) > 0

    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

    for b in range(B):
        mask = nuclei_batch[b]
        ids = torch.unique(mask, sorted=True)
        ids = ids[ids > 0]
        n_valid = int(ids.numel())
        cells_per_patch.append(n_valid)
        if n_valid == 0:
            continue

        local_rank = 0

        for cid in ids:
            c_mask = (mask == cid)
            area = c_mask.sum()
            if area.item() == 0:
                continue
            cid_int = int(cid.item())
            c_mask_f = c_mask.float()
            area_f = area.float()
            cy_local = (c_mask_f * yy).sum() / area_f
            cx_local = (c_mask_f * xx).sum() / area_f
            cy_local = (cy_local / max(H - 1, 1)) * 2 - 1
            cx_local = (cx_local / max(W - 1, 1)) * 2 - 1
            if has_coord_map and cid_int in cell_coord_map:
                cyx = cell_coord_map[cid_int]
                cy_global = torch.tensor(float(cyx[0]), dtype=torch.float32, device=device)
                cx_global = torch.tensor(float(cyx[1]), dtype=torch.float32, device=device)
                has_global_coord = True
            elif coords_batch_valid_global:
                if local_rank < coords_batch.shape[1]:
                    cxy = coords_batch[b, local_rank, :2].float().to(device)
                    cy_global = cxy[1]
                    cx_global = cxy[0]
                    has_global_coord = True
                else:
                    cy_global = cy_local
                    cx_global = cx_local
                    has_global_coord = False
            else:
                cy_global = cy_local
                cx_global = cx_local
                has_global_coord = False
            coords_local_list.append(torch.stack([cy_local, cx_local]))
            coords_global_list.append(torch.stack([cy_global, cx_global]))
            patch_assign_list.append(torch.tensor(b, dtype=torch.long, device=device))
            global_coord_list.append(torch.tensor(has_global_coord, dtype=torch.bool, device=device))
            local_rank += 1

    if coords_local_list:
        coords_local = torch.stack(coords_local_list, dim=0)
        coords_global = torch.stack(coords_global_list, dim=0)
        patch_index = torch.stack(patch_assign_list, dim=0)
        global_mask = torch.stack(global_coord_list, dim=0)
    else:
        coords_local = torch.zeros((0, 2), device=device)
        coords_global = torch.zeros((0, 2), device=device)
        patch_index = torch.zeros((0,), dtype=torch.long, device=device)
        global_mask = torch.zeros((0,), dtype=torch.bool, device=device)

    use_global_graph = (
        bool(cross_patch)
        and coords_global.shape[0] > 1
        and bool(global_mask.numel() > 0)
        and bool(global_mask.all().item())
    )
    coords = coords_global if use_global_graph else coords_local
    edge_index = _build_knn_graph(coords, patch_index, k_neighbors)
    if use_global_graph:
        k_cross = int(cross_patch_k) if cross_patch_k is not None else int(k_neighbors)
        edge_cross = _build_knn_graph_cross_patch(
            coords,
            patch_index,
            global_mask,
            k_cross,
            radius=cross_patch_radius,
        )
        edge_index = torch.cat([edge_index, edge_cross], dim=1)
        edge_index = _coalesce_edges(edge_index, n_nodes=int(coords.shape[0]))
    cells_per_patch_tensor = torch.tensor(
        cells_per_patch if cells_per_patch else [0] * B,
        dtype=torch.long,
        device=device,
    )
    return CellGraph(coords, patch_index, edge_index, cells_per_patch_tensor)

"""Metric helpers for training and validation."""

import logging
import math

import numpy as np
import torch


def morans_many(expr: np.ndarray, coords: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Compute Moran’s I for all genes at once on a kNN graph.

    expr:  (N, G) float32
    coords:(N, 2) float32 (pixel coords; any consistent scale is fine)
    """
    from scipy.spatial import cKDTree

    if expr.ndim != 2 or coords.ndim != 2 or coords.shape[0] != expr.shape[0]:
        raise ValueError("expr/coords shape mismatch for Moran's I")
    n_cells, n_genes = expr.shape
    if n_cells < 3:
        return np.zeros((n_genes,), dtype=np.float64)

    k_eff = max(1, min(int(k), n_cells - 1))
    tree = cKDTree(coords.astype(np.float32, copy=False))
    idx = tree.query(coords, k=k_eff + 1)[1][:, 1:]  # (N,k)

    vc = expr.astype(np.float32, copy=False) - expr.mean(axis=0, keepdims=True)
    den = (vc**2).sum(axis=0).astype(np.float64)  # (G,)
    # (N,k,G) -> (G,)
    neigh = vc[idx, :]
    num = (vc[:, None, :] * neigh).sum(axis=(0, 1)).astype(np.float64)
    w = float(k_eff * n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = (n_cells / w) * (num / den)
        I[~np.isfinite(I)] = 0.0
        I[den <= 0] = 0.0
    return I


def giotto_rank_scores(expr, coords, k=8):
    """
    Giotto "rank" analog:
      1) rank expression per gene
      2) compute neighbourhood-mean rank on kNN graph
      3) score = corr(centered ranks, centered neigh-mean ranks)
    """
    from scipy.spatial import cKDTree
    from scipy.stats import rankdata

    expr = np.asarray(expr, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if expr.ndim != 2 or coords.ndim != 2 or expr.shape[0] != coords.shape[0]:
        raise ValueError("giotto_rank_scores expects expr (N,G) and coords (N,2)")
    n_cells, n_genes = expr.shape
    if n_cells < 3 or n_genes <= 0:
        return np.zeros((max(n_genes, 0),), dtype=np.float32)

    k_eff = max(1, min(int(k), n_cells - 1))
    tree = cKDTree(coords)
    neigh_idx = tree.query(coords, k=k_eff + 1)[1][:, 1:]  # drop self

    scores = np.zeros(n_genes, dtype=np.float32)
    for g in range(n_genes):
        ranks = rankdata(expr[:, g], method="average").astype(np.float32)
        neigh_mean = ranks[neigh_idx].mean(axis=1)
        r_center = ranks - ranks.mean()
        n_center = neigh_mean - neigh_mean.mean()
        denom = np.sqrt((r_center**2).sum() * (n_center**2).sum())
        if denom > 0:
            scores[g] = float(np.dot(r_center, n_center) / denom)
    return scores


def summarize_gene_pcc_distribution(corr: np.ndarray):
    vals = np.asarray(corr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "median": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "n_genes": 0,
        }
    return {
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "n_genes": int(vals.size),
    }


def format_gene_pcc_triplet(stats: dict):
    if not isinstance(stats, dict):
        return "med=nan max=nan min=nan n=0"
    med = stats.get("median", float("nan"))
    mx = stats.get("max", float("nan"))
    mn = stats.get("min", float("nan"))
    n = int(stats.get("n_genes", 0) or 0)
    return "med={:.4f} max={:.4f} min={:.4f} n={}".format(
        float(med) if med is not None else float("nan"),
        float(mx) if mx is not None else float("nan"),
        float(mn) if mn is not None else float("nan"),
        n,
    )


def log_gene_pcc_epoch(metrics: dict, *, split_tag: str, epoch: int, svg_topk=(20, 50)):
    if not isinstance(metrics, dict):
        return
    dist = metrics.get("gene_pcc_distribution_per_slide") or {}
    if not isinstance(dist, dict) or len(dist) == 0:
        logging.info("%s GenePCC epoch=%d: unavailable", split_tag, int(epoch))
        return
    for sid in sorted(dist):
        sid_stats = dist.get(sid) or {}
        all_s = format_gene_pcc_triplet(sid_stats.get("all", {}))
        parts = [f"ALL({all_s})"]
        for k_svg in svg_topk:
            key = f"svg{int(k_svg)}"
            parts.append(f"SVG{int(k_svg)}({format_gene_pcc_triplet(sid_stats.get(key, {}))})")
        logging.info(
            "%s GenePCC epoch=%d slide=%s %s",
            str(split_tag).upper(),
            int(epoch),
            sid,
            " ".join(parts),
        )


def _gene_variance_weight(expr, mask=None):
    if mask is None:
        var = torch.var(expr.detach(), dim=0, unbiased=False)
    else:
        mask_f = mask.detach().float()
        den = mask_f.sum(dim=0).clamp_min(1.0)
        mean = (expr.detach() * mask_f).sum(dim=0) / den
        var = (((expr.detach() - mean.view(1, -1)) ** 2) * mask_f).sum(dim=0) / den
    var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    mean_var = var[var > 0].mean() if (var > 0).any() else var.new_tensor(1.0)
    return torch.sqrt(var / mean_var.clamp_min(1e-8)).clamp(0.5, 3.0)


def _gene_weighted_mse(pred, target, mask, gene_weight, zero_threshold=0.0, zero_weight=0.1):
    w_zero = pred.new_tensor(float(zero_weight))
    w_one = pred.new_tensor(1.0)
    w = torch.where(target > float(zero_threshold), w_one, w_zero)
    if mask is not None:
        w = w * mask
    w = w * gene_weight.to(device=pred.device, dtype=pred.dtype).view(1, -1)
    return (((pred - target) ** 2) * w).sum() / w.sum().clamp_min(1e-8)


def _genewise_pearson_loss(pred, target, mask=None, gene_weight=None, eps=1e-6):
    if mask is None:
        mask_f = torch.ones_like(pred)
    else:
        mask_f = mask.float()
    valid = mask_f.sum(dim=0)
    keep = valid > 2
    if not keep.any():
        return pred.new_tensor(0.0)
    pred_k = pred[:, keep]
    target_k = target[:, keep]
    mask_k = mask_f[:, keep]
    valid_k = valid[keep].clamp_min(1.0)
    pred_mean = (pred_k * mask_k).sum(dim=0) / valid_k
    target_mean = (target_k * mask_k).sum(dim=0) / valid_k
    pred_c = (pred_k - pred_mean.view(1, -1)) * mask_k
    target_c = (target_k - target_mean.view(1, -1)) * mask_k
    num = (pred_c * target_c).sum(dim=0)
    denom = (
        (pred_c.pow(2).sum(dim=0).clamp_min(eps).sqrt())
        * (target_c.pow(2).sum(dim=0).clamp_min(eps).sqrt())
    ).clamp_min(eps)
    loss_vec = 1.0 - (num / denom).clamp(-1.0, 1.0)
    if gene_weight is not None:
        w = gene_weight.to(device=pred.device, dtype=pred.dtype)[keep]
        return (loss_vec * w).sum() / w.sum().clamp_min(1e-8)
    return loss_vec.mean()


def _graph_edge_contrast_pearson_loss(
    pred_delta,
    target_delta,
    mask,
    edge_index,
    gene_indices=None,
    gene_weight=None,
    max_edges=50000,
    eps=1e-6,
):
    if edge_index is None or edge_index.numel() == 0:
        return pred_delta.new_tensor(0.0)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    n_cells = pred_delta.shape[0]
    valid_edge = (src >= 0) & (src < n_cells) & (dst >= 0) & (dst < n_cells)
    src = src[valid_edge]
    dst = dst[valid_edge]
    if src.numel() < 3:
        return pred_delta.new_tensor(0.0)
    if src.numel() > max_edges:
        step = int(math.ceil(float(src.numel()) / float(max_edges)))
        src = src[::step][:max_edges]
        dst = dst[::step][:max_edges]

    if gene_indices is not None:
        gene_indices = gene_indices.to(device=pred_delta.device, dtype=torch.long)
        gene_indices = gene_indices[
            (gene_indices >= 0) & (gene_indices < pred_delta.shape[1])
        ]
        if gene_indices.numel() == 0:
            return pred_delta.new_tensor(0.0)
        pred_delta = pred_delta.index_select(1, gene_indices)
        target_delta = target_delta.index_select(1, gene_indices)
        if mask is not None:
            mask = mask.index_select(1, gene_indices)
        if gene_weight is not None:
            gene_weight = gene_weight.index_select(0, gene_indices)

    pred_edge = pred_delta[src] - pred_delta[dst]
    target_edge = target_delta[src] - target_delta[dst]
    if mask is None:
        edge_mask = torch.ones_like(pred_edge)
    else:
        edge_mask = (mask[src].float() * mask[dst].float()).to(pred_edge.dtype)
    valid = edge_mask.sum(dim=0)
    keep = valid > 2
    if not keep.any():
        return pred_delta.new_tensor(0.0)

    pred_edge = pred_edge[:, keep]
    target_edge = target_edge[:, keep]
    edge_mask = edge_mask[:, keep]
    valid = valid[keep].clamp_min(1.0)
    pred_mean = (pred_edge * edge_mask).sum(dim=0) / valid
    target_mean = (target_edge * edge_mask).sum(dim=0) / valid
    pred_c = (pred_edge - pred_mean.view(1, -1)) * edge_mask
    target_c = (target_edge - target_mean.view(1, -1)) * edge_mask
    num = (pred_c * target_c).sum(dim=0)
    denom = (
        pred_c.pow(2).sum(dim=0).clamp_min(eps).sqrt()
        * target_c.pow(2).sum(dim=0).clamp_min(eps).sqrt()
    ).clamp_min(eps)
    loss_vec = 1.0 - (num / denom).clamp(-1.0, 1.0)
    if gene_weight is not None:
        w = gene_weight.to(device=pred_delta.device, dtype=pred_delta.dtype)[keep]
        return (loss_vec * w).sum() / w.sum().clamp_min(1e-8)
    return loss_vec.mean()


def graph_residual_loss_terms(
    graph_residual_delta,
    graph_residual_base,
    target_expr,
    expr_mask,
    edge_index=None,
    svg_gene_order=None,
    zero_threshold=0.0,
    zero_weight=0.1,
):
    zero = target_expr.new_tensor(0.0)
    if (
        graph_residual_delta is None
        or graph_residual_base is None
        or graph_residual_delta.shape != target_expr.shape
        or graph_residual_base.shape != target_expr.shape
    ):
        return zero, zero, zero

    graph_residual_target = target_expr - graph_residual_base.detach()
    graph_gene_weight = _gene_variance_weight(target_expr, expr_mask)
    residual_loss = _gene_weighted_mse(
        graph_residual_delta,
        graph_residual_target,
        expr_mask,
        graph_gene_weight,
        zero_threshold=zero_threshold,
        zero_weight=zero_weight,
    )
    graph_pred_aux = graph_residual_base.detach() + graph_residual_delta
    gene_pcc_loss = _genewise_pearson_loss(
        graph_pred_aux,
        target_expr,
        expr_mask,
        graph_gene_weight,
    )
    edge_contrast_loss = zero
    if svg_gene_order is not None and edge_index is not None:
        svg_order = torch.as_tensor(
            svg_gene_order,
            device=target_expr.device,
            dtype=torch.long,
        )
        svg_order = svg_order[(svg_order >= 0) & (svg_order < target_expr.shape[1])]
        svg_idx_50 = svg_order[: min(50, int(svg_order.numel()))]
        svg_idx_20 = svg_order[: min(20, int(svg_order.numel()))]
        edge_contrast_loss = (
            _graph_edge_contrast_pearson_loss(
                graph_residual_delta,
                graph_residual_target,
                expr_mask,
                edge_index,
                gene_indices=svg_idx_50,
                gene_weight=graph_gene_weight,
            )
            + _graph_edge_contrast_pearson_loss(
                graph_residual_delta,
                graph_residual_target,
                expr_mask,
                edge_index,
                gene_indices=svg_idx_20,
                gene_weight=graph_gene_weight,
            )
        )
    return residual_loss, gene_pcc_loss, edge_contrast_loss

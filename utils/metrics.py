"""Metric helpers for training and validation."""

import logging

import numpy as np


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


def pearson_loss(pred, target, eps=1e-6):
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    num = (pred * target).mean(dim=0)
    denom = (
        pred.std(dim=0, unbiased=False) * target.std(dim=0, unbiased=False)
    ).clamp_min(eps)
    corr = num / denom
    return (1.0 - corr).mean()


def masked_pearson(pred, target, mask=None, eps=1e-6):
    """
    Pearson distance (1 - corr) with an optional per-gene mask.
    """
    if mask is None:
        return pearson_loss(pred, target, eps=eps)
    mask = mask.float()
    valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_center = (pred * mask - (pred * mask).sum(dim=1, keepdim=True) / valid)
    targ_center = (target * mask - (target * mask).sum(dim=1, keepdim=True) / valid)
    num = (pred_center * targ_center * mask).sum(dim=1)
    denom = (
        ((pred_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
        * ((targ_center**2) * mask).sum(dim=1).clamp_min(eps).sqrt()
    ).clamp_min(eps)
    corr = num / denom
    return (1.0 - corr).mean()

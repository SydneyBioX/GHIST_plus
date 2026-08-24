"""Hurdle likelihood and deterministic cohort export for sparse expression.

The magnitude location is the signed GHIST+ Eq. 5 reconstruction before its
single final ReLU.  A separate occurrence logit parameterizes q=P(Y>0).
Positive log1p observations follow a Normal distribution truncated to y > 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_hurdle_truncated_gaussian_nll(
    signed_mu: torch.Tensor,
    occurrence_logits: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One masked hurdle-truncated-Gaussian negative log likelihood.

    All tensors use absolute log1p expression (never baseline-subtracted).
    Observed entries are indexed before arithmetic, so masked NaN/Inf values
    cannot contaminate the loss through ``NaN * 0``.
    """

    if signed_mu.shape != target.shape or occurrence_logits.shape != target.shape:
        raise ValueError(
            "signed_mu, occurrence_logits, and target must have identical shapes"
        )
    sigma = sigma.to(device=signed_mu.device, dtype=signed_mu.dtype).clamp_min(1e-4)
    sigma = torch.broadcast_to(sigma, target.shape)
    if mask is None:
        observed = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.shape != target.shape:
            raise ValueError("mask must have the same shape as target")
        observed = mask.to(device=target.device) > 0
    if not observed.any():
        return (
            signed_mu.sum() * 0.0
            + occurrence_logits.sum() * 0.0
            + sigma.sum() * 0.0
        )

    signed_mu = signed_mu[observed]
    occurrence_logits = occurrence_logits[observed]
    target = target[observed]
    sigma = sigma[observed]
    if not torch.isfinite(target).all():
        raise ValueError("observed expression targets must be finite")

    zero_nll = F.softplus(occurrence_logits)  # -log(1-q)
    positive_occurrence_nll = F.softplus(-occurrence_logits)  # -log(q)

    # For mu >= 0, use the direct truncated-Normal NLL.  For mu < 0,
    # rearrange with erfcx to avoid catastrophic cancellation in log Phi.
    nonnegative_mu = signed_mu.clamp_min(0.0)
    standardized_direct = (target - nonnegative_mu) / sigma
    direct_positive_density_nll = (
        0.5 * standardized_direct.square()
        + torch.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
        + torch.special.log_ndtr(nonnegative_mu / sigma)
    )
    d = (-signed_mu / sigma).clamp_min(0.0)
    v = target / sigma
    negative_mu_density_nll = (
        d * v
        + 0.5 * v.square()
        + torch.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
        + math.log(0.5)
        + torch.log(torch.special.erfcx(d / math.sqrt(2.0)))
    )
    positive_density_nll = torch.where(
        signed_mu < 0,
        negative_mu_density_nll,
        direct_positive_density_nll,
    )
    positive_nll = positive_occurrence_nll + positive_density_nll
    return torch.where(target > 0, positive_nll, zero_nll).mean()


def deterministic_threshold_control(
    signed_mu: torch.Tensor,
    occurrence_logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Threshold control only; this is not the saved hurdle prediction."""

    if signed_mu.shape != occurrence_logits.shape:
        raise ValueError("signed_mu and occurrence_logits must have identical shapes")
    gate = torch.sigmoid(occurrence_logits) >= float(threshold)
    return gate.to(signed_mu.dtype) * F.relu(signed_mu)


def deterministic_hurdle_prediction(
    signed_mu: torch.Tensor,
    occurrence_logits: torch.Tensor,
) -> torch.Tensor:
    """Return one deterministic matrix for a complete unique-cell cohort.

    For gene g, ``K_g=round(sum_i sigmoid(logit_ig))``.  The K cells with the
    largest q are positive.  This uses no target labels and no tuned threshold.
    Call this exactly once per complete slide/cohort, never per minibatch.
    """

    if signed_mu.shape != occurrence_logits.shape:
        raise ValueError("signed_mu and occurrence_logits must have identical shapes")
    if signed_mu.ndim != 2:
        raise ValueError("cohort prediction expects [n_cells, n_genes] tensors")
    n_cells, n_genes = signed_mu.shape
    if n_cells == 0:
        return F.relu(signed_mu)

    probability = torch.sigmoid(occurrence_logits)
    k_positive = probability.sum(dim=0).round().long().clamp(0, n_cells)
    # Stable sorting makes tied q values use input cell order deterministically.
    order = torch.argsort(probability, dim=0, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(n_cells, device=order.device).view(-1, 1)
    ranks.scatter_(0, order, rank_values.expand(n_cells, n_genes))
    gate = ranks < k_positive.view(1, -1)
    return gate.to(signed_mu.dtype) * F.relu(signed_mu)


def hurdle_reconstruction_loss_from_model(
    model,
    target_model_scale: torch.Tensor,
    mask: torch.Tensor | None,
    expr_scale: float,
) -> torch.Tensor:
    """Compute the sole expression reconstruction term from model auxiliaries."""

    aux = getattr(model, "last_aux_losses", {}) or {}
    signed_mu = aux.get("hurdle_signed_mu")
    occurrence_logits = aux.get("hurdle_occurrence_logits")
    if signed_mu is None or occurrence_logits is None:
        raise RuntimeError("HurdleFramework auxiliaries are missing")
    scale = float(expr_scale)
    if scale <= 0:
        raise ValueError("expr_scale must be positive")
    return masked_hurdle_truncated_gaussian_nll(
        signed_mu / scale,
        occurrence_logits,
        target_model_scale / scale,
        model.hurdle_sigma(),
        mask,
    )

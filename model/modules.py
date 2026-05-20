"""Reusable neural-network modules for GHIST+."""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as _spectral_norm
from torch.amp import autocast as amp_autocast
from torch.utils.checkpoint import checkpoint


class VQCodebook(nn.Module):
    """
    Vector-Quantization codebook with straight-through estimator.

    - Assigns each input vector to the nearest code (cosine or L2).
    - Returns quantized vectors and the classic codebook + commitment losses.
    - Designed for minimal disruption: single forward call, numerically safe.
    """
    def __init__(self,
                 n_codes: int,
                 dim: int,
                 beta: float = 0.25,
                 use_cosine: bool = True,
                 eps: float = 1e-6,
                 ema_decay: float = 0.99,
                 ema_eps: float = 1e-5):
        super().__init__()
        assert n_codes >= 2 and dim >= 1
        self.n_codes = int(n_codes)
        self.dim = int(dim)
        self.beta = float(beta)
        self.use_cosine = bool(use_cosine)
        self.eps = float(eps)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)

        # Codebook: K x D, Xavier init on unit sphere for cosine stability
        E = torch.empty(self.n_codes, self.dim)
        nn.init.xavier_uniform_(E)
        E = F.normalize(E, dim=1)
        self.codebook = nn.Parameter(E)
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_embeddings", torch.zeros_like(E))

    @torch.no_grad()
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D)
        Returns: z_q (N, D), loss (scalar), indices (N,), stats (dict)
        """
        if x.numel() == 0:
            z = x
            loss = x.new_tensor(0.0)
            idx = x.new_zeros((0,), dtype=torch.long)
            return z, loss, idx, {}

        N, D = x.shape
        assert D == self.dim

        E = self.codebook
        if self.use_cosine:
            x_n = self._norm(x.detach())
            E_n = self._norm(E)
            # Cosine similarity: pick max.
            sim = x_n @ E_n.t()                    # (N, K)
            idx = sim.argmax(dim=1)                # (N,)
        else:
            # Squared L2 distance: pick min.
            x2 = (x.detach() ** 2).sum(dim=1, keepdim=True)          # (N,1)
            e2 = (E ** 2).sum(dim=1).unsqueeze(0)                    # (1,K)
            dist2 = x2 - 2.0 * (x.detach() @ E.t()) + e2             # (N,K)
            idx = dist2.argmin(dim=1)                                # (N,)

        # Gather quantized vectors
        z_e = F.embedding(idx, E)                                     # (N,D)

        # Straight-through estimator: identity on forward, copy grad to x
        z_q = x + (z_e - x).detach()

        # Losses (classic VQ-VAE)
        # codebook loss (pull codes to data)
        codebook_loss = F.mse_loss(z_e, x.detach())
        # commitment loss (commit encoder to chosen code)
        commit_loss = F.mse_loss(x, z_e.detach())
        loss = codebook_loss + self.beta * commit_loss

        # numeric safety
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

        if self.training:
            one_hot = F.one_hot(idx, num_classes=self.n_codes).to(x.dtype)
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.t() @ x.detach()

            self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + cluster_size * (1.0 - self.ema_decay)
            self.ema_embeddings = self.ema_embeddings * self.ema_decay + embed_sum * (1.0 - self.ema_decay)

            n = self.ema_cluster_size.sum()
            cluster_size_normalised = ((self.ema_cluster_size + self.ema_eps) /
                                       (n + self.n_codes * self.ema_eps)) * n
            embed_normalised = self.ema_embeddings / cluster_size_normalised.unsqueeze(1).clamp_min(1.0)
            self.codebook.data.copy_(F.normalize(embed_normalised, dim=1) if self.use_cosine else embed_normalised)

        stats = {
            "vq_codebook_loss": codebook_loss.detach(),
            "vq_commit_loss": commit_loss.detach(),
            "vq_beta": torch.tensor(self.beta, device=x.device),
            "vq_active_codes": idx.unique().numel(),
        }
        return z_q, loss, idx, stats

class EdgeCondMixer(nn.Module):
    """
    Edge-Conditioned Residual Mixer (ECRM).
    Learns a scalar gate w_ij from relative position, cell-type difference,
    and feature difference for the k nearest neighbours, then mixes messages
    by mean-field.
    Robustified for AMP/fp16: computes pairwise features and MLP in fp32,
    clamps/cleans NaN/Inf, and guards all-zero rows.
    """
    def __init__(self, hidden_size: int, k: int = 12):
        super().__init__()
        self.k = k
        self.hidden_size = hidden_size
        # --- Adaptive-k knobs ---
        self.k_target = k         # base target degree
        self.k_min = 8            # per-cell lower bound
        self.k_max = 24           # per-cell upper bound
        self.density_gamma = 0.5  # density-to-k strength

        # Anisotropy caps; training ramps from 0 to these values.
        self.eta_max = 0.9
        self.gamma_perp_max = 0.5
        self.mlp = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        # Richer edge encoder for explicit edge-index mode.
        self.edge_mlp = nn.Sequential(
            nn.Linear(9, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        self.edge_norm = nn.LayerNorm(hidden_size)
        self.beta = nn.Parameter(torch.tensor(-0.5))   # residual gate
        self.debug_ecrm: bool = False
        self.softmax_temp = 0.7
        self.depth = 1
        self.edge_dropout = 0.0
        self.message_dropout = 0.0
        self.trust_floor = 0.1   # minimum neighbour/row trust when entropy is high
        self.trust_scale = 0.9   # scales (1 - entropy) contribution
        self.k_target_per_class = None
        self.k_min_per_class = None
        self.k_max_per_class = None
        self.density_gamma_per_class = None
        self.trust_floor_per_class = None
        self.trust_scale_per_class = None
        self.ct_conf_min = 0.0
        self.ct_same_type_only = False
        
        # debug throttle: print at most once every 5 s
        self._dbg_last = 0.0

    def _apply_edge_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm safely across mixed feature widths.
        ECRM is reused on embeddings (D=hidden), CT ref weights (D=n_ref),
        and expression residuals (D=n_genes), so fallback to non-affine
        layer-norm when the configured affine norm shape does not match.
        """
        if isinstance(self.edge_norm, nn.LayerNorm):
            norm_shape = self.edge_norm.normalized_shape
            if isinstance(norm_shape, (tuple, list)) and len(norm_shape) == 1:
                norm_dim = int(norm_shape[0])
            else:
                norm_dim = int(norm_shape)
            if x.shape[-1] == norm_dim:
                return self.edge_norm(x)
        return F.layer_norm(x, (x.shape[-1],))

    def _forward_dense(self,
                h: torch.Tensor,          # (N,D)
                coords: torch.Tensor,     # (N,2)
                ct_prob: torch.Tensor,    # (N,C), probabilities or dummy (N,1)
                immune_gate: torch.Tensor | None = None,
                invasive_gate: torch.Tensor | None = None,
                expr_pred: torch.Tensor | None = None,
                gate_h: torch.Tensor | None = None):
        N, D = h.shape
        if N < 2:
            return h

        if immune_gate is not None:
            immune_gate = immune_gate.to(h.device).clamp(0.0, 1.0)
        if invasive_gate is not None:
            invasive_gate = invasive_gate.to(h.device).clamp(0.0, 1.0)

        # Always have at least one neighbour (besides self if present)
        k_eff = max(1, min(self.k, N - 1))

        # Numerically safe path: do heavy ops in fp32 on a sparse top-k graph.
        with amp_autocast(h.device.type, enabled=False):
            h_native = h  # keep AMP dtype (fp16/bf16) to halve memory on (N,k,D) tensors
            coords32 = coords.float()
            ct32     = ct_prob.float()
            gate_h_native = h_native
            if isinstance(gate_h, torch.Tensor) and gate_h.ndim == 2 and gate_h.shape[0] == N:
                gate_h_native = gate_h.to(h_native.device)

            def _mix_param(prob_matrix, param_vec, default_val):
                if param_vec is None or param_vec.numel() != prob_matrix.size(1):
                    return prob_matrix.new_full((prob_matrix.size(0),), float(default_val))
                return prob_matrix @ param_vec.to(prob_matrix.device)

            k_target_row = _mix_param(ct32, self.k_target_per_class, self.k_target)
            k_min_row = _mix_param(ct32, self.k_min_per_class, self.k_min)
            k_max_row = _mix_param(ct32, self.k_max_per_class, self.k_max)
            density_gamma_row = _mix_param(ct32, self.density_gamma_per_class, self.density_gamma)
            r = float(getattr(self, "epoch_frac", 1.0))  # 0-to-1 schedule set each epoch

            # pairwise squared distances for kNN (computed once)
            # pairwise squared distances for kNN (computed once; no grad to save memory)
            with torch.no_grad():
                dxy_full  = coords32.unsqueeze(1) - coords32.unsqueeze(0)               # (N,N,2)
                dist_full = (dxy_full ** 2).sum(-1)
                dist_full = torch.nan_to_num(dist_full, nan=float('inf'),
                                            posinf=float('inf'), neginf=float('inf'))
                dist_full.fill_diagonal_(float('inf'))  # never pick self as neighbour

                # --- compute per-row Adaptive-k based on local density ---------------------
                k_target_row = k_target_row.clamp(min=1.0, max=N - 1)
                k_min_row = k_min_row.clamp(min=1.0, max=N - 1)
                k_max_row = k_max_row.clamp(min=1.0, max=N - 1)
                k_max_eff = int(torch.clamp(k_max_row.max(), min=1.0, max=N - 1).round().item()) if N > 1 else 1
                k_max_eff = max(1, min(k_max_eff, N - 1))

                # Take up to k_max neighbours once; rows will choose k_i <= k_max.
                nbr = dist_full.topk(k_max_eff, largest=False).indices        # (N,k_max)

                ar = torch.arange(N, device=dist_full.device)
                k_base_idx = k_target_row.round().long().clamp(min=1, max=k_max_eff)
                d_k = dist_full[ar, nbr[ar, k_base_idx - 1]]                   # (N,)
                d_med = torch.median(d_k)
                ratio = (d_med / (d_k + 1e-6)).clamp(0.5, 2.0)                # dense<1, sparse>1
            # leave the big matrices out of autograd
            del dxy_full, dist_full

            density_gamma_eff = (density_gamma_row.clamp(min=0.05) * r)
            k_i = (k_target_row * (ratio ** density_gamma_eff)).round().to(torch.long)  # (N,)
            # ensure per-row k_i respects min/max bounds
            k_min_clamped = k_min_row.round().long().clamp(min=1, max=k_max_eff)
            k_max_clamped = k_max_row.round().long().clamp(min=1, max=k_max_eff)
            k_i = k_i.clamp(min=1, max=k_max_eff)
            k_i = torch.maximum(k_i, k_min_clamped)
            k_i = torch.minimum(k_i, k_max_clamped)

            # Gather only neighbour tensors, using sparse top-k per row.
            dxy_k = coords32[nbr] - coords32.unsqueeze(1)                  # (N,k,2)
            dct_k = (ct32[nbr] - ct32.unsqueeze(1)).norm(dim=-1, keepdim=True)  # (N,k,1)
            dh_k  = (gate_h_native[nbr] - gate_h_native.unsqueeze(1)).norm(dim=-1, keepdim=True).float()   # (N,k,1)

            expr_sim_k = None
            if isinstance(expr_pred, torch.Tensor) and expr_pred.ndim == 2 and expr_pred.shape[0] == N:
                expr_feat = expr_pred.float()
                if expr_feat.shape[1] > 0:
                    expr_feat = F.normalize(expr_feat, dim=-1, eps=1e-6)
                    expr_sim_k = (expr_feat[nbr] * expr_feat.unsqueeze(1)).sum(dim=-1, keepdim=True)
            if expr_sim_k is None:
                expr_sim_k = torch.zeros((N, k_max_eff, 1), device=coords32.device, dtype=coords32.dtype)

            edge_k = torch.cat([dxy_k, dct_k, dh_k, expr_sim_k], dim=-1)               # (N,k,5)
            edge_k = torch.nan_to_num(edge_k, nan=0.0, posinf=1e6, neginf=-1e6)

            # Raw compatibility logits followed by temperature softmax.
            s_k = self.mlp(edge_k).squeeze(-1)                             # (N,k)
            std_row = s_k.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
            s_k = (s_k - s_k.mean(dim=1, keepdim=True)) / std_row
            temp = getattr(self, "softmax_temp", 0.7)
            s_k = s_k / max(temp, 1e-6)
            s_k = torch.nan_to_num(s_k, nan=0.0, posinf=30.0, neginf=-30.0)
            w_k = F.softmax(s_k, dim=1).to(h_native.dtype)                 # match h dtype to save mem

            # Per-row keep mask via k_i (False = keep).
            pos = torch.arange(k_max_eff, device=h.device).view(1, -1).expand(N, -1)
            keep_mask = pos < k_i.view(-1, 1)                               # (N,k)
            w_k = w_k.masked_fill(~keep_mask, 0.0)

            # Optionally forbid cross-slide mixing if the caller set _slide_ids.
            slide_ids = getattr(self, "_slide_ids", None)
            _ = self._slide_ids if hasattr(self, "_slide_ids") else None    # touch attribute
            if slide_ids is not None and isinstance(slide_ids, torch.Tensor) and slide_ids.numel() == N:
                si = slide_ids.view(N, 1)
                sj = slide_ids[nbr]
                cross = (si != sj)
                w_k = w_k.masked_fill(cross, 0.0)

            # Tissue-anisotropic reweighting: directional mixing along fibres.
            # local principal dir from neighbour coords (per row i)
            C = torch.einsum('nka,nkb->nab', dxy_k, dxy_k) / (dxy_k.size(1) + 1e-6)  # (N,2,2)
            evals, evecs = torch.linalg.eigh(C)                                       # ascending
            u = evecs[..., 1]                                                         # (N,2)

            # anisotropy scalar in [0,1]
            an = (evals[..., 1] - evals[..., 0]) / (evals.sum(-1) + 1e-6)             # (N,)
            an = an.clamp(0.0, 1.0)

            # Alignment of each edge i->j with u_i (absolute cosine in [0,1]).
            dir_ik = F.normalize(dxy_k, dim=-1, eps=1e-6)                              # (N,k,2)
            align_k = (dir_ik * u.unsqueeze(1)).sum(-1).abs()                          # (N,k)

            # Boost along fibres, damp across; strength scales with anisotropy
            eta_max        = getattr(self, "eta_max", 0.9)
            gamma_perp_max = getattr(self, "gamma_perp_max", 0.5)
            eta        = eta_max * r
            gamma_perp = gamma_perp_max * r
            w_k = w_k * (1.0 + eta * an.view(-1,1) * align_k)
            w_k = w_k * torch.exp(-gamma_perp * an.view(-1,1) * (1.0 - align_k))
            if immune_gate is not None or invasive_gate is not None:
                if immune_gate is not None:
                    w_k = w_k * (1.0 - 0.4 * immune_gate.view(-1, 1))
                if invasive_gate is not None:
                    w_k = w_k * (1.0 + 0.4 * invasive_gate.view(-1, 1))
            row_sum_pre = w_k.sum(dim=1, keepdim=True).clamp_min(1e-6)
            w_k = w_k / row_sum_pre

            # --- Trust-weighted neighbours (down-weight uncertain neighbour sources) ---
            nC = int(ct32.size(1))
            ct_label = None
            if nC > 1:
                p = ct32.clamp_min(1e-6)                                  # (N,nC)
                p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)
                H = -(p * p.log()).sum(dim=1) / math.log(nC)              # (N,) entropy ∈ [0,1]
                conf_raw = (1.0 - H).clamp(0.0, 1.0)                      # (N,)
                ct_label = p.argmax(dim=1)
            else:
                conf_raw = torch.ones(N, device=w_k.device, dtype=w_k.dtype)

            trust_floor_row = _mix_param(ct32, self.trust_floor_per_class, self.trust_floor).clamp(0.0, 1.0)
            trust_scale_row = _mix_param(ct32, self.trust_scale_per_class, self.trust_scale).clamp(0.0, 2.0)
            if immune_gate is not None:
                trust_floor_row = (trust_floor_row * (1.0 - 0.3 * immune_gate)).clamp(0.0, 1.0)
                trust_scale_row = (trust_scale_row * (1.0 - 0.2 * immune_gate)).clamp(0.0, 2.0)
            if invasive_gate is not None:
                trust_floor_row = (
                    trust_floor_row + (1.0 - trust_floor_row) * 0.3 * invasive_gate
                ).clamp(0.0, 1.0)
                trust_scale_row = (trust_scale_row * (1.0 + 0.3 * invasive_gate)).clamp(0.0, 2.0)
            trust_scale_eff = trust_scale_row * r
            conf = (trust_floor_row + trust_scale_eff * conf_raw).clamp(0.0, 1.0)    # (N,)
            if immune_gate is not None:
                conf = conf * (1.0 - 0.25 * immune_gate)
            if invasive_gate is not None:
                conf = conf + (1.0 - conf) * 0.25 * invasive_gate
            conf_nbr = conf[nbr]                                                      # (N,k)
            w_k = w_k * conf_nbr

            ct_conf_min = float(getattr(self, "ct_conf_min", 0.0))
            ct_same_type = bool(getattr(self, "ct_same_type_only", False))
            if (ct_conf_min > 0.0 or ct_same_type) and nC > 1:
                mask = None
                if ct_conf_min > 0.0:
                    conf_mask = conf_raw >= ct_conf_min
                    mask = conf_mask.unsqueeze(1) & conf_mask[nbr]
                if ct_same_type and ct_label is not None:
                    same_mask = ct_label[nbr] == ct_label.view(-1, 1)
                    mask = same_mask if mask is None else (mask & same_mask)
                if mask is not None:
                    w_k = w_k * mask.to(w_k.dtype)

            # Remove self messages if present among top-k
            idx = torch.arange(N, device=h.device).view(-1, 1)
            self_mask = (nbr == idx)
            w_k = w_k.masked_fill(self_mask, 0.0)

            # mean-field aggregation on sparse neighbours
            row_sum = w_k.sum(dim=1, keepdim=True)
            valid = row_sum > 1e-6
            row_sum = row_sum.clamp_min(1e-6)
            w_norm = w_k / row_sum
            h_nbr = h_native[nbr]                                                     # (N,k,D)
            m = (w_norm.unsqueeze(-1) * h_nbr).sum(dim=1)                             # (N,D)

            beta_base = self.beta.sigmoid().to(h_native.dtype)
            # Down-weight mixing for uncertain target cells (row-wise trust).
            beta_vec  = beta_base * conf.view(-1, 1).to(h_native.dtype)                # (N,1)
            beta_vec = beta_vec * valid.to(beta_vec.dtype)
            out = h_native + beta_vec * (m - h_native)

        # --- optional debug print (throttled) -------------------------------
        if self.training and getattr(self, "debug_ecrm", False):
            now = time.time()
            if now - self._dbg_last > 5.0:
                # Stats based on sparse weights
                w_safe = torch.nan_to_num(w_k.detach())
                row_w_std = w_safe.std(dim=1, correction=0).mean().item()
                active_std = w_safe[w_safe > 0].std(correction=0).item() if (w_safe > 0).any() else 0.0
                dx_max = dxy_k[..., 0].abs().max().item()
                dy_max = dxy_k[..., 1].abs().max().item()
                k_mean = float(k_i.float().mean().item())
                print(f"[ecrm] row_w_std {row_w_std:.3f} "
                      f"active_std {active_std:.3f} "
                      f"k_mean {k_mean:.1f} "
                      f"dx_max {dx_max:.1f} dy_max {dy_max:.1f}")
                self._dbg_last = now

        return out.to(h.dtype)

    def _segment_softmax(self, src: torch.Tensor, scores: torch.Tensor, n_nodes: int) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        max_per = torch.full(
            (n_nodes,),
            -float("inf"),
            device=scores.device,
            dtype=scores.dtype,
        )
        max_per.scatter_reduce_(0, src, scores, reduce="amax", include_self=True)
        logits = scores - max_per[src]
        exp = torch.exp(logits)
        denom = torch.zeros((n_nodes,), device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(0, src, exp)
        return exp / denom[src].clamp_min(1e-6)

    def _compute_ct_conf(self, ct32: torch.Tensor):
        if ct32.ndim == 2 and ct32.shape[1] > 1:
            ct_prob_soft = ct32.clamp_min(1e-6)
            ct_prob_soft = ct_prob_soft / ct_prob_soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
            ct_entropy = -(ct_prob_soft * torch.log(ct_prob_soft.clamp_min(1e-8))).sum(dim=1, keepdim=True)
            max_ent = math.log(ct_prob_soft.shape[1])
            ct_conf = (1.0 - ct_entropy / max(max_ent, 1e-6)).clamp(0.0, 1.0)
            ct_label = ct_prob_soft.argmax(dim=1)
            return ct_prob_soft, ct_conf, ct_label
        ct_prob_soft = None
        ct_conf = torch.ones((ct32.shape[0], 1), device=ct32.device, dtype=ct32.dtype)
        ct_label = None
        return ct_prob_soft, ct_conf, ct_label

    def _forward_graph_edges(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        ct_prob: torch.Tensor,
        edge_index: torch.Tensor,
        immune_gate: torch.Tensor | None = None,
        invasive_gate: torch.Tensor | None = None,
        expr_pred: torch.Tensor | None = None,
        gate_h: torch.Tensor | None = None,
    ):
        if edge_index is None or edge_index.numel() == 0:
            return h
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0 or dst.numel() == 0:
            return h

        if immune_gate is not None:
            immune_gate = immune_gate.to(h.device).clamp(0.0, 1.0)
        if invasive_gate is not None:
            invasive_gate = invasive_gate.to(h.device).clamp(0.0, 1.0)

        n_nodes = int(h.shape[0])
        valid_idx = (
            (src >= 0)
            & (src < n_nodes)
            & (dst >= 0)
            & (dst < n_nodes)
        )
        src = src[valid_idx]
        dst = dst[valid_idx]
        if src.numel() == 0:
            return h

        coords32 = coords.float()
        ct32 = ct_prob.float()
        ct_prob_soft, ct_conf, ct_label = self._compute_ct_conf(ct32)
        ct_conf_min = float(getattr(self, "ct_conf_min", 0.0))
        ct_same_type = bool(getattr(self, "ct_same_type_only", False))
        edge_dropout = float(getattr(self, "edge_dropout", 0.0))
        msg_dropout = float(getattr(self, "message_dropout", 0.0))
        depth = max(1, int(getattr(self, "depth", 1)))
        temp = max(float(getattr(self, "softmax_temp", 0.7)), 1e-6)

        slide_ids = getattr(self, "_slide_ids", None)
        if not (isinstance(slide_ids, torch.Tensor) and slide_ids.numel() == n_nodes):
            slide_ids = None

        out = h
        for _ in range(depth):
            h_native = out
            gate_h_native = h_native
            if (
                isinstance(gate_h, torch.Tensor)
                and gate_h.ndim == 2
                and gate_h.shape[0] == h_native.shape[0]
            ):
                gate_h_native = gate_h.to(h_native.device)

            dxy = coords32[dst] - coords32[src]
            dist = torch.norm(dxy, dim=-1, keepdim=True).clamp_min(1e-6)
            dist_scale = dist.detach().median().clamp_min(1e-3)
            dx = dxy[:, 0:1] / dist_scale
            dy = dxy[:, 1:2] / dist_scale
            dist_n = dist / dist_scale
            dct = (ct32[dst] - ct32[src]).norm(dim=-1, keepdim=True)
            dh = (gate_h_native[dst] - gate_h_native[src]).norm(dim=-1, keepdim=True).float()
            dh_scale = dh.detach().median().clamp_min(1e-3)
            dh_n = dh / dh_scale

            if (
                isinstance(expr_pred, torch.Tensor)
                and expr_pred.ndim == 2
                and expr_pred.shape[0] == h_native.shape[0]
                and expr_pred.shape[1] > 0
            ):
                expr_feat = F.normalize(expr_pred.float(), dim=-1, eps=1e-6)
                expr_sim = (expr_feat[dst] * expr_feat[src]).sum(dim=-1, keepdim=True)
            else:
                expr_sim = torch.zeros_like(dist_n)

            if ct_label is not None:
                same_type = (ct_label[src] == ct_label[dst]).float().unsqueeze(-1)
            else:
                same_type = torch.ones_like(dist_n)

            conf_src = ct_conf[src]
            conf_dst = ct_conf[dst]

            edge_feat = torch.cat(
                [dx, dy, dist_n, dct, same_type, dh_n, expr_sim, conf_src, conf_dst],
                dim=-1,
            )
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=1e4, neginf=-1e4)

            scores = self.edge_mlp(edge_feat).squeeze(-1)
            scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)
            scores = scores + torch.log(conf_dst.view(-1).clamp_min(1e-4))

            valid_edge = torch.ones_like(scores, dtype=torch.bool)
            if ct_conf_min > 0.0:
                conf_ok = ct_conf.view(-1) >= ct_conf_min
                valid_edge &= conf_ok[src] & conf_ok[dst]
            if ct_same_type and ct_label is not None:
                valid_edge &= (ct_label[src] == ct_label[dst])
            if slide_ids is not None:
                valid_edge &= (slide_ids[src] == slide_ids[dst])
            if self.training and edge_dropout > 0.0:
                valid_edge &= (torch.rand_like(scores) >= edge_dropout)

            if immune_gate is not None:
                scores = scores + torch.log(
                    (1.0 - 0.3 * immune_gate[src]).clamp_min(1e-4)
                )
            if invasive_gate is not None:
                scores = scores + torch.log(
                    (1.0 + 0.3 * invasive_gate[src]).clamp_min(1e-4)
                )

            masked_scores = torch.where(
                valid_edge,
                scores / temp,
                torch.full_like(scores, -30.0),
            )
            weights = self._segment_softmax(src, masked_scores, n_nodes=n_nodes)
            weights = weights * valid_edge.to(weights.dtype)
            row_sum = torch.zeros((n_nodes,), device=weights.device, dtype=weights.dtype)
            row_sum.scatter_add_(0, src, weights)
            valid_row = row_sum > 1e-6
            weights = weights / row_sum[src].clamp_min(1e-6)

            msg = h_native[dst]
            if self.training and msg_dropout > 0.0:
                msg = F.dropout(msg, p=msg_dropout, training=True)
            agg = torch.zeros_like(h_native)
            agg.index_add_(0, src, weights.unsqueeze(-1) * msg)
            agg = torch.where(valid_row.unsqueeze(-1), agg, h_native)

            beta_vec = torch.sigmoid(self.beta).to(h_native.dtype).view(1, 1).expand(n_nodes, 1)
            if immune_gate is not None:
                beta_vec = beta_vec * (1.0 - 0.2 * immune_gate.view(-1, 1).to(beta_vec.dtype))
            if invasive_gate is not None:
                beta_vec = beta_vec * (1.0 + 0.2 * invasive_gate.view(-1, 1).to(beta_vec.dtype))

            out = h_native + beta_vec * (agg - h_native)
            out = self._apply_edge_norm(out)

        return out.to(h.dtype)

    def forward(self,
                h: torch.Tensor,          # (N,D)
                coords: torch.Tensor,     # (N,2)
                ct_prob: torch.Tensor,    # (N,C) or (N,1)
                expr_pred: torch.Tensor | None = None,
                gate_h: torch.Tensor | None = None,
                immune_gate: torch.Tensor | None = None,
                invasive_gate: torch.Tensor | None = None,
                edge_index: torch.Tensor | None = None,
                patch_ids: torch.Tensor | None = None):
        """
        Wrapper that runs the dense mixer per-patch if `self._patch_ids` is set.
        This caps N per call (e.g., <= max_cells_per_patch) without changing the
        per-patch neighbour logic.
        """
        N = h.shape[0]
        patch_ids_attr = patch_ids
        if patch_ids_attr is None:
            patch_ids_attr = getattr(self, "_patch_ids", None)

        expr_pred_tensor = expr_pred
        if (
            not isinstance(expr_pred_tensor, torch.Tensor)
            or expr_pred_tensor.ndim != 2
            or expr_pred_tensor.shape[0] != N
        ):
            # Always pass a tensor through checkpointed paths; a constant vector
            # reduces to the legacy behaviour (the extra feature becomes 0).
            expr_pred_tensor = h.new_zeros((N, 1))
        gate_h_tensor = gate_h
        if (
            not isinstance(gate_h_tensor, torch.Tensor)
            or gate_h_tensor.ndim != 2
            or gate_h_tensor.shape[0] != N
        ):
            gate_h_tensor = h

        # Use activation checkpointing while training (mathematically exact, lower memory)
        use_ckpt = self.training and getattr(self, "use_checkpoint", True)

        if edge_index is not None and isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            gate_imm = immune_gate
            gate_inv = invasive_gate
            if use_ckpt:
                return checkpoint(
                    lambda a, b, c, d, e, f, g, i: self._forward_graph_edges(a, b, c, d, e, f, g, i),
                    h, coords, ct_prob, edge_index, gate_imm, gate_inv, expr_pred_tensor, gate_h_tensor,
                    use_reentrant=False,
                )
            return self._forward_graph_edges(
                h,
                coords,
                ct_prob,
                edge_index,
                gate_imm,
                gate_inv,
                expr_pred_tensor,
                gate_h_tensor,
            )

        # If patch IDs are provided and match N, mix within each patch separately
        if (isinstance(patch_ids_attr, torch.Tensor)
            and patch_ids_attr.numel() == N
            and patch_ids_attr.dtype in (torch.long, torch.int64, torch.int32)):
            out = h.new_empty(h.shape)
            # unique without sort to avoid reordering rows
            uniq = torch.unique(patch_ids_attr, sorted=False)
            for pid in uniq:
                idx = (patch_ids_attr == pid).nonzero(as_tuple=False).squeeze(1)
                # 0/1-node groups: identity.
                if idx.numel() <= 1:
                    out[idx] = h[idx]
                    continue
                gate_imm = immune_gate[idx] if immune_gate is not None else None
                gate_inv = invasive_gate[idx] if invasive_gate is not None else None
                expr_sub = expr_pred_tensor[idx]
                gate_sub = gate_h_tensor[idx]
                if use_ckpt:
                    out[idx] = checkpoint(
                        lambda a, b, c, d, e, f, g: self._forward_dense(a, b, c, d, e, f, g),
                        h[idx], coords[idx], ct_prob[idx], gate_imm, gate_inv, expr_sub, gate_sub,
                        use_reentrant=False,
                    )
                else:
                    out[idx] = self._forward_dense(
                        h[idx],
                        coords[idx],
                        ct_prob[idx],
                        gate_imm,
                        gate_inv,
                        expr_sub,
                        gate_sub,
                    )
            return out

        # Fallback: original dense behaviour on the full batch
        if use_ckpt:
            return checkpoint(
                lambda a, b, c, d, e, f, g: self._forward_dense(a, b, c, d, e, f, g),
                h, coords, ct_prob, immune_gate, invasive_gate, expr_pred_tensor, gate_h_tensor,
                use_reentrant=False,
            )
        return self._forward_dense(
            h,
            coords,
            ct_prob,
            immune_gate,
            invasive_gate,
            expr_pred_tensor,
            gate_h_tensor,
        )

    def set_class_params(self,
                         k_target=None,
                         k_min=None,
                         k_max=None,
                         density_gamma=None,
                         trust_floor=None,
                         trust_scale=None):
        """
        Configure per-class neighbourhood behaviour. Each argument should be an
        iterable of length equal to the number of cell types (same order that the
        model uses). Any parameter left as None falls back to the global scalar.
        """
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(list(x), dtype=torch.float32)

        self.k_target_per_class = _to_tensor(k_target)
        self.k_min_per_class = _to_tensor(k_min)
        self.k_max_per_class = _to_tensor(k_max)
        self.density_gamma_per_class = _to_tensor(density_gamma)
        self.trust_floor_per_class = _to_tensor(trust_floor)
        self.trust_scale_per_class = _to_tensor(trust_scale)

class CrossAttention(nn.Module):
    def __init__(self, n_classes, n_genes, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_transform = nn.Linear(n_classes, embed_dim)
        self.key_transform = nn.Linear(n_genes, embed_dim)
        self.value_transform = nn.Linear(n_genes, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear_out = nn.Linear(embed_dim, n_genes)

    def forward(self, comp, expr):
        # Input tensor of shape (batch_size, n_classes/n_genes)
        comp = comp.unsqueeze(0)
        expr = expr.unsqueeze(0)

        query = self.query_transform(comp)
        key = self.key_transform(expr)
        value = self.value_transform(expr)

        output, _ = self.attention(query, key, value)
        output = output.squeeze(0)
        output = self.linear_out(output)

        return output


class CompExprRefinerMTA(nn.Module):
    """
    Composition-aware refinement with true multi-token attention.
    Query tokens are produced from expression; key/value tokens come from
    composition-weighted class embeddings.
    """

    def __init__(
        self,
        n_classes: int,
        n_genes: int,
        embed_dim: int,
        num_heads: int = 8,
        num_query_tokens: int = 8,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_genes = int(n_genes)
        self.embed_dim = int(embed_dim)
        self.num_query_tokens = int(max(1, num_query_tokens))

        self.expr_to_queries = nn.Linear(
            self.n_genes, self.num_query_tokens * self.embed_dim
        )
        self.class_embed = nn.Parameter(torch.empty(self.n_classes, self.embed_dim))
        nn.init.xavier_uniform_(self.class_embed)

        self.query_ln = nn.LayerNorm(self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_ln = nn.LayerNorm(self.embed_dim)

        self.attn = nn.MultiheadAttention(
            self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.n_genes),
        )

    def forward(self, comp: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        if comp.numel() == 0 or expr.numel() == 0:
            return torch.zeros_like(expr)
        n = expr.shape[0]
        q = self.expr_to_queries(expr).view(n, self.num_query_tokens, self.embed_dim)
        q = self.query_ln(q)

        cls_tok = comp.unsqueeze(-1) * self.class_embed.unsqueeze(0)  # (N,C,D)
        k = self.kv_ln(self.key_proj(cls_tok))
        v = self.value_proj(cls_tok)

        out, _ = self.attn(q, k, v, need_weights=False)
        pooled = out.mean(dim=1)
        return self.out(pooled)

class Embed(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Embed, self).__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        layers_mlp = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ]
        self.mlp = nn.Sequential(*layers_mlp)

    def forward(self, x):
        out = self.mlp(x)
        fv = self.mlp[0](x)
        return out, fv


class MLPSoftmax(nn.Module):
    """
    Lightweight helper that mirrors the legacy behaviour:
    two-layer MLP followed by a probability simplex projection.
    """
    def __init__(self, in_size, hidden_size, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.temperature = 1.0  # allow external tuning to avoid weight collapse

    def forward(self, x):
        logits = self.mlp(x)
        if self.temperature != 1.0:
            logits = logits / float(self.temperature)
        return torch.softmax(logits, dim=-1)

"""Distribution-focused GHIST+ framework with one hurdle observation model."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .framework import Framework, _to_namespace
from .hurdle_distribution import deterministic_hurdle_prediction
from .modules import MLP


class HurdleFramework(Framework):
    """Eq. 5 signed magnitude plus an identifiable positive-occurrence head."""

    def __init__(self, *args, **kwargs):
        model_cfg = kwargs.get("model_cfg")
        if model_cfg is None and len(args) > 9:
            model_cfg = args[9]
        hurdle_cfg = _to_namespace(getattr(_to_namespace(model_cfg), "hurdle", None))
        occurrence_ecrm_cfg = _to_namespace(
            getattr(hurdle_cfg, "occurrence_ecrm", None)
        )
        super().__init__(*args, **kwargs)

        if not self.hurdle_enabled:
            raise ValueError("HurdleFramework requires model.hurdle.enabled=true")
        if not self.use_avgexp or not self.use_avgexp_residual:
            raise ValueError("HurdleFramework requires the signed avgexp residual path")
        if not self.ecrm_residual_target:
            raise ValueError(
                "HurdleFramework requires residual-only ECRM: "
                "apply_to_embeddings=false, apply_to_ref_weights=false, "
                "apply_to_expr_residual=true"
            )
        if self.vq_patch_inject_cell:
            raise ValueError("HurdleFramework requires vq_patch.inject_cell=false")

        # Framework leaves Eq. 5 signed in hurdle mode. This wrapper owns the
        # single final ReLU exposed as output[3].
        self.expr_relu = False
        self.occurrence_head = MLP(self.hidden_size, self.hidden_size, self.n_genes)
        nn.init.zeros_(self.occurrence_head.mlp[-1].weight)
        nn.init.zeros_(self.occurrence_head.mlp[-1].bias)

        sigma_init = float(getattr(hurdle_cfg, "sigma_init", 0.5))
        self.hurdle_sigma_floor = float(getattr(hurdle_cfg, "sigma_floor", 0.03))
        if sigma_init <= 0 or self.hurdle_sigma_floor <= 0:
            raise ValueError("hurdle sigma_init and sigma_floor must be positive")
        sigma_raw = math.log(math.expm1(sigma_init))
        self.hurdle_sigma_raw = nn.Parameter(
            torch.full((self.n_genes,), float(sigma_raw))
        )

        self.occurrence_ecrm_enabled = bool(
            getattr(occurrence_ecrm_cfg, "enabled", False)
        )
        self.occurrence_ecrm_ablation_off = bool(
            getattr(occurrence_ecrm_cfg, "ablation_off", False)
        )
        if self.occurrence_ecrm_enabled:
            if self.ecrm is None:
                raise ValueError("occurrence_ecrm requires model.ecrm.enabled=true")
            # Deep-copying after all existing modules are initialized preserves
            # every baseline parameter and gives occurrence ranking independent
            # ECRM parameters without consuming random initialization state.
            self.occurrence_ecrm = copy.deepcopy(self.ecrm)
            # The occurrence graph must be portable at inference: use only
            # morphology and coordinates, never predicted/ground-truth CT or
            # reference-expression gates.
            self.occurrence_ecrm.ct_conf_min = 0.0
            self.occurrence_ecrm.ct_same_type_only = False

            gate_max = float(
                getattr(occurrence_ecrm_cfg, "gene_gate_max", 1.0)
            )
            gate_init = float(
                getattr(occurrence_ecrm_cfg, "gene_gate_init", 0.25)
            )
            if gate_max <= 0.0 or not (0.0 < gate_init < gate_max):
                raise ValueError(
                    "occurrence_ecrm requires 0 < gene_gate_init < gene_gate_max"
                )
            gate_fraction = gate_init / gate_max
            gate_raw = math.log(gate_fraction / (1.0 - gate_fraction))
            self.occurrence_ecrm_gene_gate_raw = nn.Parameter(
                torch.full((self.n_genes,), float(gate_raw))
            )
            self.occurrence_ecrm_gene_gate_max = gate_max
        else:
            self.occurrence_ecrm = None
            self.register_parameter("occurrence_ecrm_gene_gate_raw", None)
            self.occurrence_ecrm_gene_gate_max = 0.0

    def hurdle_sigma(self) -> torch.Tensor:
        """Positive per-gene scale in unscaled log1p units."""

        return F.softplus(self.hurdle_sigma_raw) + self.hurdle_sigma_floor

    def set_epoch_progress(self, frac: float):
        super().set_epoch_progress(frac)
        if self.occurrence_ecrm is not None:
            self.occurrence_ecrm.epoch_frac = float(self._epoch_progress.item())

    def _build_hurdle_occurrence_graph_state(
        self,
        self_state,
        coords,
        edge_index,
        patch_ids,
        explicit_empty_graph,
    ):
        if (
            not self.occurrence_ecrm_enabled
            or self.occurrence_ecrm_ablation_off
            or explicit_empty_graph
            or self_state.shape[0] <= 1
        ):
            return None

        # Isolate the new occurrence correction from the shared morphology and
        # magnitude paths.  Its own ECRM parameters still train through the
        # existing Bernoulli term in the hurdle likelihood.
        state = self_state.detach()
        dummy_ct = state.new_ones((state.shape[0], 1))
        self.occurrence_ecrm._patch_ids = patch_ids
        return self.occurrence_ecrm(
            state,
            coords,
            dummy_ct,
            expr_pred=None,
            gate_h=state,
            immune_gate=None,
            invasive_gate=None,
            edge_index=edge_index,
            patch_ids=patch_ids,
        )

    def _occurrence_from_states(
        self,
        self_state: torch.Tensor,
        graph_state: torch.Tensor | None,
    ):
        self_logits, _ = self.occurrence_head(self_state)
        if self.occurrence_ecrm_enabled:
            if graph_state is None:
                delta = torch.zeros_like(self_logits)
                return self_logits, self_logits, delta
            graph_logits, _ = self.occurrence_head(graph_state)
            detached_self_logits, _ = self.occurrence_head(self_state.detach())
            gene_gate = (
                torch.sigmoid(self.occurrence_ecrm_gene_gate_raw)
                * self.occurrence_ecrm_gene_gate_max
            ).to(device=self_logits.device, dtype=self_logits.dtype)
            delta = (graph_logits - detached_self_logits) * gene_gate.view(1, -1)
            return self_logits + delta, self_logits, delta
        if graph_state is None:
            delta = torch.zeros_like(self_logits)
            return self_logits, self_logits, delta
        graph_logits, _ = self.occurrence_head(graph_state)
        delta = graph_logits - self_logits
        return self_logits + delta, self_logits, delta

    def forward(self, *args, **kwargs):
        output = list(super().forward(*args, **kwargs))
        if output[3].numel() == 0:
            return tuple(output)

        aux = self.last_aux_losses
        self_state = aux.get("ecrm_self_state")
        graph_state = (
            aux.get("occurrence_ecrm_graph_state")
            if self.occurrence_ecrm_enabled
            else aux.get("ecrm_graph_state")
        )
        if self_state is None:
            raise RuntimeError("Framework did not expose the hurdle self state")
        occurrence_logits, occurrence_base, occurrence_delta = (
            self._occurrence_from_states(self_state, graph_state)
        )

        signed_mu = output[3]
        if aux.get("ecrm_graph_residual_delta") is None:
            # Explicit empty/off graph has an exact zero magnitude correction,
            # not a missing/implicit value.
            aux["ecrm_graph_residual_delta"] = torch.zeros_like(signed_mu)
        positive_magnitude = F.relu(signed_mu)  # the one final ReLU
        output[3] = positive_magnitude
        output[4] = F.relu(output[4])
        output[5] = F.relu(output[5])

        # Base Framework formed this auxiliary cell-type output from signed mu;
        # recompute it from the actual public nonnegative magnitude.
        if self.use_celltype:
            out_cell_type_expr, fv_cell_type_expr = self.mlp_genes(
                positive_magnitude
            )
            output[6] = out_cell_type_expr
            output[7] = fv_cell_type_expr

        self.last_aux_losses.update(
            {
                "hurdle_signed_mu": signed_mu,
                "hurdle_positive_magnitude": positive_magnitude,
                "hurdle_occurrence_logits": occurrence_logits,
                "hurdle_occurrence_probability": torch.sigmoid(occurrence_logits),
                "hurdle_occurrence_base": occurrence_base,
                "hurdle_occurrence_delta": occurrence_delta,
                "hurdle_occurrence_ecrm_enabled": self.occurrence_ecrm_enabled,
                "hurdle_occurrence_ecrm_gene_gate": (
                    torch.sigmoid(self.occurrence_ecrm_gene_gate_raw).detach()
                    * self.occurrence_ecrm_gene_gate_max
                    if self.occurrence_ecrm_gene_gate_raw is not None
                    else None
                ),
                "hurdle_sigma": self.hurdle_sigma(),
                "hurdle_export_requires_complete_unique_cohort": True,
            }
        )
        return tuple(output)

    @staticmethod
    def export_deterministic_prediction(
        signed_mu: torch.Tensor,
        occurrence_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Export one matrix after complete unique-cell aggregation."""

        return deterministic_hurdle_prediction(signed_mu, occurrence_logits)

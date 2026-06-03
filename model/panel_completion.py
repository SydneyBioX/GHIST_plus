"""Panel completion module for GHIST+."""

import torch
import torch.nn as nn


class PanelCompletionHead(nn.Module):
    """
    Gene-conditioned imputation head ("panel completion").

    Inputs (per cell):
      - delta_obs: (expr_true - expr_ref_base) on observed genes only
      - mask_obs:  0/1 mask for observed genes
      - delta_morph (optional): (out_expr - expr_ref_base) as morphology residual

    Output:
      - delta_hat: predicted residual for all genes (to add on top of expr_ref_base)
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_morph: bool = True,
        morph_gate_init: float = -2.0,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.use_morph = bool(use_morph)

        in_dim = self.n_genes * 2  # delta_obs + mask_obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.n_genes),
        )
        if self.use_morph:
            self.morph_gate = nn.Parameter(
                torch.full((self.n_genes,), float(morph_gate_init))
            )
        else:
            self.register_parameter("morph_gate", None)

    def forward(
        self,
        delta_obs: torch.Tensor,
        mask_obs: torch.Tensor,
        delta_morph=None,
    ) -> torch.Tensor:
        if delta_obs.shape[-1] != self.n_genes or mask_obs.shape[-1] != self.n_genes:
            raise ValueError("PanelCompletionHead: gene dimension mismatch.")
        x = torch.cat([delta_obs, mask_obs], dim=1)
        delta_hat = self.net(x)
        if self.use_morph and delta_morph is not None:
            gate = torch.sigmoid(self.morph_gate).view(1, -1)
            delta_hat = delta_hat + gate * delta_morph
        return delta_hat

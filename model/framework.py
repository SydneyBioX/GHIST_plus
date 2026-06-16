"""GHIST+ model framework.

This module wires the image encoder, cell-type heads, ECRM neighbourhood
mixers, VQ tile prototypes, expression heads, and panel-completion branch.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    VQCodebook,
    EdgeCondMixer,
    CrossAttention,
    CompExprRefinerMTA,
    Embed,
    MLP,
    MLPSoftmax,
)
from .backbone import Uni2HAdapter, Backbone as LegacyBackbone


def _to_namespace(obj):
    if obj is None:
        return SimpleNamespace()
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        ns = SimpleNamespace()
        for k, v in obj.items():
            setattr(ns, k, _to_namespace(v))
        return ns
    if hasattr(obj, "_asdict"):
        return _to_namespace(obj._asdict())
    return obj


class Framework(nn.Module):
    """
    Drop-in replacement for the legacy Framework that now:
      * uses a ViT-H/14 encoder via Uni2HAdapter
      * mixes neighbourhoods with EdgeCondMixer (ECRM)
      * quantises patch embeddings with a VQ codebook
    """

    def __init__(
        self,
        n_classes,
        n_genes,
        emb_dim,
        device,
        n_ref,
        use_avgexp,
        use_celltype,
        use_neighb,
        in_channels=3,
        model_cfg=None,
    ):
        super().__init__()
        model_cfg = _to_namespace(model_cfg)
        self.ct_priors_ecrm_unit_enabled = bool(
            getattr(model_cfg, "ct_priors_ecrm_unit_enabled", True)
        )
        if not self.ct_priors_ecrm_unit_enabled:
            model_cfg.celltype_priors_enabled = False
            if getattr(model_cfg, "ecrm", None) is None:
                model_cfg.ecrm = SimpleNamespace()
            model_cfg.ecrm.enabled = False
        self.hidden_size = emb_dim
        self.device = device
        self.n_genes = n_genes
        self.n_ref = n_ref if n_ref is not None else 0

        # Re-enable avgexp guard and CT-tied neighbourhoods as in the newer model
        self.use_avgexp = bool(use_avgexp) and self.n_ref > 0
        self.use_celltype = bool(use_celltype)
        self.use_neighb = bool(use_neighb) and self.use_celltype
        self.use_celltype_priors = bool(
            getattr(model_cfg, "celltype_priors_enabled", True)
        )
        self.use_temp_control = bool(
            getattr(model_cfg, "avgexp_temp_enabled", True)
        )
        self.use_gt_ct_ref_weights = bool(
            getattr(model_cfg, "use_gt_ct_ref_weights", False)
        )
        self.refiner_type = str(getattr(model_cfg, "refiner_type", "mta")).lower()
        self.use_crossattn = (
            self.use_neighb
            and bool(getattr(model_cfg, "crossattn", True))
            and self.use_celltype_priors
        )
        self.ref_temperature = float(getattr(model_cfg, "avgexp_temp", 1.0))
        self.use_avgexp_residual = bool(
            getattr(model_cfg, "use_avgexp_residual", True)
        ) and self.use_celltype_priors
        self.avgexp_residual_scale = float(
            getattr(model_cfg, "avgexp_residual_scale", 0.1)
        )
        self.ct_prior_blend_alpha = float(
            getattr(model_cfg, "ct_prior_blend_alpha", 1.0)
        )
        # match initial behaviour: clamp outputs after prediction
        self.expr_relu = True

        # ------------------------------------------------------------------ #
        # Backbone selection (foundation model vs legacy UNet)
        # ------------------------------------------------------------------ #
        n_classes_backbone = (n_classes + 1) if self.use_celltype else 2
        fm_cfg = _to_namespace(getattr(model_cfg, "foundation_model", None))
        self.use_foundation_model = bool(getattr(fm_cfg, "enabled", True))
        self.legacy_backbone_frozen = bool(
            getattr(model_cfg, "legacy_backbone_frozen", False)
        )
        fm_pretrained = bool(getattr(fm_cfg, "pretrained", True))
        if self.use_foundation_model:
            self.cnn = Uni2HAdapter(
                pretrained=fm_pretrained, n_seg_classes=n_classes_backbone
            )
        else:
            self.cnn = LegacyBackbone(
                n_channels=in_channels, n_classes=n_classes_backbone
            )
        self._encoder_blocks = self._find_vit_blocks()
        self.freeze_backbone = self.use_foundation_model or self.legacy_backbone_frozen
        if self.freeze_backbone:
            self._freeze_encoder()
            self.cnn.eval()

        # feature vector layout: per-cell hd1/h1 + tile-level hd1/h1
        self.dim_hd1 = 320
        self.dim_h1 = 64
        self.dim_fv = 2 * (self.dim_hd1 + self.dim_h1)  # (cell + tile summaries)

        self.embed_hist = Embed(self.dim_fv, self.hidden_size)
        self.embed_patch = Embed(self.dim_hd1 + self.dim_h1, self.hidden_size)

        # --- VQ configuration ------------------------------------------------
        vq_cfg = _to_namespace(getattr(model_cfg, "vq_patch", None))
        self.vq_patch_space = str(getattr(vq_cfg, "space", "hidden")).lower()
        if self.vq_patch_space not in {"hidden", "tile"}:
            raise ValueError(
                f"Unknown model.vq_patch.space={self.vq_patch_space!r} (expected 'hidden' or 'tile')"
            )
        self.vq_patch_composition_requires_vq = bool(
            getattr(vq_cfg, "composition_requires_vq", False)
        )
        self.vq_patch_inject_cell = bool(getattr(vq_cfg, "inject_cell", False))
        self.vq_patch_inject_cell_scale = float(
            getattr(vq_cfg, "inject_cell_scale", 1.0)
        )
        self.vq_patch_loss_w = float(getattr(vq_cfg, "loss_w", 0.0))
        if getattr(vq_cfg, "enabled", False):
            vq_dim = (
                self.hidden_size
                if self.vq_patch_space == "hidden"
                else (self.dim_hd1 + self.dim_h1)
            )
            self.vq_patch = VQCodebook(
                n_codes=int(getattr(vq_cfg, "n_codes", 64)),
                dim=vq_dim,
                beta=float(getattr(vq_cfg, "beta", 0.25)),
                use_cosine=bool(getattr(vq_cfg, "use_cosine", True)),
            )
            self.vq_patch.codebook.requires_grad_(False)
        else:
            self.vq_patch = None

        # --- Reference mixing heads -----------------------------------------
        if self.use_avgexp:
            self.mlp_weights = MLPSoftmax(self.hidden_size, self.hidden_size, self.n_ref)
            self.mlp_weights_immune = MLPSoftmax(
                self.hidden_size, self.hidden_size, self.n_ref
            )
            self.mlp_weights_invasive = MLPSoftmax(
                self.hidden_size, self.hidden_size, self.n_ref
            )
            # soften softmax to discourage collapse
            if self.use_temp_control:
                self.mlp_weights.temperature = self.ref_temperature
                self.mlp_weights_immune.temperature = self.ref_temperature
                self.mlp_weights_invasive.temperature = self.ref_temperature
            if self.use_avgexp_residual:
                # Residual heads that learn per-gene offsets on top of ref-weighted expr
                self.mlp_avgexp_residual = MLP(self.hidden_size, self.hidden_size, n_genes)
                self.mlp_avgexp_residual_immune = MLP(self.hidden_size, self.hidden_size, n_genes)
                self.mlp_avgexp_residual_invasive = MLP(self.hidden_size, self.hidden_size, n_genes)
        else:
            self.mlp_offset = MLP(self.hidden_size, self.hidden_size, n_genes)
            self.mlp_offset_immune = MLP(
                self.hidden_size, self.hidden_size, n_genes
            )
            self.mlp_offset_invasive = MLP(
                self.hidden_size, self.hidden_size, n_genes
            )

        # --- Neighbourhood mixing & cross-attention -------------------------
        if self.use_neighb and n_classes > 0:
            self.estimate_comp = MLPSoftmax(
                self.hidden_size, self.hidden_size, n_classes
            )
        else:
            self.estimate_comp = None

        def _gate_param(init=0.1, gate_max=1.0):
            init = float(init)
            gate_max = float(gate_max)
            init = min(max(init, 1e-6), max(gate_max - 1e-6, 1e-6))
            raw = math.log(init / max(gate_max - init, 1e-8))
            return nn.Parameter(torch.tensor(raw)), gate_max

        if self.use_crossattn and n_classes > 0:
            refiner_cls = CompExprRefinerMTA if self.refiner_type == "mta" else CrossAttention
            refiner_kwargs = {}
            if refiner_cls is CompExprRefinerMTA:
                refiner_kwargs["num_query_tokens"] = int(
                    getattr(model_cfg, "refiner_query_tokens", 8)
                )
            self.refine_expr = refiner_cls(
                n_classes, n_genes, self.hidden_size, num_heads=8, **refiner_kwargs
            )
            self.refine_expr_immune = refiner_cls(
                n_classes, n_genes, self.hidden_size, num_heads=8, **refiner_kwargs
            )
            self.refine_expr_invasive = refiner_cls(
                n_classes, n_genes, self.hidden_size, num_heads=8, **refiner_kwargs
            )
            # Small gate to ramp cross-attn contribution; initialized low for stability
            self.cross_gate_raw, self.cross_gate_max = _gate_param()
            self.cross_gate_immune_raw, self.cross_gate_immune_max = _gate_param()
            self.cross_gate_invasive_raw, self.cross_gate_invasive_max = _gate_param()
        else:
            self.refine_expr = None
            self.refine_expr_immune = None
            self.refine_expr_invasive = None
            self.cross_gate_raw = None
            self.cross_gate_max = None
            self.cross_gate_immune_raw = None
            self.cross_gate_immune_max = None
            self.cross_gate_invasive_raw = None
            self.cross_gate_invasive_max = None

        expr_ref_gate_init = float(getattr(model_cfg, "expr_ref_gate_init", 0.1))
        expr_ref_gate_max = float(getattr(model_cfg, "expr_ref_gate_max", 0.5))
        if self.use_avgexp:
            self.expr_ref_gate_raw, self.expr_ref_gate_max = _gate_param(
                expr_ref_gate_init, expr_ref_gate_max
            )
            self.expr_ref_gate_immune_raw, self.expr_ref_gate_immune_max = _gate_param(
                expr_ref_gate_init, expr_ref_gate_max
            )
            self.expr_ref_gate_invasive_raw, self.expr_ref_gate_invasive_max = _gate_param(
                expr_ref_gate_init, expr_ref_gate_max
            )
        else:
            self.expr_ref_gate_raw = None
            self.expr_ref_gate_max = None
            self.expr_ref_gate_immune_raw = None
            self.expr_ref_gate_immune_max = None
            self.expr_ref_gate_invasive_raw = None
            self.expr_ref_gate_invasive_max = None

        ecrm_cfg = _to_namespace(getattr(model_cfg, "ecrm", None))
        self.use_ecrm = self.use_neighb and bool(getattr(ecrm_cfg, "enabled", True))
        if self.use_ecrm:
            self.ecrm = EdgeCondMixer(self.hidden_size, k=int(getattr(ecrm_cfg, "k_target", 16)))
            self.ecrm_apply_to_embeddings = bool(getattr(ecrm_cfg, "apply_to_embeddings", True))
            self.ecrm_apply_to_ref_weights = bool(getattr(ecrm_cfg, "apply_to_ref_weights", False))
            self.ecrm_apply_to_expr_residual = bool(getattr(ecrm_cfg, "apply_to_expr_residual", False))
            self.ecrm_ref_weights_alpha = float(getattr(ecrm_cfg, "ref_weights_alpha", 1.0))
            self.ecrm_expr_residual_alpha = float(getattr(ecrm_cfg, "expr_residual_alpha", 1.0))
            self.ecrm_use_gt_ct = bool(getattr(ecrm_cfg, "use_gt_ct", False))
            self.ecrm_gate_h_from_embeddings = bool(
                getattr(ecrm_cfg, "gate_h_from_embeddings", False)
            )
            # Allow per-class overrides for degree/trust parameters
            self.ecrm.k_target = float(getattr(ecrm_cfg, "k_target", self.ecrm.k_target))
            self.ecrm.k_min = float(getattr(ecrm_cfg, "k_min", self.ecrm.k_min))
            self.ecrm.k_max = float(getattr(ecrm_cfg, "k_max", self.ecrm.k_max))
            self.ecrm.density_gamma = float(
                getattr(ecrm_cfg, "density_gamma", self.ecrm.density_gamma)
            )
            self.ecrm.eta_max = float(getattr(ecrm_cfg, "eta_max", self.ecrm.eta_max))
            self.ecrm.gamma_perp_max = float(
                getattr(ecrm_cfg, "gamma_perp_max", self.ecrm.gamma_perp_max)
            )
            self.ecrm.trust_floor = float(
                getattr(ecrm_cfg, "trust_floor", self.ecrm.trust_floor)
            )
            self.ecrm.trust_scale = float(
                getattr(ecrm_cfg, "trust_scale", self.ecrm.trust_scale)
            )
            self.ecrm.ct_conf_min = float(
                getattr(ecrm_cfg, "ct_conf_min", self.ecrm.ct_conf_min)
            )
            self.ecrm.ct_same_type_only = bool(
                getattr(ecrm_cfg, "ct_same_type_only", self.ecrm.ct_same_type_only)
            )
            self.ecrm.depth = int(getattr(ecrm_cfg, "depth", getattr(self.ecrm, "depth", 1)))
            self.ecrm.edge_dropout = float(
                getattr(ecrm_cfg, "edge_dropout", getattr(self.ecrm, "edge_dropout", 0.0))
            )
            self.ecrm.message_dropout = float(
                getattr(
                    ecrm_cfg,
                    "message_dropout",
                    getattr(self.ecrm, "message_dropout", 0.0),
                )
            )
            residual_gate_init = float(
                getattr(ecrm_cfg, "residual_gate_init", -0.5)
            )
            self.ecrm.beta.data.fill_(residual_gate_init)
        else:
            self.ecrm = None
            self.ecrm_apply_to_embeddings = False
            self.ecrm_apply_to_ref_weights = False
            self.ecrm_apply_to_expr_residual = False
            self.ecrm_ref_weights_alpha = 0.0
            self.ecrm_expr_residual_alpha = 0.0
            self.ecrm_use_gt_ct = False
            self.ecrm_gate_h_from_embeddings = False
        # Small projection of embeddings for ECRM expression similarity
        if self.use_ecrm:
            self.ecrm_expr_proj = nn.Linear(self.hidden_size, 32, bias=False)
        else:
            self.ecrm_expr_proj = None

        if self.use_celltype:
            self.mlp_hist = MLP(self.hidden_size, self.hidden_size, n_classes)
            self.mlp_genes = MLP(n_genes, self.hidden_size, n_classes)
        else:
            self.mlp_hist = None
            self.mlp_genes = None

        self.relu = nn.ReLU()
        self.last_aux_losses = {}
        self.register_buffer(
            "_epoch_progress",
            torch.tensor(1.0),
            persistent=False,
        )
        self._coord_cache = {}

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def set_epoch_progress(self, frac: float):
        frac = float(max(0.0, min(1.0, frac)))
        self._epoch_progress.fill_(frac)
        if self.ecrm is not None:
            self.ecrm.epoch_frac = frac

    def _coord_grid(self, H, W, device):
        key = (H, W, device)
        grid = self._coord_cache.get(key)
        if grid is None:
            yy = torch.arange(H, device=device).float().view(1, H, 1).expand(1, H, W)
            xx = torch.arange(W, device=device).float().view(1, 1, W).expand(1, H, W)
            grid = torch.cat([yy, xx], dim=0)  # (2, H, W)
            self._coord_cache[key] = grid
        return grid

    def _resolve_ref_weights(self, learned_weights, ct_prob_pred, ct_labels):
        if not (self.use_celltype and self.use_celltype_priors):
            return learned_weights
        if self.use_gt_ct_ref_weights and ct_labels is not None:
            return F.one_hot(ct_labels, num_classes=self.n_ref).float()
        if ct_prob_pred is None:
            return learned_weights

        alpha = max(0.0, min(1.0, float(self.ct_prior_blend_alpha)))
        if alpha <= 0.0:
            return learned_weights

        learned = learned_weights.clamp_min(0.0)
        learned = learned / learned.sum(dim=1, keepdim=True).clamp_min(1e-6)
        prior = ct_prob_pred.clamp_min(0.0)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mixed = (1.0 - alpha) * learned + alpha * prior
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _fuse_direct_with_ref(self, expr_direct, ref_base, ref_offsets, gate_raw, gate_max):
        ref_corr = ref_base if ref_offsets is None else (ref_base + ref_offsets)
        if gate_raw is None or gate_max is None:
            gate = ref_corr.new_tensor(1.0)
        else:
            gate = torch.sigmoid(gate_raw).to(
                device=ref_corr.device, dtype=ref_corr.dtype
            ) * float(gate_max)
        out_expr = expr_direct + gate * ref_corr
        if self.expr_relu:
            out_expr = self.relu(out_expr)
        return out_expr, gate, ref_corr

    def _freeze_encoder(self):
        for p in self.cnn.parameters():
            p.requires_grad = False

    def _find_vit_blocks(self):
        if not self.use_foundation_model:
            return nn.ModuleList()
        enc = getattr(self.cnn, "enc", None)
        vit = getattr(enc, "vit", None)
        blocks = getattr(vit, "blocks", None)
        if blocks is None:
            return nn.ModuleList()
        return nn.ModuleList(list(blocks))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.cnn.eval()
        return self

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_hist,
        nuclei_mask,
        n_cells,
        ref_orig,
        batch_ct=None,
        batch_expr=None,
        patch_ids=None,
        patch_slide_idx=None,
        coords_cells=None,
        cell_edge_index=None,
        cell_patch_ids=None,
        do_st_mlp=True,
    ):
        if self.freeze_backbone:
            with torch.no_grad():
                out_map, hd1, h1 = self.cnn(x_hist)
        else:
            out_map, hd1, h1 = self.cnn(x_hist)
        B, _, H, W = hd1.shape
        device = x_hist.device

        # patch embeddings (tile-level)
        tile_features = torch.cat(
            [hd1.mean(dim=(2, 3)), h1.mean(dim=(2, 3))], dim=1
        )  # (B, 384)
        tile_features_for_cells = tile_features
        allow_composition = (not self.vq_patch_composition_requires_vq) or (
            self.vq_patch is not None
        )
        vq_loss = torch.zeros((), device=device)
        vq_patch_idx = None
        vq_patch_err = None
        if (
            self.vq_patch is not None
            and tile_features_for_cells.shape[0] > 0
            and self.vq_patch_space == "tile"
        ):
            vq_input = tile_features_for_cells
            z_q_tile, vq_loss_raw, vq_patch_idx, _ = self.vq_patch(vq_input)
            vq_patch_err = ((z_q_tile.detach() - vq_input.detach()) ** 2).mean(dim=1)
            tile_features_for_cells = z_q_tile
            vq_loss = vq_loss_raw

        patch_embed = self.embed_patch(tile_features_for_cells)
        torch.nan_to_num_(patch_embed, nan=0.0, posinf=0.0, neginf=0.0)
        if (
            self.vq_patch is not None
            and patch_embed.shape[0] > 0
            and self.vq_patch_space == "hidden"
        ):
            vq_input = patch_embed
            z_q, vq_loss_raw, vq_patch_idx, _ = self.vq_patch(vq_input)
            vq_patch_err = ((z_q.detach() - vq_input.detach()) ** 2).mean(dim=1)
            embeddings_patches = z_q
            vq_loss = vq_loss_raw
        else:
            embeddings_patches = patch_embed

        if not allow_composition:
            tile_features_for_cells = torch.zeros_like(tile_features_for_cells)

        n_cells_total = int(n_cells.sum().item())
        if n_cells_total == 0:
            self.last_aux_losses = {
                "vq_patch": vq_loss * self.vq_patch_loss_w,
                "vq_patch_idx": vq_patch_idx,
                "vq_patch_err": vq_patch_err,
            }
            zeros = torch.zeros(0, self.hidden_size, device=device)
            return (
                zeros,
                out_map,
                torch.zeros(0, device=device, dtype=torch.long),
                torch.zeros(0, self.n_genes, device=device),
                torch.zeros(0, self.n_genes, device=device),
                torch.zeros(0, self.n_genes, device=device),
                zeros,
                zeros,
                zeros,
                zeros,
                torch.zeros(0, self.n_genes, device=device),
                None,
                torch.zeros(0, device=device),
                None,
            )

        all_fv = torch.zeros((n_cells_total, self.dim_fv), device=device)
        all_area = torch.zeros(n_cells_total, device=device)
        coords_all = torch.zeros((n_cells_total, 2), device=device)
        patch_assign = torch.zeros(n_cells_total, dtype=torch.long, device=device)
        cells_written = []

        coord_grid = self._coord_grid(H, W, device)
        cursor = 0

        patch_ids_list = []
        for b in range(B):
            c_mask_all = nuclei_mask[b]
            cids = torch.unique(c_mask_all, sorted=True)
            cids = cids[cids > 0]
            n_valid = 0
            if cids.numel() == 0:
                cells_written.append(0)
                continue

            tile_vec = tile_features_for_cells[b]

            for cid in cids:
                c_mask = (c_mask_all == cid).float()
                area = c_mask.sum()
                if area.item() <= 0:
                    continue

                c_fv_hd1 = (c_mask * hd1[b]).sum((1, 2)) / area
                c_fv_h1 = (c_mask * h1[b]).sum((1, 2)) / area
                fv = torch.cat([c_fv_hd1, c_fv_h1, tile_vec])
                all_fv[cursor] = fv
                all_area[cursor] = area
                patch_assign[cursor] = b

                coords_y = (c_mask * coord_grid[0]).sum() / area
                coords_x = (c_mask * coord_grid[1]).sum() / area
                coords_all[cursor, 0] = (coords_y / max(H - 1, 1)) * 2 - 1
                coords_all[cursor, 1] = (coords_x / max(W - 1, 1)) * 2 - 1

                cursor += 1
                n_valid += 1

            cells_written.append(n_valid)
            if patch_ids is not None:
                patch_ids_list.append(patch_ids[b, :n_valid].to(device))

        effective_cells = cursor
        all_fv = all_fv[:effective_cells]
        all_area = all_area[:effective_cells]
        coords_all = coords_all[:effective_cells]
        patch_assign = patch_assign[:effective_cells]

        if (
            coords_cells is not None
            and isinstance(coords_cells, torch.Tensor)
            and coords_cells.shape[0] == coords_all.shape[0]
        ):
            coords_all = coords_cells[:, :2].to(device)
        edge_index_cells = None
        if (
            cell_edge_index is not None
            and isinstance(cell_edge_index, torch.Tensor)
            and cell_edge_index.numel() > 0
        ):
            edge_index_cells = cell_edge_index.to(device)
        if (
            cell_patch_ids is not None
            and isinstance(cell_patch_ids, torch.Tensor)
            and cell_patch_ids.shape[0] == patch_assign.shape[0]
        ):
            patch_assign = cell_patch_ids.to(device)

        if effective_cells == 0:
            self.last_aux_losses = {
                "vq_patch": vq_loss * self.vq_patch_loss_w,
                "vq_patch_idx": vq_patch_idx,
                "vq_patch_err": vq_patch_err,
            }
            zeros = torch.zeros(0, self.hidden_size, device=device)
            return (
                zeros,
                out_map,
                torch.zeros(0, device=device, dtype=torch.long),
                torch.zeros(0, self.n_genes, device=device),
                torch.zeros(0, self.n_genes, device=device),
                torch.zeros(0, self.n_genes, device=device),
                zeros,
                zeros,
                zeros,
                zeros,
                torch.zeros(0, self.n_genes, device=device),
                None,
                torch.zeros(0, device=device),
                None,
            )

        embeddings = self.embed_hist(all_fv)
        torch.nan_to_num_(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        if (
            self.vq_patch_inject_cell
            and allow_composition
            and embeddings_patches is not None
            and embeddings_patches.shape[0] > 0
            and patch_assign.shape[0] == embeddings.shape[0]
        ):
            min_pid = int(patch_assign.min().item())
            max_pid = int(patch_assign.max().item())
            if min_pid < 0 or max_pid >= embeddings_patches.shape[0]:
                raise RuntimeError(
                    "vq_patch.inject_cell expects patch indices in [0, "
                    f"{embeddings_patches.shape[0] - 1}] but got min={min_pid}, max={max_pid}"
                )
            embeddings = embeddings + (
                embeddings_patches.index_select(0, patch_assign)
                * self.vq_patch_inject_cell_scale
            )
            torch.nan_to_num_(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        embeddings_gate = embeddings
        ct_labels = None
        if self.use_celltype and batch_ct is not None and batch_ct.numel() > 0:
            ct_list_gt = []
            for b, n_valid in enumerate(cells_written):
                if n_valid == 0:
                    continue
                ct_list_gt.append(batch_ct[b, :n_valid].to(device))
            if ct_list_gt:
                ct_labels = torch.cat(ct_list_gt, dim=0).clamp(min=0).long()

        # composition prediction per patch
        comp_estimated = None
        comp_tiled_all = None
        if (
            allow_composition
            and self.estimate_comp is not None
            and embeddings_patches.shape[0] > 0
        ):
            comp_estimated = self.estimate_comp(embeddings_patches)
            comp_estimated = torch.nan_to_num(
                comp_estimated, nan=0.0, posinf=0.0, neginf=0.0
            )
            tiled = []
            idx_c = 0
            for b, n_valid in enumerate(cells_written):
                if n_valid == 0:
                    continue
                tiled.append(
                    comp_estimated[b].unsqueeze(0).expand(n_valid, -1)
                )
                idx_c += n_valid
            if tiled:
                comp_tiled_all = torch.cat(tiled, dim=0).to(device)

        # --- Cell-type logits before ECRM (needed for neighbour confidence)
        if self.use_celltype:
            out_cell_type_pre, _ = self.mlp_hist(embeddings)
            ct_prob_pred = torch.softmax(out_cell_type_pre, dim=1)
            if self.ecrm_use_gt_ct and ct_labels is not None:
                ct_prob_input = torch.nn.functional.one_hot(
                    ct_labels, num_classes=self.n_ref
                ).float()
            else:
                ct_prob_input = ct_prob_pred.detach()
        else:
            out_cell_type_pre = None
            ct_prob_pred = None
            ct_prob_input = torch.ones(
                embeddings.size(0), 1, device=device, dtype=embeddings.dtype
            )

        immune_gate = None
        invasive_gate = None
        if (
            self.use_neighb
            and self.use_avgexp
            and self.use_celltype_priors
            and embeddings.size(0) > 0
            and hasattr(self, "mlp_weights")
        ):
            w_base = self.mlp_weights(embeddings)
            w_immune = self.mlp_weights_immune(embeddings)
            w_invasive = self.mlp_weights_invasive(embeddings)
            immune_gate = 0.5 * (w_immune - w_base).abs().sum(dim=1)
            invasive_gate = 0.5 * (w_invasive - w_base).abs().sum(dim=1)
            immune_gate = immune_gate.clamp(0.0, 1.0)
            invasive_gate = invasive_gate.clamp(0.0, 1.0)

        if (
            self.use_ecrm
            and self.ecrm is not None
            and self.ecrm_apply_to_embeddings
            and coords_all.size(0) > 1
        ):
            expr_sim = None
            if self.ecrm_expr_proj is not None:
                expr_source = embeddings_gate if self.ecrm_gate_h_from_embeddings else embeddings
                expr_sim = self.ecrm_expr_proj(expr_source)
            self.ecrm._patch_ids = patch_assign
            embeddings = self.ecrm(
                embeddings,
                coords_all,
                ct_prob_input,
                expr_pred=expr_sim,
                gate_h=embeddings_gate if self.ecrm_gate_h_from_embeddings else None,
                immune_gate=immune_gate,
                invasive_gate=invasive_gate,
                edge_index=edge_index_cells,
                patch_ids=patch_assign,
            )

        # --- Expression heads ------------------------------------------------
        ref_weight_entropy = None
        ref_weight_entropy_immune = None
        ref_weight_entropy_invasive = None
        expr_ref_gate = None
        expr_ref_gate_immune = None
        expr_ref_gate_invasive = None
        if self.use_avgexp:
            # assume reference provided when avgexp is enabled
            ref = ref_orig.unsqueeze(0).to(device)
            ref = ref.expand(embeddings.size(0), -1, -1)

            ref_weights = self._resolve_ref_weights(
                self.mlp_weights(embeddings), ct_prob_pred, ct_labels
            )
            if ref_weights.numel() > 0 and ref_weights.shape[1] > 1:
                ref_weight_entropy = -(
                    ref_weights.clamp_min(1e-8) * ref_weights.clamp_min(1e-8).log()
                ).sum(dim=1).mean() / math.log(ref_weights.shape[1])

            if (
                self.use_ecrm
                and self.ecrm is not None
                and self.ecrm_apply_to_ref_weights
                and coords_all.size(0) > 1
                and ref_weights.numel() > 0
            ):
                expr_sim = None
                if self.ecrm_expr_proj is not None:
                    expr_source = embeddings_gate if self.ecrm_gate_h_from_embeddings else embeddings
                    expr_sim = self.ecrm_expr_proj(expr_source)
                self.ecrm._patch_ids = patch_assign
                ref_weights_mixed = self.ecrm(
                    ref_weights,
                    coords_all,
                    ct_prob_input,
                    expr_pred=expr_sim,
                    gate_h=embeddings_gate if self.ecrm_gate_h_from_embeddings else None,
                    immune_gate=immune_gate,
                    invasive_gate=invasive_gate,
                    edge_index=edge_index_cells,
                    patch_ids=patch_assign,
                )
                alpha = float(self.ecrm_ref_weights_alpha)
                alpha = max(0.0, min(1.0, alpha))
                ref_weights = ref_weights + alpha * (ref_weights_mixed - ref_weights)
                ref_weights = ref_weights.clamp_min(0.0)
                ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            ref_weighted = torch.sum(ref_weights.unsqueeze(-1) * ref, dim=1)
            ref_offsets = None
            if self.use_crossattn and comp_tiled_all is not None:
                ref_offsets = self.refine_expr(comp_tiled_all, ref_weighted)
                cross_gate = torch.sigmoid(self.cross_gate_raw) * self.cross_gate_max
                ref_offsets = ref_offsets * cross_gate
            if self.use_avgexp_residual and hasattr(self, "mlp_avgexp_residual"):
                ref_residual, _ = self.mlp_avgexp_residual(embeddings)
                expr_direct = self.avgexp_residual_scale * ref_residual
            else:
                expr_direct = torch.zeros_like(ref_weighted)
            torch.nan_to_num_(expr_direct, nan=0.0, posinf=0.0, neginf=0.0)

            if (
                self.use_ecrm
                and self.ecrm is not None
                and self.ecrm_apply_to_expr_residual
                and coords_all.size(0) > 1
                and expr_direct.numel() > 0
            ):
                expr_sim = None
                if self.ecrm_expr_proj is not None:
                    expr_source = embeddings_gate if self.ecrm_gate_h_from_embeddings else embeddings
                    expr_sim = self.ecrm_expr_proj(expr_source)
                self.ecrm._patch_ids = patch_assign
                expr_direct_mixed = self.ecrm(
                    expr_direct,
                    coords_all,
                    ct_prob_input,
                    expr_pred=expr_sim,
                    gate_h=embeddings_gate if self.ecrm_gate_h_from_embeddings else None,
                    immune_gate=immune_gate,
                    invasive_gate=invasive_gate,
                    edge_index=edge_index_cells,
                    patch_ids=patch_assign,
                )
                alpha = float(self.ecrm_expr_residual_alpha)
                alpha = max(0.0, min(1.0, alpha))
                expr_direct = expr_direct + alpha * (expr_direct_mixed - expr_direct)

            out_expr, expr_ref_gate, _ = self._fuse_direct_with_ref(
                expr_direct,
                ref_weighted,
                ref_offsets,
                self.expr_ref_gate_raw,
                self.expr_ref_gate_max,
            )

            ref_weights_immune = self._resolve_ref_weights(
                self.mlp_weights_immune(embeddings), ct_prob_pred, ct_labels
            )
            if ref_weights_immune.numel() > 0 and ref_weights_immune.shape[1] > 1:
                ref_weight_entropy_immune = -(
                    ref_weights_immune.clamp_min(1e-8)
                    * ref_weights_immune.clamp_min(1e-8).log()
                ).sum(dim=1).mean() / math.log(ref_weights_immune.shape[1])
            ref_immune = torch.sum(ref_weights_immune.unsqueeze(-1) * ref, dim=1)
            ref_offsets_immune = None
            if self.use_crossattn and comp_tiled_all is not None:
                ref_offsets_immune = self.refine_expr_immune(
                    comp_tiled_all, ref_immune
                )
                gate_immune = torch.sigmoid(self.cross_gate_immune_raw) * self.cross_gate_immune_max
                ref_offsets_immune = ref_offsets_immune * gate_immune
            if self.use_avgexp_residual and hasattr(self, "mlp_avgexp_residual_immune"):
                ref_residual_immune, _ = self.mlp_avgexp_residual_immune(embeddings)
                expr_direct_immune = self.avgexp_residual_scale * ref_residual_immune
            else:
                expr_direct_immune = torch.zeros_like(ref_immune)
            torch.nan_to_num_(expr_direct_immune, nan=0.0, posinf=0.0, neginf=0.0)
            out_expr_immune, expr_ref_gate_immune, _ = self._fuse_direct_with_ref(
                expr_direct_immune,
                ref_immune,
                ref_offsets_immune,
                self.expr_ref_gate_immune_raw,
                self.expr_ref_gate_immune_max,
            )

            ref_weights_invasive = self._resolve_ref_weights(
                self.mlp_weights_invasive(embeddings), ct_prob_pred, ct_labels
            )
            if ref_weights_invasive.numel() > 0 and ref_weights_invasive.shape[1] > 1:
                ref_weight_entropy_invasive = -(
                    ref_weights_invasive.clamp_min(1e-8)
                    * ref_weights_invasive.clamp_min(1e-8).log()
                ).sum(dim=1).mean() / math.log(ref_weights_invasive.shape[1])
            ref_invasive = torch.sum(ref_weights_invasive.unsqueeze(-1) * ref, dim=1)
            ref_offsets_invasive = None
            if self.use_crossattn and comp_tiled_all is not None:
                ref_offsets_invasive = self.refine_expr_invasive(
                    comp_tiled_all, ref_invasive
                )
                gate_invasive = torch.sigmoid(self.cross_gate_invasive_raw) * self.cross_gate_invasive_max
                ref_offsets_invasive = ref_offsets_invasive * gate_invasive
            if self.use_avgexp_residual and hasattr(self, "mlp_avgexp_residual_invasive"):
                ref_residual_inv, _ = self.mlp_avgexp_residual_invasive(embeddings)
                expr_direct_invasive = self.avgexp_residual_scale * ref_residual_inv
            else:
                expr_direct_invasive = torch.zeros_like(ref_invasive)
            torch.nan_to_num_(expr_direct_invasive, nan=0.0, posinf=0.0, neginf=0.0)
            out_expr_invasive, expr_ref_gate_invasive, _ = self._fuse_direct_with_ref(
                expr_direct_invasive,
                ref_invasive,
                ref_offsets_invasive,
                self.expr_ref_gate_invasive_raw,
                self.expr_ref_gate_invasive_max,
            )
        else:
            ref_offsets, _ = self.mlp_offset(embeddings)
            out_expr = self.relu(ref_offsets)

            ref_offsets_immune, _ = self.mlp_offset_immune(embeddings)
            out_expr_immune = self.relu(ref_offsets_immune)

            ref_offsets_invasive, _ = self.mlp_offset_invasive(embeddings)
            out_expr_invasive = self.relu(ref_offsets_invasive)

            if self.use_crossattn and comp_tiled_all is not None:
                out_expr = self.refine_expr(comp_tiled_all, out_expr)
                out_expr_immune = self.refine_expr_immune(
                    comp_tiled_all, out_expr_immune
                )
                out_expr_invasive = self.refine_expr_invasive(
                    comp_tiled_all, out_expr_invasive
                )

        # --- Cell-type heads -------------------------------------------------
        if self.use_celltype:
            out_cell_type, fv_hist = self.mlp_hist(embeddings)
            torch.nan_to_num_(out_cell_type, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            out_cell_type = None
            fv_hist = None

        if self.use_celltype:
            out_cell_type_expr, fv_cell_type_expr = self.mlp_genes(out_expr)
        else:
            out_cell_type_expr = None
            fv_cell_type_expr = None

        if batch_expr is not None:
            expr_list = []
            for b, n_valid in enumerate(cells_written):
                if n_valid == 0:
                    continue
                expr_list.append(batch_expr[b, :n_valid, :].to(device))
            batch_expr_pc = (
                torch.cat(expr_list, dim=0)
                if expr_list
                else torch.zeros(0, self.n_genes, device=device)
            )
        else:
            batch_expr_pc = torch.zeros(
                embeddings.size(0), self.n_genes, device=device
            )

        if (
            do_st_mlp
            and batch_expr is not None
            and self.use_celltype
        ):
            out_cell_type_gt_expr, fv_cell_type_gt_expr = self.mlp_genes(batch_expr_pc)
        else:
            out_cell_type_gt_expr = None
            fv_cell_type_gt_expr = None

        if batch_ct is not None and self.use_celltype:
            ct_list = []
            for b, n_valid in enumerate(cells_written):
                if n_valid == 0:
                    continue
                ct_list.append(batch_ct[b, :n_valid].to(device))
            batch_ct_pc = torch.cat(ct_list, dim=0) if ct_list else torch.zeros(0, device=device, dtype=torch.long)
        else:
            batch_ct_pc = torch.zeros(embeddings.size(0), device=device, dtype=torch.long)

        if not batch_expr_pc.numel():
            batch_expr_pc = torch.zeros(embeddings.size(0), self.n_genes, device=device)

        patch_ids_pc = (
            torch.cat(patch_ids_list, dim=0)
            if patch_ids_list
            else None
        )

        comp_cells = None
        if (
            allow_composition
            and self.use_celltype
            and out_cell_type is not None
            and out_cell_type.shape[0] > 0
        ):
            probs_cells = torch.softmax(out_cell_type.detach(), dim=1)
            n_patches = x_hist.size(0)
            comp_cells = torch.zeros(
                n_patches, probs_cells.shape[1], device=device
            )
            counts = torch.zeros(n_patches, device=device)
            comp_cells.index_add_(0, patch_assign, probs_cells)
            counts.index_add_(
                0,
                patch_assign,
                torch.ones(patch_assign.shape[0], device=device),
            )
            valid = counts > 0
            if valid.any():
                comp_cells[valid] = (
                    comp_cells[valid] / counts[valid].unsqueeze(-1)
                )

        self.last_aux_losses = {
            "vq_patch": vq_loss * self.vq_patch_loss_w,
            "vq_patch_idx": vq_patch_idx,
            "vq_patch_err": vq_patch_err,
            "comp_cells": comp_cells,
            "expr_ref_base": ref_weighted if self.use_avgexp and ref_orig is not None else None,
            "expr_ref_base_immune": ref_immune if self.use_avgexp and ref_orig is not None else None,
            "expr_ref_base_invasive": ref_invasive if self.use_avgexp and ref_orig is not None else None,
            "ref_weight_entropy": ref_weight_entropy,
            "ref_weight_entropy_immune": ref_weight_entropy_immune,
            "ref_weight_entropy_invasive": ref_weight_entropy_invasive,
            "expr_ref_gate": expr_ref_gate,
            "expr_ref_gate_immune": expr_ref_gate_immune,
            "expr_ref_gate_invasive": expr_ref_gate_invasive,
        }

        return (
            out_cell_type,
            out_map,
            batch_ct_pc,
            out_expr,
            out_expr_immune,
            out_expr_invasive,
            out_cell_type_expr,
            fv_cell_type_expr,
            out_cell_type_gt_expr,
            fv_cell_type_gt_expr,
            batch_expr_pc,
            comp_estimated,
            all_area[:embeddings.size(0)],
            patch_ids_pc,
        )

#!/usr/bin/env python3
"""CPU contracts for the runnable GHIST+ hurdle train/inference path."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.framework as base_framework  # noqa: E402
from dataio.dataset_input import get_region_spacing  # noqa: E402
from model.hurdle_distribution import (  # noqa: E402
    deterministic_hurdle_prediction,
    deterministic_threshold_control,
    hurdle_reconstruction_loss_from_model,
    masked_hurdle_truncated_gaussian_nll,
)
from model.hurdle_framework import HurdleFramework  # noqa: E402
from train import _unique_cell_rows_for_epoch  # noqa: E402
from utils.hurdle_evaluation import (  # noqa: E402
    aggregate_unique_hurdle_rows,
    cohort_gate_numpy,
    hurdle_gate_counts,
    hurdle_matrix_metrics,
    summarize_hurdle_gate_counts,
)


class TinyBackbone(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, **_kwargs):
        super().__init__()
        self.hd = nn.Conv2d(n_channels, 320, 1)
        self.h1 = nn.Conv2d(n_channels, 64, 1)
        self.seg = nn.Conv2d(320, n_classes, 1)

    def forward(self, x):
        hd = self.hd(x)
        h1 = self.h1(x)
        return self.seg(hd), hd, h1


class TinyFoundationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 12, 1)

    def forward(self, x):
        return self.proj(x)


class TinyFoundationAdapter(nn.Module):
    def __init__(self, pretrained=True, n_seg_classes=2):
        super().__init__()
        self.enc = TinyFoundationEncoder()
        self.proj_hd1 = nn.Conv2d(12, 320, 1)
        self.proj_h1 = nn.Conv2d(12, 64, 1)
        self.seg_head = nn.Conv2d(320, n_seg_classes, 1)

    def forward(self, x):
        feat = self.enc(x)
        hd1 = self.proj_hd1(feat)
        h1 = self.proj_h1(feat)
        return self.seg_head(hd1), hd1, h1


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def model_config(
    ablation_off=False,
    occurrence_ecrm=False,
    foundation_enabled=False,
    train_adapter=False,
    adapter_scope="all",
):
    return ns(
        hurdle=ns(
            enabled=True,
            sigma_init=0.5,
            sigma_floor=0.03,
            occurrence_ecrm=ns(
                enabled=occurrence_ecrm,
                ablation_off=False,
                gene_gate_init=0.25,
                gene_gate_max=1.0,
            ),
        ),
        foundation_model=ns(
            enabled=foundation_enabled,
            pretrained=False,
            train_adapter=train_adapter,
            adapter_scope=adapter_scope,
        ),
        legacy_backbone_frozen=False,
        celltype_priors_enabled=True,
        avgexp_temp_enabled=True,
        avgexp_temp=1.0,
        use_gt_ct_ref_weights=False,
        crossattn=False,
        use_avgexp_residual=True,
        avgexp_residual_scale=1.0,
        ct_prior_blend_alpha=1.0,
        expr_relu=True,
        vq_patch=ns(
            enabled=False,
            space="hidden",
            composition_requires_vq=False,
            inject_cell=False,
            loss_w=0.0,
        ),
        ecrm=ns(
            enabled=True,
            ablation_off=ablation_off,
            apply_to_embeddings=False,
            apply_to_ref_weights=False,
            apply_to_expr_residual=True,
            use_gt_ct=False,
            gate_h_from_embeddings=True,
            k_target=2,
            k_min=1,
            k_max=2,
            density_gamma=0.4,
            eta_max=0.0,
            gamma_perp_max=0.0,
            trust_floor=0.3,
            trust_scale=0.7,
            ct_conf_min=0.0,
            ct_same_type_only=False,
            depth=1,
            edge_dropout=0.0,
            message_dropout=0.0,
            residual_gate_init=-0.5,
        ),
    )


def make_model(seed=20260807, ablation_off=False, occurrence_ecrm=False):
    torch.manual_seed(seed)
    original = base_framework.LegacyBackbone
    base_framework.LegacyBackbone = TinyBackbone
    try:
        model = HurdleFramework(
            n_classes=3,
            n_genes=4,
            emb_dim=8,
            device=torch.device("cpu"),
            n_ref=3,
            use_avgexp=True,
            use_celltype=True,
            use_neighb=True,
            model_cfg=model_config(ablation_off, occurrence_ecrm),
        )
    finally:
        base_framework.LegacyBackbone = original
    return model


def make_control_model(
    seed=20260807,
    *,
    explicit_controls=True,
    vq_ablation_off=False,
    attention_ablation_off=False,
    reference_ablation_off=False,
    reference_mix_temperature=1.0,
):
    cfg = model_config()
    cfg.crossattn = True
    cfg.vq_patch.enabled = True
    cfg.vq_patch.n_codes = 8
    cfg.vq_patch.beta = 0.25
    cfg.vq_patch.use_cosine = True
    cfg.vq_patch.composition_requires_vq = True
    cfg.vq_patch.loss_w = 0.05
    if explicit_controls:
        cfg.vq_patch.ablation_off = vq_ablation_off
        cfg.attention_ablation_off = attention_ablation_off
        cfg.reference_ablation_off = reference_ablation_off
        cfg.reference_mix_temperature = reference_mix_temperature

    torch.manual_seed(seed)
    original = base_framework.LegacyBackbone
    base_framework.LegacyBackbone = TinyBackbone
    try:
        model = HurdleFramework(
            n_classes=3,
            n_genes=4,
            emb_dim=8,
            device=torch.device("cpu"),
            n_ref=3,
            use_avgexp=True,
            use_celltype=True,
            use_neighb=True,
            model_cfg=cfg,
        )
    finally:
        base_framework.LegacyBackbone = original
    return model


def make_foundation_model(seed=20260807, train_adapter=False, adapter_scope="all"):
    torch.manual_seed(seed)
    original = base_framework.Uni2HAdapter
    base_framework.Uni2HAdapter = TinyFoundationAdapter
    try:
        model = HurdleFramework(
            n_classes=3,
            n_genes=4,
            emb_dim=8,
            device=torch.device("cpu"),
            n_ref=3,
            use_avgexp=True,
            use_celltype=True,
            use_neighb=True,
            model_cfg=model_config(
                foundation_enabled=True,
                train_adapter=train_adapter,
                adapter_scope=adapter_scope,
            ),
        )
    finally:
        base_framework.Uni2HAdapter = original
    return model


def forward_inputs(edge_index=None):
    torch.manual_seed(91)
    nuclei = torch.zeros(1, 6, 6, dtype=torch.long)
    nuclei[0, :3, :3] = 1
    nuclei[0, :3, 3:] = 2
    nuclei[0, 3:, :] = 3
    if edge_index is None:
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
        )
    return {
        "x_hist": torch.randn(1, 3, 6, 6),
        "nuclei_mask": nuclei,
        "n_cells": torch.tensor([3]),
        "ref_orig": torch.tensor(
            [[0.0, 0.6, 1.0, 0.2], [0.4, 0.0, 0.7, 1.2], [0.9, 0.2, 0.0, 0.5]]
        ),
        "batch_ct": torch.tensor([[0, 1, 2]]),
        "batch_expr": torch.tensor(
            [[[0.0, 1.38, 2.19, 0.0], [1.38, 0.0, 1.38, 2.77], [2.19, 1.38, 0.0, 1.38]]]
        ),
        "patch_ids": torch.tensor([[101, 102, 103]]),
        "coords_cells": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]]),
        "cell_edge_index": edge_index,
        "cell_patch_ids": torch.zeros(3, dtype=torch.long),
    }


def test_likelihood_stability_mask_and_gradients():
    mu = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.5, 0.0]])
    sigma = torch.tensor([0.5, 0.5])
    loss = masked_hurdle_truncated_gaussian_nll(mu, logits, target, sigma)
    loss.backward()
    assert torch.isfinite(loss)
    assert mu.grad[0, 0] < 0
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0

    extreme_mu = torch.tensor([[-80.0, 80.0]], requires_grad=True)
    extreme_logits = torch.tensor([[-80.0, 80.0]], requires_grad=True)
    extreme = masked_hurdle_truncated_gaussian_nll(
        extreme_mu, extreme_logits, target, sigma
    )
    extreme.backward()
    assert torch.isfinite(extreme)
    assert torch.isfinite(extreme_mu.grad).all()

    target_nan = target.clone()
    target_nan[0, 1] = float("nan")
    mask = torch.tensor([[1.0, 0.0]])
    masked = masked_hurdle_truncated_gaussian_nll(
        mu.detach(), logits.detach(), target_nan, sigma, mask
    )
    first = masked_hurdle_truncated_gaussian_nll(
        mu.detach()[:, :1], logits.detach()[:, :1], target[:, :1], sigma[:1]
    )
    assert torch.isfinite(masked)
    assert torch.allclose(masked, first, atol=1e-7, rtol=0)


def test_full_framework_one_relu_gradient_checkpoint_and_graph_contract():
    model = make_model()
    model.train()
    inputs = forward_inputs()
    output = model(**inputs)
    aux = model.last_aux_losses
    assert torch.equal(output[3], F.relu(aux["hurdle_signed_mu"]))
    assert aux["hurdle_occurrence_logits"].shape == (3, 4)
    assert model.ecrm_graph_residual_head is None
    assert model.ecrm_graph_gene_gate_raw is None
    assert not hasattr(model, "set_occurrence_prior")

    loss = hurdle_reconstruction_loss_from_model(
        model, output[10], torch.ones_like(output[10]), expr_scale=2.0
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.count_nonzero(model.occurrence_head.mlp[-1].weight.grad) > 0
    assert torch.count_nonzero(model.mlp_avgexp_residual.mlp[-1].weight.grad) > 0
    assert torch.count_nonzero(model.hurdle_sigma_raw.grad) > 0

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.02
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model(**inputs)
    second_loss = hurdle_reconstruction_loss_from_model(
        model, second[10], torch.ones_like(second[10]), expr_scale=2.0
    )
    second_loss.backward()
    edge_grad = sum(
        float(p.grad.abs().sum())
        for p in model.ecrm.edge_mlp.parameters()
        if p.grad is not None
    )
    assert edge_grad > 0

    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = make_model()
    restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)
    restored.eval()
    model.eval()
    with torch.no_grad():
        expected = model(**inputs)
        actual = restored(**inputs)
    assert torch.equal(expected[3], actual[3])
    assert torch.equal(
        model.last_aux_losses["hurdle_occurrence_logits"],
        restored.last_aux_losses["hurdle_occurrence_logits"],
    )


def test_retained_nonexpression_supervision_cannot_change_magnitude():
    model = make_model(seed=31).train()
    output = model(**forward_inputs())
    # These are the retained hurdle-mode families: morphology CT/map and
    # composition. They must have no path into signed magnitude, q, sigma, or
    # residual-only ECRM.
    retained = F.cross_entropy(output[0], output[2]) + output[1].square().mean()
    if output[11] is not None:
        retained = retained + output[11].square().mean()
    model.zero_grad(set_to_none=True)
    retained.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.mlp_hist.parameters()
    )
    isolated_parameters = (
        list(model.mlp_avgexp_residual.parameters())
        + list(model.occurrence_head.parameters())
        + [model.hurdle_sigma_raw]
        + list(model.ecrm.parameters())
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in isolated_parameters
    )


def test_empty_identity_off_and_graph_sensitivity():
    empty = torch.empty((2, 0), dtype=torch.long)
    model = make_model(seed=12).eval()
    with torch.no_grad():
        model(**forward_inputs(edge_index=empty))
    aux = model.last_aux_losses
    assert torch.equal(
        aux["hurdle_occurrence_delta"], torch.zeros_like(aux["hurdle_occurrence_delta"])
    )
    assert torch.equal(
        aux["ecrm_graph_residual_delta"], torch.zeros_like(aux["ecrm_graph_residual_delta"])
    )
    assert aux["ecrm_explicit_empty_graph"] is True

    state = torch.randn(5, model.hidden_size)
    with torch.no_grad():
        logits, base, delta = model._occurrence_from_states(state, state)
    assert torch.equal(logits, base)
    assert torch.equal(delta, torch.zeros_like(delta))

    off = make_model(seed=12, ablation_off=True).eval()
    with torch.no_grad():
        off(**forward_inputs())
    assert torch.count_nonzero(off.last_aux_losses["hurdle_occurrence_delta"]) == 0
    assert torch.count_nonzero(off.last_aux_losses["ecrm_graph_residual_delta"]) == 0

    with torch.no_grad():
        model.occurrence_head.mlp[-1].weight.normal_(0, 0.1)
        model.mlp_avgexp_residual.mlp[-1].weight.normal_(0, 0.1)
        model(**forward_inputs())
    assert torch.count_nonzero(model.last_aux_losses["hurdle_occurrence_delta"]) > 0
    assert torch.count_nonzero(model.last_aux_losses["ecrm_graph_residual_delta"]) > 0


def test_topology_preserving_ablation_controls_and_reference_temperature():
    seed = 43
    legacy = make_control_model(seed=seed, explicit_controls=False).eval()
    full = make_control_model(seed=seed).eval()
    vq_off = make_control_model(seed=seed, vq_ablation_off=True).eval()
    attention_off = make_control_model(
        seed=seed, attention_ablation_off=True
    ).eval()
    reference_off = make_control_model(
        seed=seed, reference_ablation_off=True
    ).eval()
    temp_half = make_control_model(
        seed=seed, reference_mix_temperature=0.5
    ).eval()
    temp_two = make_control_model(
        seed=seed, reference_mix_temperature=2.0
    ).eval()

    # Flags never change initialization, modules, or optimizer-visible state.
    full_state = full.state_dict()
    for candidate in (
        legacy,
        vq_off,
        attention_off,
        reference_off,
        temp_half,
        temp_two,
    ):
        candidate_state = candidate.state_dict()
        assert candidate_state.keys() == full_state.keys()
        for key, value in full_state.items():
            assert torch.equal(value, candidate_state[key]), key
    assert vq_off.vq_patch is not None
    assert attention_off.refine_expr is not None
    assert reference_off.refine_expr is not None

    inputs = forward_inputs()
    captured = {}

    def capture_embed_hist(name):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()
        return hook

    full_hook = full.embed_hist.register_forward_pre_hook(capture_embed_hist("full"))
    off_hook = vq_off.embed_hist.register_forward_pre_hook(capture_embed_hist("off"))
    with torch.no_grad():
        legacy_output = legacy(**inputs)
        full_output = full(**inputs)
        vq_off_output = vq_off(**inputs)
    full_hook.remove()
    off_hook.remove()

    # Missing flags and explicit defaults are a bitwise-identical FULL path.
    for left, right in zip(legacy_output, full_output):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)

    # Grouped VQ+composition OFF skips VQ and both composition products while
    # retaining the raw (nonzero) tile features in every cell embedding.
    assert torch.equal(captured["full"], captured["off"])
    assert torch.count_nonzero(captured["off"][:, 384:]) > 0
    assert full.last_aux_losses["vq_patch_idx"] is not None
    assert full_output[11] is not None
    assert vq_off.last_aux_losses["vq_patch_idx"] is None
    assert vq_off.last_aux_losses["vq_patch_err"] is None
    assert torch.equal(
        vq_off.last_aux_losses["vq_patch"],
        torch.zeros_like(vq_off.last_aux_losses["vq_patch"]),
    )
    assert vq_off_output[11] is None
    assert vq_off.last_aux_losses["comp_cells"] is None

    # Attention OFF leaves all refiners instantiated but never invokes them;
    # changing only their parameters therefore cannot change any expression.
    attention_calls = []
    handles = [
        module.register_forward_hook(
            lambda _module, _args, _output: attention_calls.append(True)
        )
        for module in (
            attention_off.refine_expr,
            attention_off.refine_expr_immune,
            attention_off.refine_expr_invasive,
        )
    ]
    with torch.no_grad():
        attention_before = attention_off(**inputs)
        for module in (
            attention_off.refine_expr,
            attention_off.refine_expr_immune,
            attention_off.refine_expr_invasive,
        ):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter))
        attention_after = attention_off(**inputs)
    for handle in handles:
        handle.remove()
    assert attention_calls == []
    for index in (3, 4, 5):
        assert torch.equal(attention_before[index], attention_after[index])

    # Reference OFF is invariant to every scRNA reference value. Attention is
    # still active and receives an exact-zero reference tensor in all branches.
    attention_ref_inputs = []

    def capture_ref(_module, args):
        attention_ref_inputs.append(args[1].detach().clone())

    handles = [
        module.register_forward_pre_hook(capture_ref)
        for module in (
            reference_off.refine_expr,
            reference_off.refine_expr_immune,
            reference_off.refine_expr_invasive,
        )
    ]
    changed_inputs = dict(inputs)
    changed_inputs["ref_orig"] = inputs["ref_orig"] * -17.0 + 31.0
    with torch.no_grad():
        reference_first = reference_off(**inputs)
        reference_second = reference_off(**changed_inputs)
    for handle in handles:
        handle.remove()
    assert len(attention_ref_inputs) == 6
    assert all(torch.count_nonzero(value) == 0 for value in attention_ref_inputs)
    for index in (3, 4, 5):
        assert torch.equal(reference_first[index], reference_second[index])
    for key in (
        "expr_ref_base",
        "expr_ref_base_immune",
        "expr_ref_base_invasive",
    ):
        assert torch.count_nonzero(reference_off.last_aux_losses[key]) == 0

    # The new temperature acts on the final resolved simplex. T=1 returns the
    # exact tensor object; 0.5 sharpens and 2.0 softens its entropy.
    simplex = torch.tensor(
        [[0.70, 0.20, 0.10], [0.55, 0.35, 0.10]], dtype=torch.float32
    )
    assert full._apply_reference_mix_temperature(simplex) is simplex
    half = temp_half._apply_reference_mix_temperature(simplex)
    two = temp_two._apply_reference_mix_temperature(simplex)

    def entropy(value):
        return -(value * value.clamp_min(1e-8).log()).sum(dim=1).mean()

    assert torch.allclose(half.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0)
    assert torch.allclose(two.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0)
    assert entropy(half) < entropy(simplex) < entropy(two)
    learned = torch.full_like(simplex, 1.0 / 3.0)
    assert torch.equal(
        temp_half._resolve_ref_weights(learned, simplex, None), half
    )


def test_occurrence_ecrm_isolated_residual_contract():
    seed = 73
    baseline = make_model(seed=seed).eval()
    candidate = make_model(seed=seed, occurrence_ecrm=True).eval()

    # Adding the optional branch happens after all baseline initialization and
    # must not perturb any pre-existing parameter or buffer.
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in baseline_state.items():
        assert key in candidate_state
        assert torch.equal(value, candidate_state[key]), key

    inputs = forward_inputs()
    with torch.no_grad():
        baseline(**inputs)
        baseline_mu = baseline.last_aux_losses["hurdle_signed_mu"].clone()
        baseline_q = baseline.last_aux_losses["hurdle_occurrence_logits"].clone()
        candidate(**inputs)
        candidate_mu = candidate.last_aux_losses["hurdle_signed_mu"].clone()
        candidate_q = candidate.last_aux_losses["hurdle_occurrence_logits"].clone()
    assert torch.equal(candidate_mu, baseline_mu)
    assert torch.equal(candidate_q, baseline_q)
    assert candidate.occurrence_ecrm.ct_same_type_only is False
    assert candidate.occurrence_ecrm.ct_conf_min == 0.0
    assert torch.allclose(
        candidate.last_aux_losses["hurdle_occurrence_ecrm_gene_gate"],
        torch.full((candidate.n_genes,), 0.25),
        atol=1e-7,
        rtol=0,
    )

    # Make the already-existing occurrence readout non-constant so graph
    # sensitivity and isolation can be tested without adding a new loss.
    with torch.no_grad():
        candidate.occurrence_head.mlp[-1].weight.normal_(0.0, 0.05)
        candidate.occurrence_head.mlp[-1].bias.zero_()
        candidate(**inputs)
        q_full = candidate.last_aux_losses["hurdle_occurrence_logits"].clone()
        mu_full = candidate.last_aux_losses["hurdle_signed_mu"].clone()
        magnitude_delta = candidate.last_aux_losses[
            "ecrm_graph_residual_delta"
        ].clone()

        candidate.occurrence_ecrm_ablation_off = True
        candidate(**inputs)
        q_off = candidate.last_aux_losses["hurdle_occurrence_logits"].clone()
        mu_off = candidate.last_aux_losses["hurdle_signed_mu"].clone()
        assert candidate.last_aux_losses["occurrence_ecrm_graph_state"] is None
        magnitude_delta_off = candidate.last_aux_losses[
            "ecrm_graph_residual_delta"
        ].clone()
        candidate.occurrence_ecrm_ablation_off = False

    assert torch.count_nonzero(q_full - q_off) > 0
    assert torch.equal(mu_full, mu_off)
    assert torch.equal(magnitude_delta, magnitude_delta_off)

    # The graph correction itself has exactly zero gradient into the shared
    # morphology state; only the self/base occurrence score may train it.
    probe_state = torch.randn(5, candidate.hidden_size, requires_grad=True)
    probe_graph = probe_state.detach() + torch.randn_like(probe_state) * 0.2
    _, _, probe_delta = candidate._occurrence_from_states(
        probe_state, probe_graph
    )
    probe_delta.square().mean().backward()
    assert probe_state.grad is None or torch.count_nonzero(probe_state.grad) == 0

    # External/ground-truth CT and reference expression cannot enter the new
    # occurrence graph.  For fixed image/coords/edges, q is identical.
    changed_inputs = dict(inputs)
    changed_inputs["batch_ct"] = torch.tensor([[2, 2, 0]])
    changed_inputs["ref_orig"] = torch.flip(inputs["ref_orig"], dims=[0, 1])
    with torch.no_grad():
        candidate(**inputs)
        q_original = candidate.last_aux_losses["hurdle_occurrence_logits"].clone()
        candidate(**changed_inputs)
        q_changed = candidate.last_aux_losses["hurdle_occurrence_logits"].clone()
        candidate(**forward_inputs(edge_index=torch.empty((2, 0), dtype=torch.long)))
        empty_delta = candidate.last_aux_losses["hurdle_occurrence_delta"].clone()
        assert candidate.last_aux_losses["occurrence_ecrm_graph_state"] is None
    assert torch.equal(q_original, q_changed)
    assert torch.equal(empty_delta, torch.zeros_like(empty_delta))

    # The existing occurrence likelihood trains the dedicated branch, while
    # its detached input prevents it from changing the magnitude ECRM/head.
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    candidate(**inputs)
    logits = candidate.last_aux_losses["hurdle_occurrence_logits"]
    labels = (inputs["batch_expr"].reshape(3, 4) > 0).to(logits.dtype)
    F.binary_cross_entropy_with_logits(logits, labels).backward()
    occurrence_edge_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in candidate.occurrence_ecrm.edge_mlp.parameters()
        if parameter.grad is not None
    )
    magnitude_edge_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in candidate.ecrm.edge_mlp.parameters()
        if parameter.grad is not None
    )
    assert occurrence_edge_grad > 0
    assert magnitude_edge_grad == 0.0
    assert candidate.occurrence_ecrm_gene_gate_raw.grad is not None
    assert torch.count_nonzero(candidate.occurrence_ecrm_gene_gate_raw.grad) > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in candidate.mlp_avgexp_residual.parameters()
    )

    candidate.zero_grad(set_to_none=True)
    candidate(**inputs)
    candidate.last_aux_losses["hurdle_signed_mu"].square().mean().backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in candidate.occurrence_ecrm.parameters()
    )
    assert (
        candidate.occurrence_ecrm_gene_gate_raw.grad is None
        or torch.count_nonzero(candidate.occurrence_ecrm_gene_gate_raw.grad) == 0
    )

    payload = io.BytesIO()
    torch.save(candidate.state_dict(), payload)
    payload.seek(0)
    restored = make_model(seed=999, occurrence_ecrm=True)
    restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)


def test_foundation_encoder_frozen_random_adapter_trainable_contract():
    seed = 107
    frozen = make_foundation_model(seed=seed, train_adapter=False).eval()
    trainable = make_foundation_model(seed=seed, train_adapter=True).eval()

    # The switch changes only autograd/mode boundaries, never initialization or
    # state-dict structure.
    for key, value in frozen.state_dict().items():
        assert key in trainable.state_dict()
        assert torch.equal(value, trainable.state_dict()[key]), key
    assert all(not parameter.requires_grad for parameter in frozen.cnn.parameters())
    assert all(
        not parameter.requires_grad for parameter in trainable.cnn.enc.parameters()
    )
    adapter_parameters = (
        list(trainable.cnn.proj_hd1.parameters())
        + list(trainable.cnn.proj_h1.parameters())
        + list(trainable.cnn.seg_head.parameters())
    )
    assert all(parameter.requires_grad for parameter in adapter_parameters)

    trainable.train()
    assert trainable.cnn.enc.training is False
    assert trainable.cnn.proj_hd1.training is True
    assert trainable.cnn.proj_h1.training is True
    assert trainable.cnn.seg_head.training is True
    trainable.eval()
    assert trainable.cnn.enc.training is False
    assert trainable.cnn.proj_hd1.training is False

    inputs = forward_inputs()
    with torch.no_grad():
        frozen_output = frozen(**inputs)
        trainable_output = trainable(**inputs)
    for left, right in zip(frozen_output[:11], trainable_output[:11]):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)

    # Existing hurdle likelihood reaches both random feature projections but
    # never the pretrained encoder or segmentation-only head.
    trainable.train()
    trainable.zero_grad(set_to_none=True)
    output = trainable(**inputs)
    hurdle_reconstruction_loss_from_model(
        trainable,
        output[10],
        torch.ones_like(output[10]),
        expr_scale=2.0,
    ).backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in trainable.cnn.proj_hd1.parameters()
        if parameter.grad is not None
    ) > 0
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in trainable.cnn.proj_h1.parameters()
        if parameter.grad is not None
    ) > 0
    assert all(parameter.grad is None for parameter in trainable.cnn.enc.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in trainable.cnn.seg_head.parameters()
    )

    # The already-existing map CE trains proj_hd1 and seg_head, not proj_h1 or
    # the frozen encoder.
    trainable.zero_grad(set_to_none=True)
    output = trainable(**inputs)
    F.cross_entropy(output[1], inputs["nuclei_mask"]).backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in trainable.cnn.proj_hd1.parameters()
        if parameter.grad is not None
    ) > 0
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in trainable.cnn.seg_head.parameters()
        if parameter.grad is not None
    ) > 0
    assert all(parameter.grad is None for parameter in trainable.cnn.enc.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in trainable.cnn.proj_h1.parameters()
    )

    encoder_before = {
        key: value.detach().clone() for key, value in trainable.cnn.enc.state_dict().items()
    }
    hd1_before = trainable.cnn.proj_hd1.weight.detach().clone()
    h1_before = trainable.cnn.proj_h1.weight.detach().clone()
    seg_before = trainable.cnn.seg_head.weight.detach().clone()
    optimizer = torch.optim.SGD(
        [parameter for parameter in trainable.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    output = trainable(**inputs)
    total = hurdle_reconstruction_loss_from_model(
        trainable,
        output[10],
        torch.ones_like(output[10]),
        expr_scale=2.0,
    ) + F.cross_entropy(output[1], inputs["nuclei_mask"])
    total.backward()
    optimizer.step()
    assert all(
        torch.equal(value, trainable.cnn.enc.state_dict()[key])
        for key, value in encoder_before.items()
    )
    assert not torch.equal(hd1_before, trainable.cnn.proj_hd1.weight)
    assert not torch.equal(h1_before, trainable.cnn.proj_h1.weight)
    assert not torch.equal(seg_before, trainable.cnn.seg_head.weight)

    payload = io.BytesIO()
    torch.save(trainable.state_dict(), payload)
    payload.seek(0)
    restored = make_foundation_model(seed=999, train_adapter=True)
    restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)

    h1_only = make_foundation_model(
        seed=seed,
        train_adapter=True,
        adapter_scope="h1",
    ).train()
    assert all(
        parameter.requires_grad for parameter in h1_only.cnn.proj_h1.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in h1_only.cnn.proj_hd1.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in h1_only.cnn.seg_head.parameters()
    )
    h1_only.zero_grad(set_to_none=True)
    h1_output = h1_only(**inputs)
    hurdle_reconstruction_loss_from_model(
        h1_only,
        h1_output[10],
        torch.ones_like(h1_output[10]),
        expr_scale=2.0,
    ).backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in h1_only.cnn.proj_h1.parameters()
        if parameter.grad is not None
    ) > 0
    assert all(parameter.grad is None for parameter in h1_only.cnn.enc.parameters())
    assert all(parameter.grad is None for parameter in h1_only.cnn.proj_hd1.parameters())
    assert all(parameter.grad is None for parameter in h1_only.cnn.seg_head.parameters())


def test_cohort_gate_q_mean_dedup_and_metrics():
    mu = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    q = torch.full((5, 1), 0.4)
    logits = torch.logit(q)
    exported = deterministic_hurdle_prediction(mu, logits)
    control = deterministic_threshold_control(mu, logits)
    assert int((exported > 0).sum()) == 2
    assert int((control > 0).sum()) == 0
    assert torch.equal((exported[:, 0] > 0), torch.tensor([True, True, False, False, False]))

    ids = np.array([10, 10, 20])
    logits_dup = np.array([[np.log(0.1 / 0.9)], [0.0], [np.log(0.8 / 0.2)]])
    unique = aggregate_unique_hurdle_rows(ids, np.ones((3, 1)), logits_dup)
    mean_q = 1.0 / (1.0 + np.exp(-unique[2]))
    assert np.allclose(mean_q[:, 0], [0.3, 0.8], atol=1e-6)
    gated = cohort_gate_numpy(unique[1], unique[2])
    assert gated.shape == (2, 1)

    target = np.array([[0.0], [2.0]])
    metrics = hurdle_matrix_metrics(gated, target)
    assert "hurdle_mean_per_gene_w1" in metrics
    assert "hurdle_mean_per_gene_zero_gap" in metrics
    assert "hurdle_distribution_score" not in metrics

    # A top-K-selected cell can remain zero because its signed magnitude is <=0.
    signed_short = torch.tensor([[-1.0], [2.0], [3.0]])
    logits_short = torch.logit(torch.full((3, 1), 0.6))  # requested K=2
    pred_short = deterministic_hurdle_prediction(signed_short, logits_short).numpy()
    requested, effective = hurdle_gate_counts(pred_short, logits_short.numpy())
    summary = summarize_hurdle_gate_counts(requested, effective)
    assert requested.tolist() == [2]
    assert effective.tolist() == [1]
    assert summary["hurdle_positive_shortfall_total"] == 1
    assert summary["hurdle_effective_positive_fraction_of_requested"] == 0.5

    # The gate and its diagnostic must use identical float32 Torch reduction
    # semantics.  NumPy can put this sum just below 1.5 while Torch rounds it
    # to 2, causing a false "effective exceeds requested" failure.
    boundary_mu = np.ones((3, 1), dtype=np.float32)
    boundary_logits = np.full((3, 1), -1e-7, dtype=np.float64)
    boundary_pred = cohort_gate_numpy(boundary_mu, boundary_logits)
    requested, effective = hurdle_gate_counts(boundary_pred, boundary_logits)
    assert requested.tolist() == [2]
    assert effective.tolist() == [2]


def test_unique_cell_epoch_supervision():
    seen = set()
    first = _unique_cell_rows_for_epoch(torch.tensor([1, 2, 1]), 3, seen)
    second = _unique_cell_rows_for_epoch(torch.tensor([2, 3, 0]), 3, seen)
    other_slide = _unique_cell_rows_for_epoch(torch.tensor([1]), 4, seen)
    assert torch.equal(first, torch.tensor([True, True, False]))
    assert torch.equal(second, torch.tensor([False, True, False]))
    assert torch.equal(other_slide, torch.tensor([True]))
    assert seen == {(3, 1), (3, 2), (3, 3), (4, 1)}


def test_focused_config_and_train_inference_policy_contracts():
    cfg = json.loads((ROOT / "configs/breast2_ablation9/full.json").read_text())
    assert cfg["model"]["hurdle"] == {
        "enabled": True,
        "sigma_init": 0.5,
        "sigma_floor": 0.03,
    }
    assert cfg["model"]["avgexp_residual_scale"] == 1.0
    assert cfg["model"]["ecrm"]["apply_to_embeddings"] is False
    assert cfg["model"]["ecrm"]["apply_to_ref_weights"] is False
    assert cfg["model"]["ecrm"]["apply_to_expr_residual"] is True
    assert cfg["training"]["uniform_sampler"] is True
    assert cfg["training"]["use_expr_baseline"] is False
    assert cfg["training"]["expr_var_penalty_weight"] == 0.0
    assert cfg["training"]["panel_completion_loss_weight"] == 0.0

    val = get_region_spacing(100, "val", [0.0, 0.2])
    train = get_region_spacing(100, "train", [0.0, 0.2])
    assert len(val) == 20 and len(train) == 80
    assert np.intersect1d(val, train).size == 0
    assert np.union1d(val, train).size == 100

    train_source = (ROOT / "train.py").read_text()
    assert "hurdle_reconstruction_loss_from_model" in train_source
    assert "loss_expression_val = loss_expr_val" in train_source
    assert "expression-conditioned auxiliaries reshape mu or q" in train_source
    assert "if (not uniform_sampler) and immune_sampler_boost" in train_source
    assert "not use_batch_sampler and not uniform_sampler" in train_source
    assert '"drop_last": not uniform_sampler' in train_source
    assert "_unique_cell_rows_for_epoch" in train_source
    assert "select_svg_joint_rank_checkpoint" in train_source
    assert 'GHIST_RUN_ROLE", "FULL"' in train_source
    assert '"selection_status": "deferred_to_full"' in train_source

    inference_source = (ROOT / "tools/inference.py").read_text()
    assert "aggregate_unique_hurdle_rows" in inference_source
    assert "cohort_gate_numpy" in inference_source
    assert "mean_q_and_signed_mu_then_unique_slide_topk" in inference_source
    assert "hurdle_effective_positive" in inference_source
    assert "hurdle_positive_shortfall" in inference_source


def run():
    tests = [
        test_likelihood_stability_mask_and_gradients,
        test_full_framework_one_relu_gradient_checkpoint_and_graph_contract,
        test_retained_nonexpression_supervision_cannot_change_magnitude,
        test_empty_identity_off_and_graph_sensitivity,
        test_topology_preserving_ablation_controls_and_reference_temperature,
        test_occurrence_ecrm_isolated_residual_contract,
        test_foundation_encoder_frozen_random_adapter_trainable_contract,
        test_cohort_gate_q_mean_dedup_and_metrics,
        test_unique_cell_epoch_supervision,
        test_focused_config_and_train_inference_policy_contracts,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} runnable hurdle integration test groups")


if __name__ == "__main__":
    run()

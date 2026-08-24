#!/usr/bin/env python3
"""Focused contracts for legacy panel completion on the hurdle core."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.hurdle_distribution import (  # noqa: E402
    masked_hurdle_truncated_gaussian_nll,
)
from model.panel_completion import PanelCompletionHead  # noqa: E402
import utils.gene_mask_imputer as imputer_task  # noqa: E402
import utils.hurdle_evaluation as hurdle_evaluation  # noqa: E402
from utils.hurdle_evaluation import (  # noqa: E402
    evaluate_hurdle_validation,
    new_panel_completion_holdout_accumulator,
    panel_completion_holdout_metrics,
    sanitize_hurdle_panel_holdout_inputs,
    update_panel_completion_holdout_accumulator,
)


def test_original_panel_head_architecture_and_morph_gate():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    head = PanelCompletionHead(
        n_genes=3,
        hidden_dim=7,
        dropout=0.0,
        use_morph=True,
        morph_gate_init=-2.0,
    ).eval()
    after = torch.random.get_rng_state()

    assert not torch.equal(before, after), "ordinary initialization must advance RNG"
    assert head.net[0].in_features == 6
    assert head.net[0].out_features == 7
    assert head.net[3].in_features == 7
    assert head.net[3].out_features == 3
    assert torch.equal(head.morph_gate, torch.full((3,), -2.0))
    assert set(head.state_dict()) == {
        "morph_gate",
        "net.0.weight",
        "net.0.bias",
        "net.3.weight",
        "net.3.bias",
    }

    delta_obs = torch.tensor([[0.2, 0.0, -0.4], [0.0, 0.7, 0.0]])
    mask_obs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    delta_morph = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.25, -0.5]])
    expected = head.net(torch.cat([delta_obs, mask_obs], dim=1))
    expected = expected + torch.sigmoid(head.morph_gate).view(1, -1) * delta_morph
    assert torch.equal(head(delta_obs, mask_obs, delta_morph), expected)


def test_original_completed_expression_uses_core_q_and_hurdle_nll():
    torch.manual_seed(19)
    head = PanelCompletionHead(3, hidden_dim=8, use_morph=True)
    expression = torch.tensor(
        [[0.0, 0.8, 1.2], [0.7, 0.2, 0.0], [1.1, 0.4, 0.5]]
    )
    reference = torch.full_like(expression, 0.25)
    morphology = torch.tensor(
        [[0.3, 0.9, 0.7], [0.5, 0.3, 0.2], [0.8, 0.5, 0.4]]
    )
    mask_obs = torch.tensor(
        [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    mask_target = 1.0 - mask_obs
    occurrence_logits = torch.zeros_like(expression, requires_grad=True)

    delta_hat = head(
        (expression - reference) * mask_obs,
        mask_obs,
        morphology - reference,
    )
    pred_completed = torch.relu(reference + delta_hat)
    pred_completed = mask_obs * expression + (1.0 - mask_obs) * pred_completed
    loss = masked_hurdle_truncated_gaussian_nll(
        pred_completed / 2.0,
        occurrence_logits,
        expression / 2.0,
        sigma=torch.full((3,), 0.5),
        mask=mask_target,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert head.net[3].weight.grad is not None
    assert torch.count_nonzero(head.net[3].weight.grad) > 0
    assert occurrence_logits.grad is not None
    assert torch.count_nonzero(occurrence_logits.grad) > 0
    assert not hasattr(head, "occurrence_residual_head")


def test_legacy_holdout_accumulator_keeps_duplicate_emitted_rows():
    accumulator = new_panel_completion_holdout_accumulator()
    holdout = np.array([1.0, 0.0], dtype=np.float32)
    prediction = np.array([[1.0, 99.0], [2.0, 99.0]], dtype=np.float32)
    target = np.array([[1.0, -99.0], [2.0, -99.0]], dtype=np.float32)
    update_panel_completion_holdout_accumulator(
        accumulator, 7, prediction, target, holdout
    )
    update_panel_completion_holdout_accumulator(
        accumulator, 7, prediction, target, holdout
    )
    metrics = panel_completion_holdout_metrics(
        accumulator,
        {7: holdout},
        gene_names=["A", "B"],
    )

    assert accumulator["holdout_n"] == 4.0
    assert accumulator["stats_by_slide"][7]["count"].tolist() == [4.0, 0.0]
    assert np.isclose(metrics["holdout_mse"], 0.0)
    assert np.isclose(metrics["holdout_mae"], 0.0)
    assert np.isclose(metrics["holdout_gene_pooled_median"], 1.0)
    assert metrics["holdout_gene_pooled_n_genes"] == 1
    assert not any("hurdle" in key for key in metrics)


def test_fixed_holdout_sanitizer_preserves_raw_target():
    raw = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    raw_copy = raw.clone()
    reference = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    sanitized = sanitize_hurdle_panel_holdout_inputs(
        raw,
        torch.tensor([2]),
        torch.tensor([[0, 1]]),
        reference,
        np.array([1.0, 0.0]),
    )

    assert torch.equal(raw, raw_copy)
    assert torch.equal(
        sanitized,
        torch.tensor([[[0.25, 2.0], [0.75, 4.0]]]),
    )


class _PanelEvaluationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.completion_head = PanelCompletionHead(2, hidden_dim=5, use_morph=True)
        with torch.no_grad():
            for parameter in self.completion_head.net.parameters():
                parameter.zero_()
            self.completion_head.morph_gate.fill_(100.0)
        self.forward_expression = None
        self.last_aux_losses = {}

    def forward(
        self,
        batch_he_img,
        batch_nuclei,
        batch_n_cells,
        expr_ref,
        batch_ct,
        batch_expr,
        **_kwargs,
    ):
        del batch_he_img, batch_nuclei, batch_n_cells, expr_ref, batch_ct
        self.forward_expression = batch_expr.detach().clone()
        morphology = torch.tensor(
            [[2.0, 1.0], [4.0, 2.0], [2.0, 0.0]],
            device=batch_expr.device,
        )
        signed_mu = torch.tensor(
            [[-5.0, 1.0], [-5.0, 2.0], [-5.0, 0.0]],
            device=batch_expr.device,
        )
        logits = torch.full_like(signed_mu, -100.0)
        self.last_aux_losses = {
            "expr_ref_base": torch.tensor(
                [[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]],
                device=batch_expr.device,
            ),
            "hurdle_signed_mu": signed_mu,
            "hurdle_occurrence_logits": logits,
        }
        empty = torch.empty(0, device=batch_expr.device)
        output = [empty for _ in range(14)]
        output[0] = empty
        output[2] = empty
        output[3] = morphology
        output[13] = torch.tensor([10, 20, 10], device=batch_expr.device)
        return tuple(output)


def _panel_evaluation_batch():
    expression = torch.tensor([[[2.0, 1.0], [4.0, 2.0], [2.0, 0.0]]])
    expression_mask = torch.ones_like(expression)
    expression_mask[:, :, 0] = 0.0
    return expression, (
        torch.zeros(1, 4, 4, dtype=torch.long),
        torch.zeros(1, 4, 4, dtype=torch.long),
        torch.zeros(1, 3, 4, 4),
        expression,
        torch.tensor([3]),
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[10, 20, 10]], dtype=torch.long),
        expression_mask,
        torch.tensor([7]),
    )


def test_evaluator_uses_direct_panel_completion_only_for_holdout_metrics():
    original_build_cell_graph = hurdle_evaluation.graph_utils.build_cell_graph
    hurdle_evaluation.graph_utils.build_cell_graph = (
        lambda *_args, **_kwargs: SimpleNamespace(
            coords=torch.zeros(3, 2),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            patch_index=torch.zeros(3, dtype=torch.long),
        )
    )
    try:
        raw, batch = _panel_evaluation_batch()
        raw_copy = raw.clone()
        holdouts = {7: np.array([1.0, 0.0], dtype=np.float32)}
        enabled_model = _PanelEvaluationModel()
        enabled = evaluate_hurdle_validation(
            enabled_model,
            [batch],
            torch.tensor([[0.5, 0.6]]),
            torch.device("cpu"),
            n_classes=1,
            expr_scale=2.0,
            gene_names=["A", "B"],
            panel_completion_enabled=True,
            holdout_mask_by_slide=holdouts,
        )
        disabled_model = _PanelEvaluationModel()
        disabled = evaluate_hurdle_validation(
            disabled_model,
            [batch],
            torch.tensor([[0.5, 0.6]]),
            torch.device("cpu"),
            n_classes=1,
            expr_scale=2.0,
            gene_names=["A", "B"],
            panel_completion_enabled=False,
            holdout_mask_by_slide=holdouts,
        )

        assert torch.equal(raw, raw_copy)
        np.testing.assert_allclose(
            enabled_model.forward_expression[0, :, 0].numpy(),
            [0.5, 0.5, 0.5],
        )
        assert np.isclose(enabled["holdout_mse"], 0.0)
        assert np.isclose(enabled["holdout_gene_pooled_median"], 1.0)
        assert "holdout_mse" not in disabled
        for key in (
            "pearson_gene_pooled_mean",
            "hurdle_mean_per_gene_w1",
            "hurdle_pred_zero_fraction",
            "hurdle_target_zero_fraction",
        ):
            assert np.isclose(enabled[key], disabled[key], equal_nan=True)
    finally:
        hurdle_evaluation.graph_utils.build_cell_graph = original_build_cell_graph


def test_evaluator_falls_back_to_morphology_when_panel_head_fails():
    class FailingHead(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise RuntimeError("expected test failure")

    original_build_cell_graph = hurdle_evaluation.graph_utils.build_cell_graph
    hurdle_evaluation.graph_utils.build_cell_graph = (
        lambda *_args, **_kwargs: SimpleNamespace(
            coords=torch.zeros(3, 2),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            patch_index=torch.zeros(3, dtype=torch.long),
        )
    )
    try:
        _, batch = _panel_evaluation_batch()
        holdouts = {7: np.array([1.0, 0.0], dtype=np.float32)}
        for replacement in (None, FailingHead()):
            model = _PanelEvaluationModel()
            model.completion_head = replacement
            metrics = evaluate_hurdle_validation(
                model,
                [batch],
                torch.tensor([[0.5, 0.6]]),
                torch.device("cpu"),
                n_classes=1,
                expr_scale=2.0,
                gene_names=["A", "B"],
                panel_completion_enabled=True,
                holdout_mask_by_slide=holdouts,
            )
            assert np.isclose(metrics["holdout_mse"], 0.0)
            assert np.isclose(metrics["holdout_gene_pooled_median"], 1.0)
    finally:
        hurdle_evaluation.graph_utils.build_cell_graph = original_build_cell_graph


def test_task_config_and_training_source_keep_original_b_contract():
    config = json.loads((ROOT / "configs/config_gene_mask_imputer.json").read_text())
    base_config = json.loads((ROOT / "configs/config_multi_breast.json").read_text())
    task = config["gene_mask_imputer"]
    assert task["copy_observed"] is True
    assert task["create_fixed_gene_csv_if_missing"] is True
    assert "loss_weight" not in task
    for training in (config["training"], base_config["training"]):
        assert "panel_completion_loss_weight" not in training
        assert "panel_hide_frac" not in training
        assert "panel_use_natural_missing" not in training

    resolved = imputer_task.resolve_config(
        SimpleNamespace(enabled=True), repo_root=str(ROOT)
    )
    assert resolved["copy_observed"] is True
    assert resolved["zero_masked_gene_avgexp"] is False
    assert "loss_weight" not in resolved

    source = (ROOT / "train.py").read_text()
    assert "panel_hide_in_forward = False" in source
    assert "torch.rand_like(batch_expr_mask_pc) < panel_hide_frac" in source
    assert "panel_completion.PanelCompletionHead(" in source
    assert "HurdlePanelCompletionHead" not in source
    assert "pred_completed_model = F.relu(ref_base_model + delta_hat)" in source
    assert "hidden_mask_flat=mask_hide_pc" not in source


def run():
    tests = [
        test_original_panel_head_architecture_and_morph_gate,
        test_original_completed_expression_uses_core_q_and_hurdle_nll,
        test_legacy_holdout_accumulator_keeps_duplicate_emitted_rows,
        test_fixed_holdout_sanitizer_preserves_raw_target,
        test_evaluator_uses_direct_panel_completion_only_for_holdout_metrics,
        test_evaluator_falls_back_to_morphology_when_panel_head_fails,
        test_task_config_and_training_source_keep_original_b_contract,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} legacy-panel-on-hurdle test groups")


if __name__ == "__main__":
    run()

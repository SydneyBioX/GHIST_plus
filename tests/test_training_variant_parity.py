"""Static guards for shared GHIST+ training variants.

These tests intentionally parse source/configuration files without importing
``train``.  Importing the training module pulls in the GPU and image stack,
which is unnecessary for enforcing structural parity.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "config_multi_breast.json"
GENE_MASK_CONFIG = REPO_ROOT / "configs" / "config_gene_mask_imputer.json"

GENE_MASK_TASK_KEYS = {
    "enabled",
    "mask_n_genes",
    "mask_strategy",
    "svg_knn_k",
    "svg_sample_cap",
    "fixed_gene_csv_dir",
    "create_fixed_gene_csv_if_missing",
    "random_mask_frac",
    "use_morph",
    "copy_observed",
    "mask_seed",
}


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_gene_mask_config_is_base_config_plus_one_allowlisted_task_block():
    """No model, loader, loss, cohort, seed, or path drift is permitted."""

    base = _load_json(BASE_CONFIG)
    gene_mask = _load_json(GENE_MASK_CONFIG)

    assert set(gene_mask) - set(base) == {"gene_mask_imputer"}
    task = gene_mask.pop("gene_mask_imputer")
    assert gene_mask == base
    assert set(task) == GENE_MASK_TASK_KEYS

    # Pin the Figure 3 task definition and keep its loss out of shared training
    # settings.  The common config comparison above guarantees that the stain
    # reference and source panels are also byte-for-byte equivalent as values.
    assert task["enabled"] is True
    assert task["mask_strategy"] == "fixed_giotto_csv_per_slide"
    assert task["fixed_gene_csv_dir"] == "configs/gene_mask_imputer_fixed_giotto"
    assert task["create_fixed_gene_csv_if_missing"] is True
    assert task["random_mask_frac"] == 0.3
    assert task["use_morph"] is True
    assert task["copy_observed"] is True
    assert "loss_weight" not in task
    assert "panel_completion_loss_weight" not in gene_mask["training"]
    assert "panel_hide_frac" not in gene_mask["training"]
    assert "panel_use_natural_missing" not in gene_mask["training"]


def _without_docstring(module: ast.Module):
    body = list(module.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _assert_thin_variant_wrapper(filename: str, variant_name: str):
    path = REPO_ROOT / filename
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = _without_docstring(module)

    assert len(body) == 2, (
        f"{filename} must contain only the shared-core import and __main__ call; "
        f"found {len(body)} executable top-level statements"
    )

    import_node, main_node = body
    assert isinstance(import_node, ast.ImportFrom)
    assert import_node.module == "train"
    assert import_node.level == 0
    assert [(item.name, item.asname) for item in import_node.names] == [
        ("TrainingVariant", None),
        ("run_cli", None),
    ]

    assert isinstance(main_node, ast.If)
    assert isinstance(main_node.test, ast.Compare)
    assert isinstance(main_node.test.left, ast.Name)
    assert main_node.test.left.id == "__name__"
    assert len(main_node.test.ops) == 1
    assert isinstance(main_node.test.ops[0], ast.Eq)
    assert len(main_node.test.comparators) == 1
    comparator = main_node.test.comparators[0]
    assert isinstance(comparator, ast.Constant)
    assert comparator.value == "__main__"
    assert main_node.orelse == []
    assert len(main_node.body) == 1

    expression = main_node.body[0]
    assert isinstance(expression, ast.Expr)
    call = expression.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_cli"
    assert call.keywords == []
    assert len(call.args) == 1
    variant = call.args[0]
    assert isinstance(variant, ast.Attribute)
    assert isinstance(variant.value, ast.Name)
    assert variant.value.id == "TrainingVariant"
    assert variant.attr == variant_name


def test_gene_mask_entrypoint_is_a_thin_shared_core_wrapper():
    _assert_thin_variant_wrapper(
        "train_gene_mask_imputer.py",
        "GENE_MASK_IMPUTER",
    )


def test_tma_entrypoint_is_a_thin_shared_core_wrapper():
    _assert_thin_variant_wrapper("train_tma_select.py", "TMA_SELECT")


def _dotted_name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _call_names(node: ast.AST):
    return {
        name
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        for name in [_dotted_name(item.func)]
        if name is not None
    }


def _loaded_names(node: ast.AST):
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _assigns_name(node: ast.Assign, expected: str):
    return any(
        isinstance(target, ast.Name) and target.id == expected
        for target in node.targets
    )


def test_hurdle_variants_cannot_reintroduce_zero_aware_mse():
    """Hurdle variants have one core NLL plus the imputer's panel NLL only."""

    path = REPO_ROOT / "train.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    # The shared hurdle expression branch must use the hurdle reconstruction
    # helper directly and must not call the legacy zero-aware MSE helper.
    hurdle_branches = []
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        if "hurdle_enabled" not in _loaded_names(node.test):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        calls = _call_names(body)
        if "hurdle_distribution.hurdle_reconstruction_loss_from_model" in calls:
            hurdle_branches.append((node, calls))

    assert len(hurdle_branches) == 1
    hurdle_node, hurdle_calls = hurdle_branches[0]
    assert "masked_mse" not in hurdle_calls
    assert any(
        isinstance(statement, ast.Assign)
        and _assigns_name(statement, "loss_expr_val")
        for statement in hurdle_node.body
    )

    # Panel completion used to assign zero-aware MSE to this symbol.  Any
    # trainable panel loss must now be the task-specific hurdle likelihood.
    panel_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and _assigns_name(node, "loss_panel_completion_val")
        and _call_names(node.value)
    ]
    assert panel_assignments, "the gene-mask variant must define a panel hurdle loss"
    panel_calls = set().union(*(_call_names(node.value) for node in panel_assignments))
    assert "masked_mse" not in panel_calls
    assert (
        "hurdle_distribution.masked_hurdle_truncated_gaussian_nll"
        in panel_calls
    )

    # The common total can differ only by this weighted task term (plus the
    # already-common VQ term); variants may not fork the overall loss formula.
    auxiliary_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and _assigns_name(node, "loss_auxiliary_val")
    ]
    assert len(auxiliary_assignments) == 1
    assert _loaded_names(auxiliary_assignments[0].value) == {
        "panel_completion_loss_weight",
        "loss_panel_completion_val",
        "loss_vq_val",
    }


def test_resume_metadata_and_best_output_preserve_original_b_contract():
    source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")

    # Resume and checkpoint serialization are one common A_new path; B/C do
    # not introduce stricter contracts or require an optimizer sidecar.
    assert "training_contract" not in source
    assert "controlled variants cannot partially resume" not in source
    assert "variant is not TrainingVariant.BASE" not in source

    # Only B requires hurdle because its sole task loss is hurdle NLL. TMA
    # follows the core config exactly and has no independent hurdle mandate.
    assert "if gene_mask_variant and not hurdle_enabled:" in source
    assert "gene_mask_imputer requires model.hurdle.enabled=true" in source

    # Original B metadata and strict-best schema are retained; the transient
    # panel_best output and post-hoc epoch re-ranking are not.
    assert '"task_name": imputer_task.TASK_NAME' in source
    assert '"task_description": imputer_task.TASK_DESCRIPTION' in source
    assert '"entrypoint": "train_gene_mask_imputer.py"' in source
    assert '"selection_metric": best_selection_metric_name' in source
    assert '"best_val_selection_metric"' in source
    assert '"last_val_holdout_gene_pooled_median"' in source
    assert "panel_best.json" not in source

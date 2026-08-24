#!/usr/bin/env python3
"""Golden regression tests for the Figure3.ipynb GT-only SVG selector."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import figure3_giotto_scores_and_order  # noqa: E402


def _assert_value_error(match, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as exc:
        assert match in str(exc), str(exc)
        return
    raise AssertionError(f"expected ValueError containing {match!r}")


def _assert_runtime_error(match, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except RuntimeError as exc:
        assert match in str(exc), str(exc)
        return
    raise AssertionError(f"expected RuntimeError containing {match!r}")


def _figure3_golden_fixture():
    """Return raw GT/coordinates with score ties that expose sort drift."""

    coordinates = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    raw_ground_truth = np.column_stack(
        [
            np.arange(10),
            np.array([0, 1] * 5),
            np.zeros(10),
            np.zeros(10),
            np.arange(9, -1, -1),
            np.array([0, 0, 5, 5, 0, 0, 5, 5, 0, 0]),
        ]
    ).astype(np.float64)
    return raw_ground_truth, coordinates


def test_exact_figure3_scores_log1p_pipeline_and_reverse_mergesort_ties():
    raw_ground_truth, coordinates = _figure3_golden_fixture()

    # Figure3.ipynb performs this transform in the caller, before the exact
    # float32 Giotto rank-correlation helper.
    expression = np.log1p(raw_ground_truth)
    scores, order = figure3_giotto_scores_and_order(
        expression, coordinates, k=8
    )

    expected_scores = np.array(
        [
            0.444949209690094,
            -0.632455587387085,
            0.0,
            0.0,
            0.444949209690094,
            -1.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-7)
    assert scores.dtype == np.float32

    # The notebook does an ascending stable mergesort and then reverses the
    # entire result. Equal scores therefore appear in reverse input order.
    np.testing.assert_array_equal(order, np.array([4, 0, 3, 2, 1, 5]))
    assert order.dtype == np.int64


def test_exact_figure3_selector_rejects_invalid_gt_or_coordinates():
    raw_ground_truth, coordinates = _figure3_golden_fixture()
    expression = np.log1p(raw_ground_truth)

    negative = expression.copy()
    negative[0, 0] = -0.01
    _assert_value_error(
        "raw GT",
        figure3_giotto_scores_and_order,
        negative,
        coordinates,
        k=8,
    )

    nonfinite_expression = expression.copy()
    nonfinite_expression[0, 0] = np.nan
    _assert_value_error(
        "finite",
        figure3_giotto_scores_and_order,
        nonfinite_expression,
        coordinates,
        k=8,
    )

    nonfinite_coordinates = coordinates.copy()
    nonfinite_coordinates[0, 0] = np.inf
    _assert_value_error(
        "finite",
        figure3_giotto_scores_and_order,
        expression,
        nonfinite_coordinates,
        k=8,
    )


def test_exact_figure3_selector_does_not_clamp_knn_k():
    raw_ground_truth, coordinates = _figure3_golden_fixture()
    expression = np.log1p(raw_ground_truth)

    _assert_value_error(
        "requires at least 9 cells",
        figure3_giotto_scores_and_order,
        expression[:8],
        coordinates[:8],
        k=8,
    )


def test_frozen_val_cohort_uses_emitted_ids_gt_order_and_shared_hash():
    # Importing here keeps the pure numeric selector tests lightweight.
    import train

    raw_base, coordinates = _figure3_golden_fixture()
    cell_ids = np.array([105, 101, 109, 102, 108, 103, 107, 104, 106, 100])
    gene_names_gt = [f"G{index:02d}" for index in range(50)]
    raw_ground_truth = np.column_stack(
        [raw_base[:, index % raw_base.shape[1]] + (index // raw_base.shape[1])
         for index in range(len(gene_names_gt))]
    )
    model_gene_names = list(reversed(gene_names_gt))

    with tempfile.TemporaryDirectory(prefix="ghist_figure3_cohort_test_") as tmp:
        base = Path(tmp)
        raw_path = base / "raw_gt.csv"
        pd.DataFrame(
            raw_ground_truth,
            index=cell_ids,
            columns=gene_names_gt,
        ).to_csv(raw_path)

        # One deterministic VAL patch emits all ten cells. Its sorted patch
        # order deliberately differs from the raw-GT CSV order above.
        dataset = SimpleNamespace(
            slide_idx=3,
            all_intersect=cell_ids.tolist(),
            coords_starts=[(0, 0)],
            nuclei=np.sort(cell_ids).reshape(2, 5),
            hsize=2,
            wsize=5,
            max_cells_per_patch=10,
        )
        dataloader = SimpleNamespace(dataset=dataset)
        source = SimpleNamespace(slide_idx=3, fp_expr=str(raw_path))
        coordinate_map = {
            int(cell_id): (float(coordinates[row, 1]), float(coordinates[row, 0]))
            for row, cell_id in enumerate(cell_ids)
        }

        cohorts, audit = train._build_fixed_figure3_svg_cohort(
            dataloader,
            [source],
            model_gene_names,
            {3: coordinate_map},
            k_neighbors=8,
        )
        frozen = cohorts[3]
        np.testing.assert_array_equal(frozen["cell_ids"], cell_ids)
        assert frozen["gene_names_gt_order"] == gene_names_gt
        np.testing.assert_array_equal(
            frozen["model_gene_indices_gt_order"],
            np.array([model_gene_names.index(gene) for gene in gene_names_gt]),
        )
        direct_scores, direct_order = figure3_giotto_scores_and_order(
            np.log1p(raw_ground_truth), coordinates, k=8
        )
        np.testing.assert_array_equal(frozen["giotto_scores"], direct_scores)
        np.testing.assert_array_equal(
            frozen["giotto_order_gt_positions"], direct_order
        )

        _, repeated_audit = train._build_fixed_figure3_svg_cohort(
            dataloader,
            [source],
            model_gene_names,
            {3: coordinate_map},
            k_neighbors=8,
        )
        assert (
            audit["combined_frozen_sha256"]
            == repeated_audit["combined_frozen_sha256"]
        )

        previous_manifest = os.environ.get("GHIST_SVG_COHORT_MANIFEST")
        shared_manifest = base / "shared" / "fixed_gt_svg_cohort.json"
        os.environ["GHIST_SVG_COHORT_MANIFEST"] = str(shared_manifest)
        try:
            train._lock_svg_cohort_manifest(audit, str(base / "arm_full"))
            train._lock_svg_cohort_manifest(audit, str(base / "arm_repeat"))
            changed = dict(audit)
            changed["combined_frozen_sha256"] = "0" * 64
            _assert_runtime_error(
                "Cross-arm fixed SVG validation cohort mismatch",
                train._lock_svg_cohort_manifest,
                changed,
                str(base / "arm_mismatch"),
            )
        finally:
            if previous_manifest is None:
                os.environ.pop("GHIST_SVG_COHORT_MANIFEST", None)
            else:
                os.environ["GHIST_SVG_COHORT_MANIFEST"] = previous_manifest


if __name__ == "__main__":
    test_exact_figure3_scores_log1p_pipeline_and_reverse_mergesort_ties()
    test_exact_figure3_selector_rejects_invalid_gt_or_coordinates()
    test_exact_figure3_selector_does_not_clamp_knn_k()
    test_frozen_val_cohort_uses_emitted_ids_gt_order_and_shared_hash()
    print("Figure3 SVG selector tests passed")

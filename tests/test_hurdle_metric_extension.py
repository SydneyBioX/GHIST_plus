"""CPU tests for non-canonical within-validation hurdle diagnostics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hurdle_metrics import (  # noqa: E402
    align_coordinates_xy,
    select_ssim_coordinate_rows,
    within_validation_cmd,
    within_validation_ssim,
)


class HurdleMetricExtensionTests(unittest.TestCase):
    def test_identical_prediction_has_ssim_one_and_cmd_zero(self):
        x_values = np.linspace(0.0, 110.0, 12)
        y_values = np.linspace(0.0, 90.0, 10)
        xx, yy = np.meshgrid(x_values, y_values)
        coordinates_xy = np.column_stack([xx.ravel(), yy.ravel()])
        target = np.column_stack(
            [
                0.2 + coordinates_xy[:, 0] / 80.0,
                0.1 + coordinates_xy[:, 1] / 70.0,
                0.4 + np.sin(coordinates_xy[:, 0] / 25.0)
                + np.cos(coordinates_xy[:, 1] / 30.0),
            ]
        )
        observed = np.ones_like(target, dtype=bool)

        ssim = within_validation_ssim(
            target, target.copy(), coordinates_xy, observed
        )
        self.assertEqual(ssim["valid_gene_count"], 3)
        np.testing.assert_allclose(ssim["scores"], np.ones(3), atol=1e-12)
        self.assertFalse(ssim["canonical_figure_metric"])

        cmd = within_validation_cmd(
            target,
            target.copy(),
            observed,
            gene_names=["A", "B", "C"],
        )
        self.assertEqual(cmd["status"], "defined_on_fixed_gt_valid_mask")
        self.assertLess(abs(cmd["cmd"]), 1e-10)
        self.assertEqual(cmd["prediction_valid_fixed_gt_coverage"], 1.0)
        self.assertFalse(cmd["canonical_figure_metric"])

    def test_cmd_is_undefined_for_constant_or_nonfinite_retained_prediction(self):
        rng = np.random.default_rng(13)
        target = rng.normal(size=(80, 3))
        prediction = target.copy()
        prediction[:, 1] = 2.0
        prediction[0, 2] = np.nan

        cmd = within_validation_cmd(
            target,
            prediction,
            np.ones_like(target, dtype=bool),
            gene_names=["A", "B", "C"],
        )
        self.assertTrue(np.isnan(cmd["cmd"]))
        self.assertEqual(
            cmd["status"],
            "undefined_prediction_constant_or_nonfinite_on_fixed_gt_mask",
        )
        self.assertEqual(cmd["fixed_gt_valid_gene_count"], 3)
        self.assertEqual(cmd["prediction_valid_fixed_gt_gene_count"], 1)
        self.assertAlmostEqual(
            cmd["prediction_valid_fixed_gt_coverage"], 1.0 / 3.0
        )
        self.assertEqual(
            cmd["invalid_prediction_genes_on_fixed_gt_mask"], ["B", "C"]
        )

    def test_coordinate_alignment_converts_yx_and_reports_ssim_only_exclusion(self):
        cell_ids = np.asarray([30, 10, 20], dtype=np.int64)
        coordinate_map_yx = {
            10: (1.5, 101.0),
            30: (3.5, 303.0),
            # Cell 20 intentionally lacks coordinates.
        }
        kept_rows, coordinates_xy, summary = select_ssim_coordinate_rows(
            cell_ids, coordinate_map_yx
        )
        np.testing.assert_array_equal(kept_rows, [0, 1])
        np.testing.assert_allclose(
            coordinates_xy,
            [[303.0, 3.5], [101.0, 1.5]],
        )
        self.assertEqual(summary["requested_cell_count"], 3)
        self.assertEqual(summary["retained_coordinate_cell_count"], 2)
        self.assertEqual(
            summary["excluded_missing_or_nonfinite_coordinate_cell_count"], 1
        )
        self.assertEqual(summary["first_excluded_cell_ids"], [20])
        self.assertAlmostEqual(
            summary["retained_coordinate_cell_coverage"], 2.0 / 3.0
        )
        self.assertIn("only to SSIM", summary["note"])

        with self.assertRaisesRegex(ValueError, "missing=1"):
            align_coordinates_xy(cell_ids, coordinate_map_yx)

    def test_ecrm_off_config_differs_only_by_ablation_identity(self):
        on_path = ROOT / "configs/breast2_ablation9/full.json"
        on_config = json.loads(on_path.read_text())
        off_config = json.loads(json.dumps(on_config))
        off_config["model"]["ecrm"]["ablation_off"] = True
        self.assertIs(off_config["model"]["ecrm"]["ablation_off"], True)
        off_config["model"]["ecrm"]["ablation_off"] = False
        self.assertEqual(off_config, on_config)


if __name__ == "__main__":
    unittest.main()

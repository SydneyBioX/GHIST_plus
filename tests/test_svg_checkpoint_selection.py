"""Focused tests for the exact Figure3 SVG rank and FULL epoch selector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint_selection import select_svg_joint_rank_checkpoint  # noqa: E402
from utils.hurdle_evaluation import fixed_gt_svg_validation_metrics  # noqa: E402


def _record(epoch, rank_pattern, *, checkpoint=True):
    keys = (
        "val_svg20_pcc_median",
        "val_svg20_ssim_median",
        "val_svg20_cmd",
        "val_svg50_pcc_median",
        "val_svg50_ssim_median",
        "val_svg50_cmd",
    )
    row = {
        "epoch": epoch,
        "checkpoint": f"epoch_{epoch}_model.pth" if checkpoint else None,
        "val_svg20_full_k_of_k": True,
        "val_svg50_full_k_of_k": True,
    }
    row.update(dict(zip(keys, rank_pattern)))
    return row


class SvgValidationMetricTests(unittest.TestCase):
    def test_strict_topk_metrics_reorder_actual_ids_to_frozen_order(self):
        cell_ids = np.arange(1, 31, dtype=np.int64)
        target = np.column_stack(
            [
                np.linspace(0.1, 2.0, 30),
                np.linspace(0.2, 3.0, 30) ** 2,
                np.sin(np.linspace(0.0, 2.0, 30)) + 2.0,
                np.cos(np.linspace(0.0, 2.0, 30)) + 2.0,
            ]
        )
        permutation = np.arange(29, -1, -1)
        diagnostic = [{
            "slide_id": 3,
            "cell_ids": cell_ids[permutation],
            "prediction": target[permutation].copy(),
            "target": target[permutation].copy(),
            "observed": np.ones_like(target[permutation], dtype=bool),
        }]
        frozen = {3: {
            "cell_ids": cell_ids,
            "model_gene_indices_gt_order": np.arange(4),
            "target_log1p_gt_order": target,
            "coordinates_xy": np.column_stack(
                [np.linspace(0, 100, 30), np.linspace(0, 60, 30) ** 1.1]
            ),
            "giotto_order_gt_positions": np.asarray([2, 0, 1, 3]),
            "gene_names_gt_order": ["G0", "G1", "G2", "G3"],
            "frozen_sha256": "abc",
        }}
        result = fixed_gt_svg_validation_metrics(
            diagnostic,
            frozen,
            svg_topk=(2, 3),
        )
        for k_value in (2, 3):
            values = result[f"top{k_value}"]
            self.assertTrue(values["full_k_of_k"])
            self.assertAlmostEqual(values["pcc_median"], 1.0)
            self.assertAlmostEqual(values["pcc_max"], 1.0)
            self.assertAlmostEqual(values["pcc_min"], 1.0)
            self.assertAlmostEqual(values["ssim_median"], 1.0)
            self.assertAlmostEqual(values["ssim_max"], 1.0)
            self.assertAlmostEqual(values["ssim_min"], 1.0)
            self.assertLess(abs(values["cmd"]), 1e-10)
            self.assertLess(abs(values["cmd_median"]), 1e-10)
            self.assertLess(abs(values["cmd_max"]), 1e-10)
            self.assertLess(abs(values["cmd_min"]), 1e-10)


class SvgCheckpointSelectorTests(unittest.TestCase):
    def test_rank_sum_tie_prefers_lowest_worst_rank(self):
        # The first three metrics order A>B>C; the last three C>B>A.
        # A/B/C all have rank sum 12, but B's worst rank is 2 rather than 3.
        a = _record(1, [3, 3, 1, 1, 1, 3])
        b = _record(2, [2, 2, 2, 2, 2, 2])
        c = _record(3, [1, 1, 3, 3, 3, 1])
        selected = select_svg_joint_rank_checkpoint([a, b, c])
        self.assertEqual(selected["best_epoch"], 2)
        self.assertEqual(selected["eligible_epoch_count"], 3)
        self.assertIsNone(selected["fallback"])

    def test_exact_metric_tie_prefers_earliest_epoch(self):
        first = _record(4, [0.7, 0.6, 0.2, 0.8, 0.7, 0.1])
        second = _record(9, [0.7, 0.6, 0.2, 0.8, 0.7, 0.1])
        selected = select_svg_joint_rank_checkpoint([second, first])
        self.assertEqual(selected["best_epoch"], 4)

    def test_no_eligible_epoch_has_no_fallback(self):
        record = _record(1, [0.7, 0.6, 0.2, 0.8, 0.7, 0.1])
        record["val_svg50_full_k_of_k"] = False
        selected = select_svg_joint_rank_checkpoint([record])
        self.assertIsNone(selected["best_epoch"])
        self.assertIsNone(selected["best_checkpoint"])
        self.assertIsNone(selected["fallback"])


if __name__ == "__main__":
    unittest.main()

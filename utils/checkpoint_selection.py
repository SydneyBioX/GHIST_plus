"""Deterministic validation-only checkpoint selection for SVG metrics."""

from __future__ import annotations

import math


SVG_TOPK = (20, 50)


def _metric_specs(topk=SVG_TOPK):
    specs = []
    for k_value in topk:
        prefix = f"val_svg{int(k_value)}"
        specs.extend(
            [
                (f"{prefix}_pcc_median", "higher"),
                (f"{prefix}_ssim_median", "higher"),
                (f"{prefix}_cmd", "lower"),
            ]
        )
    return tuple(specs)


def _average_ranks(values, *, higher_is_better):
    """Return one-based average ranks, including exact-value ties."""

    ordered = sorted(
        range(len(values)),
        key=lambda index: values[index],
        reverse=bool(higher_is_better),
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        tied_value = values[ordered[start]]
        while stop < len(ordered) and values[ordered[stop]] == tied_value:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[ordered[position]] = float(average_rank)
        start = stop
    return ranks


def select_svg_joint_rank_checkpoint(records, *, topk=SVG_TOPK):
    """Select FULL's epoch from six validation metrics with no fallback.

    An epoch is eligible only when a saved checkpoint exists, both Top-K
    summaries report strict full-K/K eligibility, and all six values are
    finite.  Average ranks are summed. Exact rank-sum ties minimize the worst
    component rank, then the sum of squared ranks, then the epoch number.
    """

    specs = _metric_specs(topk)
    eligible = []
    ineligible = []
    for record in records:
        reasons = []
        if not record.get("checkpoint"):
            reasons.append("checkpoint_not_saved")
        for k_value in topk:
            if not bool(record.get(f"val_svg{int(k_value)}_full_k_of_k", False)):
                reasons.append(f"svg{int(k_value)}_not_full_k_of_k")
        for key, _direction in specs:
            value = record.get(key)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"{key}_not_finite")
        if reasons:
            ineligible.append(
                {"epoch": int(record.get("epoch", -1)), "reasons": reasons}
            )
        else:
            eligible.append(record)

    ranked = []
    if eligible:
        ranks_by_metric = {}
        for key, direction in specs:
            values = [float(record[key]) for record in eligible]
            ranks_by_metric[key] = _average_ranks(
                values, higher_is_better=(direction == "higher")
            )
        for row_index, record in enumerate(eligible):
            component_ranks = {
                key: float(ranks_by_metric[key][row_index]) for key, _ in specs
            }
            rank_values = list(component_ranks.values())
            ranked.append(
                {
                    "epoch": int(record["epoch"]),
                    "checkpoint": record["checkpoint"],
                    "metric_ranks": component_ranks,
                    "rank_sum": float(sum(rank_values)),
                    "worst_rank": float(max(rank_values)),
                    "squared_rank_sum": float(sum(value * value for value in rank_values)),
                    "record": record,
                }
            )
        ranked.sort(
            key=lambda row: (
                row["rank_sum"],
                row["worst_rank"],
                row["squared_rank_sum"],
                row["epoch"],
            )
        )

    selected = ranked[0] if ranked else None
    return {
        "selection_metric": "fixed_gt_svg20_svg50_joint_average_rank",
        "selection_scope": "validation_only",
        "metric_directions": {key: direction for key, direction in specs},
        "eligibility": "saved checkpoint; full K/K PCC+SSIM and defined CMD at Top20 and Top50; all six metrics finite",
        "tie_breakers": ["lowest_worst_rank", "lowest_squared_rank_sum", "earliest_epoch"],
        "fallback": None,
        "eligible_epoch_count": int(len(eligible)),
        "ineligible_epochs": ineligible,
        "ranked_eligible_epochs": [
            {key: value for key, value in row.items() if key != "record"}
            for row in ranked
        ],
        "best_epoch": int(selected["epoch"]) if selected is not None else None,
        "best_checkpoint": selected["checkpoint"] if selected is not None else None,
        "selected_metrics": selected["record"] if selected is not None else None,
    }

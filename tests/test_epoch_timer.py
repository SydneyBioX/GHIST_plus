#!/usr/bin/env python3
"""CPU contracts for the opt-in epoch timer (utils/timing.py)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.timing import EpochTimer  # noqa: E402


def test_disabled_timer_is_a_pass_through(caplog):
    timer = EpochTimer(0, "cpu", enabled=False)
    items = [1, 2, 3]
    with caplog.at_level(logging.INFO):
        with timer.h2d():
            pass
        assert list(timer.iterate(iter(items))) == items
    assert timer.iters == 0 and not caplog.records


def test_enabled_timer_splits_wait_h2d_and_other(caplog):
    timer = EpochTimer(2, "cpu", enabled=True)

    def slow_source():
        for item in range(3):
            time.sleep(0.02)  # counted as data_wait
            yield item

    seen = []
    with caplog.at_level(logging.INFO):
        for item in timer.iterate(slow_source()):
            with timer.h2d():
                time.sleep(0.01)  # counted as h2d
            time.sleep(0.005)  # "other"
            seen.append(item)
    assert seen == [0, 1, 2]
    assert timer.iters == 3
    assert timer.wait >= 0.06 and timer.h2d_s >= 0.03
    (record,) = [r for r in caplog.records if r.getMessage().startswith("TIMING epoch=3 ")]
    assert "iters=3" in record.getMessage() and "data_wait_s=" in record.getMessage()


def test_timing_line_passes_the_concise_log_filter():
    import train

    flt = train._ConciseTrainingLogFilter()
    record = logging.LogRecord("x", logging.INFO, "f", 1, "TIMING epoch=1 iters=2", None, None)
    assert flt.filter(record)

"""Regression checks for the locked concise full-training console contract."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("ghist_train", ROOT / "train.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class ConciseTrainingLoggingTest(unittest.TestCase):
    def test_only_locked_info_lines_and_errors_are_visible(self):
        filt = train._ConciseTrainingLogFilter()

        def accepted(level, message):
            record = logging.LogRecord("test", level, __file__, 1, message, (), None)
            return bool(filt.filter(record))

        for message in (
            "Using visible GPU(s): 0",
            "Cell types: ['B', 'T']",
            "Num cell types: 2",
            "Train cells: 70180",
            "Val cells: 13621",
            "280 genes (union)",
            "VAL epoch=1 | SVG20 PCCmed=0.5000",
        ):
            self.assertTrue(accepted(logging.INFO, message), message)
        for message in (
            "Reproducibility seed: 1",
            "Preparing data",
            "Checkpoint epoch=1",
            "Epoch[1/30], Loss:1.2345",
            "Training finished",
        ):
            self.assertFalse(accepted(logging.INFO, message), message)
        self.assertFalse(accepted(logging.WARNING, "routine warning"))
        self.assertTrue(accepted(logging.ERROR, "fatal error"))

    def test_explicit_error_handler_is_fatal(self):
        handler = train._FatalLogErrorHandler()
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, "boom", (), None)
        with self.assertRaisesRegex(RuntimeError, "fatal logged error: boom"):
            handler.emit(record)


if __name__ == "__main__":
    unittest.main()

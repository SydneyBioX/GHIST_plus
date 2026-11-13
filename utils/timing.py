"""Opt-in epoch timing for the training loop.

Set ``GHIST_TIMING=1`` to split each training epoch's wall-clock into

* ``data_wait``  - main process blocked waiting for the next DataLoader batch,
* ``h2d``        - host-to-device copy of the batch (CUDA-synchronised), and
* ``other``      - everything else (forward/backward/optimiser, metrics).

and log one ``TIMING`` line per epoch. Off by default: when disabled the
wrapper is a plain pass-through and ``h2d()`` is a no-op context.
"""

import contextlib
import logging
import os
import time

import torch

ENABLED = os.environ.get("GHIST_TIMING", "0") == "1"


class EpochTimer:
    def __init__(self, epoch, device, enabled=ENABLED):
        self.epoch = epoch
        self.enabled = enabled
        self.cuda = enabled and torch.device(device).type == "cuda"
        self.wait = 0.0
        self.h2d_s = 0.0
        self.iters = 0

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def h2d(self):
        if not self.enabled:
            yield
            return
        self._sync()
        t = time.perf_counter()
        yield
        self._sync()
        self.h2d_s += time.perf_counter() - t

    def iterate(self, iterable):
        if not self.enabled:
            yield from iterable
            return
        start = time.perf_counter()
        it = iter(iterable)
        while True:
            t = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                break
            self.wait += time.perf_counter() - t
            self.iters += 1
            yield item
        self._sync()
        total = time.perf_counter() - start
        logging.info(
            "TIMING epoch=%d iters=%d train_s=%.1f it_per_s=%.3f data_wait_s=%.1f "
            "(%.1f%%) h2d_s=%.1f (%.1f%%) other_s=%.1f (%.1f%%)",
            self.epoch + 1,
            self.iters,
            total,
            self.iters / total if total > 0 else 0.0,
            self.wait,
            100 * self.wait / total if total > 0 else 0.0,
            self.h2d_s,
            100 * self.h2d_s / total if total > 0 else 0.0,
            total - self.wait - self.h2d_s,
            100 * (total - self.wait - self.h2d_s) / total if total > 0 else 0.0,
        )

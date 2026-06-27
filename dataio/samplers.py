"""Batch samplers used by training dataloaders."""

import numpy as np


class SpatialSlideBatchSampler:
    """Yield slide-pure, spatially local patch batches with optional epoch shuffling."""

    def __init__(self, datasets, batch_size, shuffle_batches=True, seed=0):
        self.batch_size = int(batch_size)
        self.shuffle_batches = bool(shuffle_batches)
        self.seed = int(seed)
        self._epoch_index = 0
        self.batches = _ordered_slide_batches(datasets, self.batch_size)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = np.arange(len(self.batches), dtype=np.int64)
        if self.shuffle_batches and len(order) > 1:
            rng = np.random.default_rng(self.seed + self._epoch_index)
            rng.shuffle(order)
        self._epoch_index += 1
        for idx in order:
            yield self.batches[int(idx)]


def _ordered_slide_batches(datasets, batch_size):
    offsets = []
    lengths = []
    acc = 0
    for ds in datasets:
        offsets.append(acc)
        l = len(ds)
        lengths.append(l)
        acc += l

    batches = []
    for off, l in zip(offsets, lengths):
        idxs = list(range(off, off + l))
        for i in range(0, l, batch_size):
            batches.append(idxs[i : i + batch_size])
    return batches


def slide_batch_sampler(datasets, batch_size, shuffle_batches=True):
    return SpatialSlideBatchSampler(
        datasets,
        batch_size,
        shuffle_batches=shuffle_batches,
    )

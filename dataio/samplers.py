"""Batch samplers used by training dataloaders."""

import logging

import numpy as np


class _InterleavedSlideBatchSampler:
    """
    Keep each batch slide-pure while interleaving slide batches across the epoch.

    This preserves the per-slide avgexp assumption used later in training while
    preventing AdamW from seeing long same-slide blocks.
    """

    def __init__(
        self,
        datasets,
        batch_size: int,
        seed: int = 0,
        shuffle_within_slide: bool = True,
        shuffle_slide_order: bool = True,
        slide_weights=None,
        weight_cap: float = 0.0,
    ):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.shuffle_slide_order = bool(shuffle_slide_order)
        self.slide_weights = list(slide_weights) if slide_weights is not None else None
        self.weight_cap = float(weight_cap)
        self.offsets = []
        self.lengths = []
        acc = 0
        for ds in datasets:
            self.offsets.append(acc)
            length = len(ds)
            self.lengths.append(length)
            acc += length
        self.total_batches = sum(
            (length + self.batch_size - 1) // self.batch_size for length in self.lengths
        )
        self._epoch_index = 0

    def __len__(self):
        return self.total_batches

    def _sample_slide_indices(self, rng, length: int, weights):
        if length <= 0:
            return np.zeros((0,), dtype=np.int64)
        if weights is None:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        try:
            probs = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception:
            probs = None
        if probs is None or probs.shape[0] != length:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, a_min=0.0, a_max=None)
        if self.weight_cap > 0:
            probs = np.minimum(probs, self.weight_cap)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            idxs = np.arange(length, dtype=np.int64)
            if self.shuffle_within_slide and length > 1:
                rng.shuffle(idxs)
            return idxs

        probs = probs / total
        return rng.choice(length, size=length, replace=True, p=probs).astype(np.int64)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch_index)
        self._epoch_index += 1

        slide_batches = []
        for slide_idx, (offset, length) in enumerate(zip(self.offsets, self.lengths)):
            weights = None
            if self.slide_weights is not None and slide_idx < len(self.slide_weights):
                weights = self.slide_weights[slide_idx]
            local_idxs = self._sample_slide_indices(rng, length, weights)
            idxs = local_idxs + offset
            batches = [
                idxs[i : i + self.batch_size].tolist()
                for i in range(0, length, self.batch_size)
            ]
            slide_batches.append(batches)

        positions = [0] * len(slide_batches)
        active = [i for i, batches in enumerate(slide_batches) if batches]
        while active:
            round_order = list(active)
            if self.shuffle_slide_order and len(round_order) > 1:
                rng.shuffle(round_order)
            next_active = []
            for slide_idx in round_order:
                pos = positions[slide_idx]
                batches = slide_batches[slide_idx]
                if pos >= len(batches):
                    continue
                yield batches[pos]
                positions[slide_idx] += 1
                if positions[slide_idx] < len(batches):
                    next_active.append(slide_idx)
            active = next_active


def slide_batch_sampler(datasets, batch_size, training_cfg, interleave=False):
    sampler_seed = int(getattr(training_cfg, "batch_sampler_seed", 0))
    shuffle_within_slide = bool(
        getattr(training_cfg, "shuffle_within_slide_batches", True)
    )
    weighted_interleave = bool(
        getattr(training_cfg, "weighted_interleave_slide_batches", True)
    )
    weight_cap = float(getattr(training_cfg, "sampler_weight_cap", 3.0))
    offsets = []
    lengths = []
    acc = 0
    for ds in datasets:
        offsets.append(acc)
        l = len(ds)
        lengths.append(l)
        acc += l
    if interleave:
        logging.info(
            "Using interleaved slide batch sampler: seed=%d shuffle_within_slide=%s weighted=%s",
            sampler_seed,
            str(shuffle_within_slide),
            str(weighted_interleave),
        )
        slide_weights = None
        if weighted_interleave:
            slide_weights = []
            for ds in datasets:
                slide_weights.append(getattr(ds, "patch_weights", None))
        return _InterleavedSlideBatchSampler(
            datasets,
            batch_size,
            seed=sampler_seed,
            shuffle_within_slide=shuffle_within_slide,
            shuffle_slide_order=True,
            slide_weights=slide_weights,
            weight_cap=weight_cap,
        )
    batches = []
    for off, l in zip(offsets, lengths):
        idxs = list(range(off, off + l))
        for i in range(0, l, batch_size):
            batches.append(idxs[i : i + batch_size])
    return batches

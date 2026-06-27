"""Batch samplers used by training dataloaders."""


def slide_batch_sampler(datasets, batch_size):
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

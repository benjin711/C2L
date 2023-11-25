from typing import List

import torch

from c2l.datasets.c2l_dataclasses import C2LDataSample, C2LDataBatch


def collate_c2l_data_samples(samples: List[C2LDataSample]) -> C2LDataBatch:
    """
    Collate a list of C2LDataSamples into a C2LDataBatch.
    Args:
        samples: A list of C2LDataSamples.
    Returns:
        A C2LDataBatch.
    """
    pcl = [torch.tensor(sample.pcl) for sample in samples]
    img = torch.stack([torch.tensor(sample.img) for sample in samples])
    K = torch.stack([torch.tensor(sample.K) for sample in samples])
    T = torch.stack([torch.tensor(sample.T) for sample in samples])
    metadata = {key: [sample.metadata[key] for sample in samples]
                for key in samples[0].metadata.keys()}

    return C2LDataBatch(pcl, img, K, T, metadata)

from typing import Protocol
import numpy as np
from torch.utils.data import Dataset
from c2l.datasets.c2l_dataclasses import C2LDataSample
from c2l.utils.augmentor import Augmentor
from c2l.utils.transformation_sampler import TransformationSampler


class C2LDataset(Protocol):

    def __len__(self) -> int:
        ...

    def get_sample(self, idx) -> C2LDataSample:
        ...


class C2LDatasetWrapper(Dataset):

    def __init__(
        self,
        dataset: C2LDataset,
        augmentator: Augmentor,
        transformation_sampler: TransformationSampler
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.augmentator = augmentator
        self.transformation_sampler = transformation_sampler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> C2LDataSample:
        sample = self.dataset.get_sample(index)

        # Transform pcl from velo to cam frame
        pcl, intensity = sample.pcl[:, :3], sample.pcl[:, 3]
        pcl = np.hstack((pcl, np.ones((pcl.shape[0], 1))))
        pcl = np.dot(sample.T, pcl.T).T[:, :3]
        pcl = np.hstack((pcl, intensity[:, None]))

        # Image, pcl, K tuple augmentations and format standardization
        sample.img, pcl, sample.K = self.augmentator(
            image=sample.img,
            pcl=pcl,
            K=sample.K
        )

        # Transform pcl from cam to prior frame
        pcl, intensity = sample.pcl[:, :3], sample.pcl[:, 3]
        pcl = np.hstack((pcl, np.ones((pcl.shape[0], 1))))
        T = self.transformation_sampler().numpy()
        pcl = np.dot(T, pcl.T).T[:, :3]
        sample.pcl = np.hstack((pcl, intensity[:, None]))

        # Update velo to cam to prior to cam transformation
        sample.T = np.linalg.inv(T)

        return sample

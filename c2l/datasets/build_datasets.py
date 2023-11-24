from enum import Enum
from typing import Dict
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyparsing import Any

from torch.utils.data import ConcatDataset

from c2l.datasets.c2l_dataset_wrapper import C2LDatasetWrapper


class DatasetTypes(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def build_datasets(cfg: DictConfig) -> Dict[str, Any]:
    """
    Build datasets from config

    Args:
        cfg (DictConfig): Config

    Returns:
        Dict[str, Any]: Dict of datasets
    """

    datasets = {}
    dataset_types = [item.value for item in DatasetTypes]

    for dataset_type, dataset_cfgs in cfg.items():
        if dataset_type not in dataset_types:
            continue

        if dataset_cfgs:
            datasets[dataset_type] = ConcatDataset([
                C2LDatasetWrapper(
                    instantiate(cfg.dataset),
                    instantiate(cfg.augmentor),
                    instantiate(cfg.transformation_sampler)
                ) for cfg in dataset_cfgs
            ])

    return datasets

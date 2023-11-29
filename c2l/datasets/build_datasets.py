import logging
from enum import Enum
from typing import Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from pyparsing import Any
from torch.utils.data import ConcatDataset

from c2l.datasets.c2l_dataset_wrapper import C2LDatasetWrapper

logger = logging.getLogger(__name__)


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
    logger.info("Start building datasets")

    datasets = {}
    dataset_types = [item.value for item in DatasetTypes]

    for dataset_type, dataset_cfgs in cfg.items():
        if dataset_type not in dataset_types:
            continue

        if dataset_cfgs:
            ds_wrappers = []

            for idx, dataset_cfg in enumerate(dataset_cfgs):
                ds_wrappers.append(C2LDatasetWrapper(
                    instantiate(dataset_cfg.dataset),
                    instantiate(dataset_cfg.augmentor),
                    instantiate(dataset_cfg.transformation_sampler)
                ))
                logger.info(f"{dataset_type.upper()} dataset {idx}:\n{ds_wrappers[-1].dataset}")

            datasets[dataset_type] = ConcatDataset(ds_wrappers)

    logger.info("Finished building datasets")
    return datasets

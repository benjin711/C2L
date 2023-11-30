import logging
from enum import Enum
from typing import Any, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from c2l.datasets.c2l_dataset_wrapper import C2LDatasetWrapper

logger = logging.getLogger(__name__)


class DatasetTypes(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def build_dataloaders(cfg: DictConfig, datasets: Dict[str, Any]) -> Dict[str, Any]:
    """Build dataloaders from config and datasets.

    Args:
        cfg (DictConfig): Config.
        datasets (Dict[str, Any]): Datasets.

    Returns:
        Dict[str, Any]: Dataloaders.
    """
    logger.info("Start building dataloaders")

    dataloaders = {}
    for dataset_type, dataset in datasets.items():
        dataloaders[dataset_type] = instantiate(
            cfg[dataset_type], _target_=DataLoader, dataset=dataset)

    logger.info("Finished building dataloaders")
    return dataloaders


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


def build_models(cfg: DictConfig) -> Dict[str, Any]:
    """Build models from config.

    Args:
        cfg (DictConfig): Config.

    Returns:
        Dict[str, Any]: Models.
    """
    logger.info("Start building models")

    models = {}
    for model_type, model_cfg in cfg.items():
        models[model_type] = instantiate(model_cfg)

    logger.info("Finished building models")
    return models

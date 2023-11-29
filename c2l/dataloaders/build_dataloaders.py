import logging
from typing import Any, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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

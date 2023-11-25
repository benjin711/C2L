

from typing import Any, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def build_dataloaders(cfg: DictConfig, datasets: Dict[str, Any]) -> Dict[str, Any]:
    """Build dataloaders from config and datasets.

    Args:
        cfg (DictConfig): Config.
        datasets (Dict[str, Any]): Datasets.

    Returns:
        Dict[str, Any]: Dataloaders.
    """
    dataloaders = {}
    for dataset_type, dataset in datasets.items():
        dataloaders[dataset_type] = instantiate(
            cfg[dataset_type], _target_=DataLoader, dataset=dataset)

    return dataloaders

import logging

import hydra
from omegaconf import DictConfig

from c2l.dataloaders.build_dataloaders import build_dataloaders
from c2l.datasets.build_datasets import DatasetTypes, build_datasets
from c2l.utils.utils import configure_logging, set_seeds

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    # Remove test dataset
    if DatasetTypes.TEST.value in cfg.datasets:
        del cfg.datasets.test

    datasets = build_datasets(cfg.datasets)
    dataloaders = build_dataloaders(cfg.dataloaders, datasets)  # pylint: disable=unused-variable


def test(cfg: DictConfig) -> None:
    # Remove training and validation datasets
    if DatasetTypes.TRAIN.value in cfg.datasets:
        del cfg.datasets.train
    if DatasetTypes.VAL.value in cfg.datasets:
        del cfg.datasets.val

    datasets = build_datasets(cfg.datasets)
    dataloaders = build_dataloaders(cfg.dataloaders, datasets)  # pylint: disable=unused-variable


@hydra.main(version_base=None, config_path="conf", config_name="main_config")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging)

    logging.info(f"Experiment name: {cfg.general.exp_name}")
    logging.info(f"Run mode: {cfg.general.run_mode}")

    set_seeds(cfg.general.seed)

    if cfg.general.run_mode == 'train':
        train(cfg)
    elif cfg.general.run_mode == 'test':
        test(cfg)
    else:
        raise ValueError(f'Unknown mode {cfg.general.run_mode}')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

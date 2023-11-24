import hydra
from omegaconf import DictConfig

from c2l.datasets.build_datasets import build_datasets, DatasetTypes
from c2l.utils.utils import set_seeds


def train(cfg: DictConfig) -> None:
    # Remove test dataset
    if DatasetTypes.TEST.value in cfg.datasets:
        del cfg.datasets.test

    datasets = build_datasets(cfg.datasets)  # pylint: disable=unused-variable


def test(cfg: DictConfig) -> None:
    # Remove training and validation datasets
    if DatasetTypes.TRAIN.value in cfg.datasets:
        del cfg.datasets.train
    if DatasetTypes.VAL.value in cfg.datasets:
        del cfg.datasets.val

    datasets = build_datasets(cfg.datasets)  # pylint: disable=unused-variable


@hydra.main(version_base=None, config_path="conf", config_name="main_config")
def main(cfg: DictConfig) -> None:

    set_seeds(cfg.general.seed)

    if cfg.general.run_mode == 'train':
        train(cfg)
    elif cfg.general.run_mode == 'test':
        test(cfg)
    else:
        raise ValueError(f'Unknown mode {cfg.general.run_mode}')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

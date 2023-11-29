import logging
import random
import timeit

import numpy as np
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

from c2l.utils.augmentor import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def set_seeds(seed) -> None:
    """
    Set seeds for reproducibility. See 
    https://pytorch.org/docs/stable/notes/randomness.html 
    for more info.

    Args:
        seed (int): seed to set
    """
    logger.info(f"Setting seeds to {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_logging(cfg) -> None:
    """
    Hydra automatically adds a stream and a file handler to the root logger.
    This function adjust and overwrites the hydra configuration.
    Args:
        cfg (DictConfig): Logging config
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Overwrite hydra config from INFO to DEBUG

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    for handler in root.handlers:
        handler.setFormatter(formatter)
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(cfg.log_level.file)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(cfg.log_level.stream)


def revert_imagenet_normalization(img: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Revert normalization applied to ImageNet images.

    Args:
        img (torch.Tensor, BxCxHxW): Image to revert normalization

    Returns:
        torch.Tensor: Reverted image
    """
    rmean = -torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD)
    rstd = 1 / torch.tensor(IMAGENET_STD)
    return F.normalize(img, mean=rmean, std=rstd, inplace=inplace)


def profile_dataloading_bottleneck(dataloader: torch.utils.data.DataLoader) -> None:
    """
    Profile dataloading time.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to profile
    """
    def _profile_dataloading():
        # pylint: disable=unused-variable
        for batch in tqdm(dataloader, desc="Profiling dataloading"):
            pass

    t = timeit.timeit(_profile_dataloading, number=1)
    logger.info("Finished profiling dataloading")

    num_batches = len(dataloader)
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Total time: {t:.2f}s")
    logger.info(f"Time per batch: {t / num_batches:.2f}s")

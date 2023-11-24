
import random
import numpy as np
import torch
from torchvision.transforms import functional as F
from c2l.utils.augmentor import IMAGENET_STD, IMAGENET_MEAN


def set_seeds(seed) -> None:
    """
    Set seeds for reproducibility. See 
    https://pytorch.org/docs/stable/notes/randomness.html 
    for more info.

    Args:
        seed (int): seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


import random
import timeit
from tqdm import tqdm
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
    print("Finished profiling dataloading")

    num_batches = len(dataloader)
    print(f"Number of batches: {num_batches}")
    print(f"Total time: {t:.2f}s")
    print(f"Time per batch: {t / num_batches:.2f}s")


import random
import numpy as np
import torch


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

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, List, Any
import numpy as np
import torch


@dataclass
class C2LDataSample:
    pcl: Union[Path, np.ndarray]
    img: Union[Path, np.ndarray]
    K: np.ndarray
    T: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        # Enforce C=3, H, W for the image
        if isinstance(self.img, np.ndarray):
            assert len(self.img.shape) == 3, \
                f"Image must be C=3, H, W, but got {self.img.shape}"
            assert self.img.shape[0] == 3, \
                f"Image must be C=3, H, W, but got {self.img.shape}"


@dataclass
class C2LDataBatch:
    pcl: torch.Tensor
    img: torch.Tensor
    K: torch.Tensor
    T: torch.Tensor
    metadata: Dict[str, List[Any]]

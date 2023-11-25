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
    pcl: List[torch.Tensor]
    img: torch.Tensor
    K: torch.Tensor
    T: torch.Tensor
    metadata: Dict[str, List[Any]]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.pcl = [pcl.pin_memory() for pcl in self.pcl]
        self.img = self.img.pin_memory()
        self.K = self.K.pin_memory()
        self.T = self.T.pin_memory()
        return self

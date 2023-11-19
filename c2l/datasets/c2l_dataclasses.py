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


@dataclass
class C2LDataBatch:
    pcl: torch.Tensor
    img: torch.Tensor
    K: torch.Tensor
    T: torch.Tensor
    metadata: Dict[str, List[Any]]

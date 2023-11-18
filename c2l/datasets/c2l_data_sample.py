from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np


@dataclass
class C2LDataSample:
    pcl: Union[Path, np.ndarray]
    img: Union[Path, np.ndarray]
    K: np.ndarray
    T: np.ndarray
    metadata: dict

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Dataset:
    texts_train: List[str]
    texts_test: List[str]
    targets_train: np.ndarray
    targets_test: np.ndarray

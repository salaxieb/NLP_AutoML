from abc import abstractmethod

import numpy as np

from nlp_automl.base import Base


class MlMethod(Base):
    """Base class for string vrctorizers."""

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Should fit model."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Should predict on batch."""

    @abstractmethod
    def predict_one(self, features: np.ndarray) -> np.ndarray:
        """Should predict on one example."""

from abc import abstractmethod
from typing import List

import numpy as np

from nlp_automl.base import Base


class Vectorizer(Base):
    """Base class for string vrctorizers."""

    @abstractmethod
    def fit(self, messages: List[str]) -> None:
        """It should do exactly how its named."""

    @abstractmethod
    def transform_one(self, message: str) -> np.ndarray:
        """Should vectorize message."""

    @abstractmethod
    def transform(self, messages: List[str]) -> np.ndarray:
        """Should vectorize list of messages."""

    def fit_transform(self, messages: List[str]) -> np.ndarray:
        """Should fit self and return transformed."""
        self.fit(messages)
        return self.transform(messages)

from abc import abstractmethod
from typing import List

from nlp_automl.base import Base


class Preprocessor(Base):
    """Base class for string preprocessing."""

    @abstractmethod
    def preprocess_one(self, message: str) -> str:
        """Should preprocess message according to config."""

    @abstractmethod
    def preprocess(self, messages: List[str]) -> List[str]:
        """Should preprocess message according to config."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Base(ABC):
    """Base class for string vrctorizers."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        self.trial = trial
        self.hyperparams = hyperparams

    @abstractmethod
    def default_config(self, hyperparams: Optional[Dict[str, Any]]) -> Any:
        """Method giving default method config."""

    @abstractmethod
    def optuna_config(self, trial) -> Any:
        """Method giving random method config."""

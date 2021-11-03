import re
from typing import Any, Dict, List, Optional

from nlp_automl.preprocess.base import Preprocessor


class Basic(Preprocessor):
    """Makes tokens."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.re_w, self.re_yo = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def default_config(self, hyperparams: Optional[Dict[str, Any]] = None) -> Any:
        return re.compile(r"[A-z,А-я]+"), re.compile(r"ё")

    def optuna_config(self, trial) -> Dict[str, Any]:
        return self.default_config()

    def preprocess_one(self, message: str) -> str:
        """Should preprocess message according to config."""
        message = message.lower()
        message = self.re_yo.sub("е", message)
        return " ".join(self.re_w.findall(message))

    def preprocess(self, messages: List[str]) -> List[str]:
        """Should preprocess message according to config."""
        return [self.preprocess_one(msg) for msg in messages]

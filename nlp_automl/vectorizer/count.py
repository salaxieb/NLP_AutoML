from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from nlp_automl.vectorizer.base import Vectorizer


class CountVec(Vectorizer):
    """Simple wrap up on count vectorizer."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(hyperparams)
        self.vect = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )
        self.fitted = False

    def _vectorizer(self, hyperparams: Dict[str, Any]) -> Any:
        return CountVectorizer(
            max_features=hyperparams["max_features"],
            min_df=hyperparams["min_df"],
        )

    def default_config(self, hyperparams: Optional[Dict[str, Any]] = None) -> Any:
        hyperparams = hyperparams or {
            "max_features": 50000,
            "min_df": 0.001,
        }
        return self._vectorizer(hyperparams)

    def optuna_config(self, trial) -> Any:
        hyperparams = {
            "max_features": trial.suggest_int("max_features", 10, 1e5),
            "min_df": trial.suggest_float("min_df", 0.0001, 0.1, log=True),
        }
        return self._vectorizer(hyperparams)

    def fit(self, messages: List[str]) -> None:
        try:
            self.vect.fit(messages)
            self.fitted = True
        except ValueError:  # appears if min_df too high
            pass

    def transform_one(self, message: str) -> np.ndarray:
        if not self.fitted:
            return np.zeros((1, 1))
        return self.vect.transform([message]).toarray()[0]

    def transform(self, messages: List[str]) -> np.ndarray:
        if not self.fitted:
            return np.zeros((len(messages), 1))
        return self.vect.transform(messages).toarray()

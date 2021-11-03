from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nlp_automl.method.base import MlMethod


class RandomForest(MlMethod):
    """Simple wrap up on Log Reg linear model."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.seed = 777
        self.rand_forest = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def _rand_forest(self, hyperparams: Dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=hyperparams["n_estimators"], random_state=self.seed
        )

    def default_config(
        self, hyperparams: Optional[Dict[str, Any]] = None
    ) -> RandomForestClassifier:
        hyperparams = {
            "n_estimators": 100,
        }
        return self._rand_forest(hyperparams)

    def optuna_config(self, trial) -> RandomForestClassifier:
        hyperparams = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                10,
                3000,
            ),
        }
        return self._rand_forest(hyperparams)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.rand_forest.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.rand_forest.predict(features)

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        return self.rand_forest.predict([features])[0]

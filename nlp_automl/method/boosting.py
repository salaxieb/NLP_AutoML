from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from nlp_automl.method.base import MlMethod


class Boosting(MlMethod):
    """Simple wrap up on Log Reg linear model."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.seed = 777
        self.boosting = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def _boosting(self, hyperparams: Dict[str, Any]) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=hyperparams["n_estimators"],
            learning_rate=hyperparams["learning_rate"],
            max_depth=hyperparams["max_depth"],
            random_state=self.seed,
        )

    def default_config(
        self, hyperparams: Optional[Dict[str, Any]] = None
    ) -> GradientBoostingClassifier:
        hyperparams = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
        }
        return self._boosting(hyperparams)

    def optuna_config(self, trial) -> GradientBoostingClassifier:
        hyperparams = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 5.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
        }
        return self._boosting(hyperparams)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.boosting.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.boosting.predict(features)

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        return self.boosting.predict([features])[0]

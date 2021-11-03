from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from nlp_automl.method.base import MlMethod


class LogReg(MlMethod):
    """Simple wrap up on Log Reg linear model."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.seed = 777
        self.logreg = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def _logreg(self, hyperparams: Dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(
            penalty="l2",  # {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            C=hyperparams["reg_strength"],  # Inverse of reg. strngth; float>0.
            fit_intercept=True,  # Specifies if a constant (a.k.a. bias or intercept)
            class_weight="balanced",  # The “balanced” to automatically adjust weights
            random_state=self.seed,  # RandomState instance, default=None
            solver=hyperparams["solver"],
            max_iter=hyperparams["max_iter"],
            multi_class="ovr",
            verbose=0,
            warm_start=hyperparams["warm_start"],
            n_jobs=1,  # Number of CPU cores used when parallelizing over classes
        )

    def default_config(
        self, hyperparams: Optional[Dict[str, Any]] = None
    ) -> LogisticRegression:
        hyperparams = {
            "reg_strength": 1.0,
            "solver": "lbfgs",
            "max_iter": 100,
            "warm_start": False,
        }
        return self._logreg(hyperparams)

    def optuna_config(self, trial) -> LogisticRegression:
        hyperparams = {
            "reg_strength": trial.suggest_float("reg_strength", 0.001, 5, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs"]),
            "max_iter": trial.suggest_int("max_iter", 10, 300),
            "warm_start": trial.suggest_categorical("warm_start", [False]),
        }
        return self._logreg(hyperparams)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.logreg.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.logreg.predict(features)

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        return self.logreg.predict([features])[0]

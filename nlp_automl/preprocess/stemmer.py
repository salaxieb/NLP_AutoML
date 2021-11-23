from typing import Any, Dict, List, Optional

from nltk.stem.snowball import SnowballStemmer

from nlp_automl.preprocess.base import Preprocessor


class Stemmer(Preprocessor):
    """Makes tokens."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.stemmer = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def default_config(
        self, hyperparams: Optional[Dict[str, Any]] = None
    ) -> SnowballStemmer:
        return SnowballStemmer("russian")

    def optuna_config(self, trial) -> SnowballStemmer:
        return self.default_config()

    def preprocess_one(self, message: str) -> str:
        """Should preprocess message according to config."""
        message = message.lower()
        return self.stemmer.stem(message)

    def preprocess(self, messages: List[str]) -> List[str]:
        """Should preprocess message according to config."""
        return [self.preprocess_one(msg) for msg in messages]

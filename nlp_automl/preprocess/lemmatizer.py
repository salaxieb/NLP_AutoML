from string import punctuation
from typing import Any, Dict, List, Optional, Set, Tuple

from nltk.corpus import stopwords
from pymystem3 import Mystem

from nlp_automl.preprocess.base import Preprocessor


class Lemmatizer(Preprocessor):
    """Makes token lemmas."""

    def __init__(
        self, trial: Optional[Any] = None, hyperparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(trial, hyperparams)
        self.mystem, self.stopwords = (
            self.optuna_config(trial) if trial else self.default_config(hyperparams)
        )

    def default_config(
        self, hyperparams: Optional[Dict[str, Any]] = None
    ) -> Tuple[Mystem, Set[str]]:
        hyperparams = hyperparams or {
            "delete_stopwords": True,
        }
        return Mystem(), set(stopwords.words("russian"))

    def optuna_config(self, trial) -> Tuple[Mystem, Set[str]]:
        hyperparams = {
            "delete_stopwords": trial.suggest_categorical(
                "delete_stopwords", [True, False]
            )
        }
        return self.default_config(hyperparams)

    def preprocess_one(self, message: str) -> str:
        message = message.lower().strip()
        tokens = [
            token
            for token in message.split()
            if token not in self.stopwords and token.strip() not in punctuation
        ]
        tokens = self.mystem.lemmatize(message.lower())
        return " ".join(tokens)

    def preprocess(self, messages: List[str]) -> List[str]:
        return [self.preprocess_one(msg) for msg in messages]

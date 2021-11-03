from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nlp_automl.vectorizer.base import Vectorizer


def flatten(vec, embedding_size: int = 300):
    vec = np.array(vec)
    if vec.shape == (0,):
        return np.zeros((4 * embedding_size + 1,))
    arr = np.append(vec.mean(axis=0), vec.max(axis=0))
    arr = np.append(arr, vec.min(axis=0))
    arr = np.append(arr, vec.std(axis=0))
    return np.append(vec.shape[0], arr)


class FastText(Vectorizer):
    """Wrap up on 2D fast text embedding to make it 1D."""

    def __init__(
        self,
        trial: Optional[Any] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(hyperparams)
        if not hyperparams:
            raise ValueError('please provide "token_embedder" in hyperparams')
        self.token_embedder: Callable = hyperparams["token_embedder"]
        self.embedding_size: int = self.token_embedder("the").shape[0]

    def default_config(self, hyperparams: Optional[Dict[str, Any]] = None) -> Any:
        pass

    def optuna_config(self, trial) -> Any:
        pass

    def fit(self, messages: List[str]) -> None:
        pass

    def transform_one(self, message: str) -> np.ndarray:
        return flatten(
            [self.token_embedder(token) for token in message.split()],
            self.embedding_size,
        )

    def transform(self, messages: List[str]) -> np.ndarray:
        return np.array([self.transform_one(message) for message in messages])

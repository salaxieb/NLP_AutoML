from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_automl.vectorizer.count import CountVec


class TfIdfVectorizer(CountVec):
    """Simple wrap up on tf-idf vectorizer."""

    def _vectorizer(self, hyperparams: Dict[str, Any]) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=hyperparams["max_features"],
            min_df=hyperparams["min_df"],
        )

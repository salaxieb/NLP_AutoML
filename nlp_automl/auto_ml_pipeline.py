from logging import getLogger
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nlp_automl.datatypes import Dataset
from nlp_automl.preprocess.base import Preprocessor
from nlp_automl.preprocess.basic import Basic
from nlp_automl.preprocess.lemmatizer import Lemmatizer
from nlp_automl.preprocess.stemmer import Stemmer

from nlp_automl.vectorizer.base import Vectorizer
from nlp_automl.vectorizer.count import CountVec
from nlp_automl.vectorizer.fast_text import FastText
from nlp_automl.vectorizer.tf_idf import TfIdfVectorizer

from nlp_automl.method.base import MlMethod
from nlp_automl.method.log_reg import LogReg
from nlp_automl.method.boosting import Boosting
from nlp_automl.method.rand_forest import RandomForest

logger = getLogger(__name__)


class AutoMLPipeline:
    """Main training pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes class with label names and search space.
        Args:
            config: Dict[str, str
            ex.: {
                'text_column': 'message',
                'target_column': 'label',
                'evaluator': Callable,
                'dataset': pd.DataFrame,
                }
        """
        config = config or {}
        self.evaluator: Callable
        self.dataset: Dataset
        self.fit_pipeline: bool

        self.preprocessors: Dict[str, Type[Preprocessor]] = {}
        if config.get("use_basic", True):
            self.preprocessors.update({"basic": Basic})
        if config.get("use_stemmer", True):
            self.preprocessors.update({"stemmer": Stemmer})
        if config.get("use_lemmatizer", True):
            self.preprocessors.update({"lemmtizer": Lemmatizer})

        self.vectorizers: Dict[str, Type[Vectorizer]] = {}
        self.fast_text_vectorizers: Dict[str, Callable] = {}
        if config.get("use_tf_idf", True):
            self.vectorizers.update({"tf_idf": TfIdfVectorizer})
        if config.get("use_count_vec", True):
            self.vectorizers.update({"count_vec": CountVec})
        if config.get("use_fasttext", False) and config.get("fast_text_vectorizer"):
            self.fast_text_vectorizers.update(config["fast_text_vectorizer"])
            self.vectorizers.update({"fast_text": FastText})

        self.models: Dict[str, Type[MlMethod]] = {}
        if config.get("use_logreg", True):
            self.models.update({"logreg": LogReg})
        if config.get("use_boosting", True):
            self.models.update({"boosting": Boosting})
        if config.get("use_rand_forest", True):
            self.models.update({"rand_forest": RandomForest})

    def find_solution(
        self, task: Dict[str, Any], timeout: int = 3600
    ) -> Tuple[Dict[str, Any], Any]:
        """Main method to search solution for Initialized automl object."""
        self.evaluator = task["evaluator"]
        self.fit_pipeline = task.get("fit_pipeline", True)
        df_train, df_test = train_test_split(
            task["dataset"],
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        if task.get("use_label_encoder", True):
            label_encoder = LabelEncoder()
            targets_train: np.ndarray = label_encoder.fit_transform(
                df_train[task["target_column"]].values
            )
            targets_test: np.ndarray = label_encoder.transform(
                df_test[task["target_column"]].values
            )
            logger.info(f"label encoder classes_ {label_encoder.classes_}")
        else:
            targets_train = df_train[task["target_column"]].values
            targets_test = df_test[task["target_column"]].values

        self.dataset = Dataset(
            texts_train=df_train[task["text_column"]].values,
            texts_test=df_test[task["text_column"]].values,
            targets_train=targets_train,
            targets_test=targets_test,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.one_optuna_run, timeout=timeout)
        return study.best_params, self.give_pipeline(study.best_params)

    def one_optuna_run(self, trial) -> float:
        """Makes one run and evaluation with optuna trials."""

        preprocessor_name = trial.suggest_categorical(
            "preprocessor", self.preprocessors.keys()
        )
        preprocessor = self.preprocessors[preprocessor_name](trial=trial)

        vectorizer_name = trial.suggest_categorical(
            "vectorizer", self.vectorizers.keys()
        )
        hyperparams = {}
        # if vectorizer fast text we should also provide token embedder in hyperparams
        if vectorizer_name == "fast_text":
            fast_text_vectorizer_name = trial.suggest_categorical(
                "fast_text_vectorizer", self.fast_text_vectorizers.keys()
            )
            hyperparams = {
                "token_embedder": self.fast_text_vectorizers[fast_text_vectorizer_name]
            }
        vectorizer = self.vectorizers[vectorizer_name](
            trial=trial, hyperparams=hyperparams
        )

        model_name = trial.suggest_categorical("model", self.models.keys())
        model = self.models[model_name](trial=trial)

        logger.debug("preprocessing")
        preprocessed_train = preprocessor.preprocess(self.dataset.texts_train)

        logger.debug("vectorizing")
        vectorizer.fit(preprocessed_train)
        vectors_train = vectorizer.transform(preprocessed_train)

        logger.debug("model fit")
        model.fit(vectors_train, self.dataset.targets_train)
        predicts_train = model.predict(vectors_train)

        score_test, score_train = self.evaluate(
            predicts_train, preprocessor, vectorizer, model
        )
        logger.info(
            f"""
            {str(preprocessor)}
            {str(vectorizer)}
            {str(model)}
            pipeline test score {score_test},
            pipeline train score {score_train}""",
        )
        return score_test

    def give_pipeline(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Preprocessor, Vectorizer, MlMethod]:
        preprocessor = self.preprocessors[hyperparams["preprocessor"]](
            hyperparams=hyperparams
        )
        if hyperparams["vectorizer"] == "fast_text":
            hyperparams.update(
                {
                    "token_embedder": self.fast_text_vectorizers[
                        hyperparams["fast_text_vectorizer"]
                    ]
                }
            )
        vectorizer = self.vectorizers[hyperparams["vectorizer"]](
            hyperparams=hyperparams
        )
        model = self.models[hyperparams["model"]](hyperparams=hyperparams)
        if self.fit_pipeline:
            preprocessed = preprocessor.preprocess(self.dataset.texts_train)
            vectorized = vectorizer.fit_transform(preprocessed)
            model.fit(vectorized, self.dataset.targets_train)
        return (preprocessor, vectorizer, model)

    def evaluate(
        self,
        predicts_train: np.ndarray,
        preprocessor: Preprocessor,
        vectorizer: Vectorizer,
        model: MlMethod,
    ) -> Tuple[float, float]:
        preprocessed_test = preprocessor.preprocess(self.dataset.texts_test)
        vectors_test = vectorizer.transform(preprocessed_test)
        predicts_test = model.predict(vectors_test)
        return (
            self.evaluator(self.dataset.targets_test, predicts_test),
            self.evaluator(self.dataset.targets_train, predicts_train),
        )

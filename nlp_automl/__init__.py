import logging
import nltk
import optuna

from nlp_automl.auto_ml_pipeline import AutoMLPipeline

nltk.download("stopwords")
logging.basicConfig(filename=".log", level=logging.INFO)

optuna.logging.set_verbosity(optuna.logging.DEBUG)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.


__all__ = [
    'AutoMLPipeline'
]
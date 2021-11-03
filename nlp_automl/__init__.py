import logging
import nltk
import optuna

nltk.download("stopwords")
logging.basicConfig(filename=".log", level=logging.INFO)

optuna.logging.set_verbosity(optuna.logging.DEBUG)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.

import fasttext.util
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path

from nlp_automl import AutoMLPipeline

fasttext.util.download_model("ru", if_exists="ignore")
ft = fasttext.load_model("cc.ru.300.bin")


dataset = pd.read_csv(Path(__file__).parent / "data/avito1k_train.csv")
dataset.dropna(subset=["description"], inplace=True)


def get_word_vector_cc_bin(token):
    return ft.get_word_vector(token)


auto_ml_config = {
    "fast_text_vectorizer": {  # token vectorizers, optional
        "cc.ru.300.bin": get_word_vector_cc_bin  # callable
    },
    # vectorizers # optional, default: True
    "use_count_vect": True,
    "use_tf_idf": True,
    "use_fasttext": True,
    # prerocessors # optional, default: True
    "use_basic": True,
    "use_lemmatizer": True,
    "use_stemmer": True,
    # models # optional, default: True
    "use_boosting": False,
    "use_logreg": True,
    "use_rand_forest": True,
}


automl = AutoMLPipeline(config=auto_ml_config)

task = {
    "text_column": "description",  # str - required
    "target_column": "user_type",  # str - required
    "dataset": dataset,  # pd.Dataframe - required
    "use_label_encoder": True,  # Optional, default: True
    "evaluator": accuracy_score,  # callable, required
    "fit_pipeline": True,  # Optional, default: True
}

best_params, pipeline = automl.find_solution(task=task, timeout=200)
preprocessor, vectorizer, model = pipeline


# usage
text = "hello, world"
preprocessed = preprocessor.preprocess_one(text)
vector = vectorizer.transform_one(text)
prediction = model.predict_one(vector)
label = automl.label_encoder.inverse_transform([prediction])[0]

print("preprocessed: ", preprocessed)
print("vector: ", vector.shape)
print("predict: ", prediction)
print("label: ", label)
print("pipeline:", pipeline)

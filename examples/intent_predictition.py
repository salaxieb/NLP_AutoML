import os
import sys


import fasttext.util
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from nlp_automl.auto_ml_pipeline import AutoMLPipeline

fasttext.util.download_model('ru', if_exists='ignore')
ft = fasttext.load_model('cc.ru.300.bin')


dataset = pd.read_csv('examples/data/avito1k_train.csv')
dataset.dropna(subset=['description'], inplace=True)


def get_word_vector_cc_bin(token):
    return ft.get_word_vector(token)


task = {
    'text_column': 'description',  # required
    'target_column': 'user_type',  # required
    'dataset': dataset,  # required
    'use_label_encoder': True,  # Optional, default: True
    'evaluator': accuracy_score,  # required
    'fit_pipeline': True, # Optional, default: True
}
auto_ml_config = {
    'fast_text_vectorizer': { # token vectorizers, optional
        'cc.ru.300.bin': get_word_vector_cc_bin  # callable
    },
    # vectorizers # optional, default: True
    'use_count_vect': True,
    'use_tf_idf': True,
    'use_fasttext': True,

    # prerocessors # optional, default: True
    'use_basic': True,
    'use_lemmatizer': True,
    'use_stemmer': True,

    # models # optional, default: True
    'use_boosting': False,
    'use_logreg': True,
    'use_rand_forest': True,
}


automl = AutoMLPipeline(config=auto_ml_config)
best_params, pipeline = automl.find_solution(task=task, timeout=200)
preprocessor, vectorizer, model = pipeline


# usage
my_text = 'Здарова, бандиты!'
preprocessed = preprocessor.preprocess_one(my_text)
vector = vectorizer.transform_one(my_text)
predict = model.predict_one(vector)

print('preprocessed: ', preprocessed)
print('vector: ', vector.shape)
print('predict: ', predict)
print('+'*100)
print('pipeline:', pipeline)

import os
import sys


import fasttext.util
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score


sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))

from nlp_automl.auto_ml_pipeline import AutoMLPipeline

fasttext.util.download_model('ru', if_exists='ignore')
ft = fasttext.load_model('cc.ru.300.bin')


dataset = pd.read_csv('examples/data/avito1k_train.csv')
dataset.dropna(subset=['description'], inplace=True)


def get_word_vector_cc_bin(token):
    return ft.get_word_vector(token)


config = {
    'text_column': 'description',
    'target_column': 'user_type',
    'dataset': dataset,
    'use_label_encoder': True,  # optional
    'evaluator': accuracy_score,

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
    'use_boosting': True,
    'use_logreg': True,
    'use_rand_forest': True,
}

automl = AutoMLPipeline(config=config)
best_params, pipeline = automl.find_solution(timeout=200)
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

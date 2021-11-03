import pytest
import pandas as pd
from sklearn.metrics import accuracy_score
from nlp_automl.auto_ml_pipeline import AutoMLPipeline
import fasttext.util
from sklearn.preprocessing import LabelEncoder


def test_something(dataset):
    # fasttext.util.download_model('ru', if_exists='ignore')
    # ft = fasttext.load_model('cc.ru.300.bin')

    # def get_word_vector_cc_bin(token):
    #     return ft.get_word_vector(token)

    target_column = 'user_type'
    text_column = 'description'

    le = LabelEncoder()
    le.fit(dataset[target_column])

    dataset['user_type'] = le.transform(dataset['user_type'].values)

    sub_dataset = dataset.sample(100, random_state=42)
    dataset = dataset.drop(sub_dataset.index)

    config = {
        'text_column': text_column,
        'target_column': target_column,
        'dataset': dataset,
        'use_label_encoder': False,  # optional
        'evaluator': accuracy_score,

        'fast_text_vectorizer': { # token vectorizers, optional
            # 'cc.ru.300.bin': get_word_vector_cc_bin  # callable
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
    best_params, pipeline = automl.find_solution(timeout=20)
    preprocessor, vectorizer, model = pipeline

    preprocessed = preprocessor.preprocess(sub_dataset[text_column])
    vectors = vectorizer.transform(preprocessed)
    predict = model.predict(vectors)
    assert 0.5 < accuracy_score(sub_dataset[target_column], predict) < 0.75

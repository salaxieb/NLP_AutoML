import pytest
import pandas as pd
from sklearn.metrics import accuracy_score
from nlp_automl import AutoMLPipeline
import fasttext.util
from sklearn.preprocessing import LabelEncoder


def test_something(dataset):
    target_column = 'user_type'
    text_column = 'description'

    le = LabelEncoder()
    le.fit(dataset[target_column])

    dataset['user_type'] = le.transform(dataset['user_type'].values)

    sub_dataset = dataset.sample(100, random_state=42)
    dataset = dataset.drop(sub_dataset.index)

    task = {
        'text_column': text_column,  # required
        'target_column': target_column,  # required
        'dataset': dataset,  # required
        'use_label_encoder': True,  # Optional, default: True
        'evaluator': accuracy_score,  # required
        'fit_pipeline': True, # Optional, default: True
    }
    auto_ml_config = {
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
    best_params, pipeline = automl.find_solution(task=task, timeout=20)
    preprocessor, vectorizer, model = pipeline
    
    preprocessed = preprocessor.preprocess(sub_dataset[text_column])
    vectors = vectorizer.transform(preprocessed)
    predict = model.predict(vectors)

    assert 0.5 < accuracy_score(sub_dataset[target_column], predict) < 0.75

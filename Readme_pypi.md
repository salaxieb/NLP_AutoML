## AutoML NLP library
use to find baseline in text to label task

[Source code](https://github.com/salaxieb/NLP_AutoML)


#### Usage
```python
from sklearn.metrics import accuracy_score

from nlp_automl.auto_ml_pipeline import AutoMLPipeline

config = {
    'text_column': 'message',
    'target_column': 'intent',
    'dataset': dataset,
    'evaluator': accuracy_score,
}
automl = AutoMLPipeline(config=config)
best_params, pipeline = automl.find_solution(timeout=200)
preprocessor, vectorizer, model = pipeline
```
More detailed usage example [./examples/intent_prediction.py](https://github.com/salaxieb/NLP_AutoML/blob/master/examples/intent_predictition.py)

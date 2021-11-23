## AutoML NLP library
use to find baseline in text to label task

[Source code](https://github.com/salaxieb/NLP_AutoML)


#### Usage
```python
from sklearn.metrics import accuracy_score

from nlp_automl import AutoMLPipeline

task = {
    'text_column': 'description',  # required
    'target_column': 'user_type',  # required
    'dataset': dataset,  # required
    'use_label_encoder': True,  # Optional, default: True
    'evaluator': accuracy_score,  # required
    'fit_pipeline': True, # Optional, default: True
}
automl = AutoMLPipeline()
best_params, pipeline = automl.find_solution(task=task, timeout=200)
preprocessor, vectorizer, model = pipeline
```
More detailed usage example [./examples/intent_prediction.py](https://github.com/salaxieb/NLP_AutoML/blob/master/examples/intent_predictition.py)

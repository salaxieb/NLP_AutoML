## Library for text -> label task baseline search
choosing **preprocesser**, **vecotorizer**, **model** with optuna

### Demo
examples/intent_prediction.py


### Requirements
```bash
poetry install --no-dev
```


### Dev commands
```bash
poetry install
make lint  # runs linter checkers
make test  # runs test in current python version
make tox  # runs tests in specified python versions
make requirements  # inport poetry requirements to requirements.txt
```
include .env

lint:
	@poetry run mypy nlp_automl
	@poetry run pylint nlp_automl
	@poetry run flake8 nlp_automl
	@poetry run black nlp_automl --check


test:
	@poetry run pytest


tox:
	@poetry run tox


requirements:
	@poetry export -f requirements.txt --output requirements.txt
	@poetry export -f requirements.txt --output requirements.dev.txt --dev


clean:
	@rm -rf .mypy_cache
	@rm -rf .tox
	@rm -rf .pytest_cache

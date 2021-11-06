lint: clean
	@poetry run black nlp_automl
	@poetry run mypy nlp_automl
	@poetry run pylint nlp_automl


test: clean
	@poetry run pytest -s


tox: requirements
	@poetry run tox


requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes
	@poetry export -f requirements.txt --output requirements.dev.txt --dev --without-hashes


clean:
	@rm -rf .mypy_cache
	@rm -rf .tox
	@rm -rf .pytest_cache
	@rm -rf dist
	@rm -rf NLP_AutoML.egg-info
	@find . -type d -name __pycache__ -prune -exec rm -rf {} \;

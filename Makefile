include .env

lint:
	@poetry run black nlp_automl
	@poetry run mypy nlp_automl
	@poetry run pylint nlp_automl


test:
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

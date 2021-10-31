include .env

lint:
	@poetry run mypy ds_template
	@poetry run pylint ds_template
	@poetry run flake8 ds_template
	@poetry run black ds_template --check


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

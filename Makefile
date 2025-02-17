PYTHON_VERSION := $(shell cat .python-version)

create-venv:
	uv venv .venv --prompt venv-$(PYTHON_VERSION)

install:
	uv sync

lock:
	uv lock

test:
	uv run pytest

setup:
	make create-venv
	make install
	uv run pre-commit install

pre-commit-update:
	uv run pre-commit autoupdate

lint-format:
	uv run ruff check --config pyproject.toml
	uv run ruff format --config pyproject.toml

interrogate:
	uv run interrogate --config pyproject.toml

run-main:
	uv run python main.py $(ARGS)

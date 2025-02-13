PYTHON_VERSION := $(shell cat .python-version)

create-venv:
	uv venv .venv --prompt venv-$(PYTHON_VERSION)

install:
	uv sync

lock:
	uv lock

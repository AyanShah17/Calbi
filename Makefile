.PHONY: install test lint format typecheck check clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff
MYPY = $(VENV)/bin/mypy

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -m "not live"

test-live:
	$(PYTEST) -m live

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

typecheck:
	$(MYPY) src tests

check: lint typecheck test

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache

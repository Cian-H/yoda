.PHONY: help install test lint format check run clean

help:
	@echo "Targets:"
	@echo "  install      Install dependencies (and dev deps)"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  format       Run formatter"
	@echo "  check        Lint + test (CI-style)"
	@echo "  run          Run the application (override per project)"
	@echo "  clean        Remove build / cache artefacts"

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint test

run:
	@echo "Override the 'run' target per project (e.g. 'uv run python -m yoda')"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: install check fix test lint format pipeline train

install:
	uv sync --group dev

# The read-only gate. There is no CI, so `main` is only as good as the last time
# someone ran this.
#
# `format --check`, not `format`. `fix` is the mutating counterpart: a gate that
# rewrites files to make itself pass cannot fail on a formatting problem, which
# is how a repository ends up with 97 lint findings and 33 green tests.
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest -q

fix:
	uv run ruff format .
	uv run ruff check . --fix

test:
	uv run pytest -v

lint:
	uv run ruff check .
	uv run mypy

format:
	uv run ruff format .

# Corpus -> labelled train/valid split. Raises if a sample lands in both.
pipeline:
	uv run python -m mon_language_detector.pipeline

train:
	uv run python -m mon_language_detector.train

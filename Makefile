.PHONY: install check fix test lint format datasets pipeline train evaluate

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

# Corpus -> datasets/ -> labelled split -> model. Run in order.
#
# `datasets` is a SELECTION, not a copy: mon_OCR's corpus is bucketed for OCR,
# where the label is the text, and four of its Mon directories are English or
# Burmese by content. `uv run datasets --explain` prints what is dropped and why.
# CORPUS_ROOT has no default: the corpus is not in this repository, and a
# published package should not assume a path on the author's machine.
CORPUS_ROOT ?=
datasets:
	@test -n "$(CORPUS_ROOT)" || { echo "make datasets needs a corpus: make datasets CORPUS_ROOT=/path/to/corpus"; echo "  layout: one directory per source, each holding .txt files"; echo "  see:    uv run datasets --explain"; exit 1; }
	uv run datasets --corpus-root $(CORPUS_ROOT)

# Corpus -> labelled train/valid split. Raises if a sample lands in both.
pipeline:
	uv run python -m mon_language_detector.pipeline

train:
	uv run python -m mon_language_detector.train

# The number the README cites. Scores the DETECTOR, not the raw model.
evaluate:
	uv run python -m mon_language_detector.evaluate

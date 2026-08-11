# Changelog

Notable changes, newest first. Dates are when the work landed, not when it was
released. Every number names what it measures.

## 0.2.0 — 2026-08-11

The shipped model is retrained, and three of the four defects fixed here were
found by auditing the character rules rather than by a failing test.

### Model

- **Retrained on a clean split.** The 0.1.0 artifact predated the leakage fix
  (`7a9e4ca`) and was trained where upsampled lines appeared on both sides, so
  its accuracy was withdrawn rather than restated. Measured on 34,988 held-out
  lines: **0.9980 where `reliable`** over 27,085 lines (77.4% of the split),
  0.9565 overall. Per class `eng` 1.0000, `mya` 0.9675, `mnw` 0.9107.
- **Corpus selected for language, not provenance.** The source corpus is
  organised for OCR, where a line's label is the text on the image. Four of its
  Mon directories are English or Burmese by content — roughly 219,000 lines.
  `datasets.py` holds the exclusions with the measurement behind each.
- **Class membership gated on script dominance.** 6.51% of Mon-labelled
  validation lines were over 85% Latin — Wikipedia reference rows like
  "Heinz, L.C. (6 March 1962)." The detector called them English and the
  evaluation scored it wrong.

### Fixed

- **Shan, Karen and Khamti characters were treated as Mon.** Of the 45
  codepoints the Mon-exclusive rule matched, **7 were Mon**; the rest were
  Eastern Pwo Karen, Shan, Khamti, Aiton and Pao Karen. A match overrides the
  model and forces `reliable=True`, so Shan text returned `mnw` at confidence
  1.0000. Three genuinely Mon-exclusive characters were missing at the same
  time. The set is now every codepoint whose Unicode name carries MON as a word,
  re-derived by a test on every run.
- **Text in another Myanmar-script language now returns `unknown`.** Correcting
  the character rule did not fix the model: fastText has three classes and no way
  to say "none of these". A character exclusive to Shan, Khamti, Aiton, Karen or
  Palaung, with no Mon-exclusive character present, is refused. Measured cost:
  0.051% of Mon lines, 0% of Burmese.
- **Mixed-script labels no longer default to Burmese.** The branch asked what the
  whole string was, so an 82%-Latin sentence answered `eng` and a Mon fragment
  fell through to `mya-eng` with nothing suggesting Burmese. The Myanmar half is
  now classified on the Myanmar characters alone.
- **The Myanmar length floor is 30, up from 20.** Every remaining error is Mon
  against Burmese, and 97.7% of them are in lines of 40 characters or fewer.
  Costs 5.8 points of coverage, cuts the error rate 5.6x. Selected on the split
  it is scored on, so that figure is optimistic by an unknown margin.

### Added

- `Detection.basis` — whether `confidence` is a model posterior or a
  hand-chosen constant. Closes audit finding H1. Appended with a default, so the
  three-field shape still unpacks.
- `uv run evaluate` — scores the detector rather than the raw classifier. The
  two differ by more than two points, and the README quotes this one.
- `uv run datasets` — the corpus selection, with `--explain`.
- Per-class precision, recall and F1 in `train.py`, for the quantized artifact as
  well as the full one. Quantization raised the aggregate from 0.9549 to 0.9561
  while dropping Burmese precision from 0.8143 to 0.7645.
- CI: lint, types and tests on 3.11 to 3.13, plus a job that installs the built
  wheel outside the repo and loads the model from it.

### Changed

- **pandas, pyarrow and tqdm moved to a `wrangle` extra.** Installing an 8 MB
  language detector pulled over 100 MB of wheels for a CLI unrelated to
  detection. `pip install mon-language-detector[wrangle]` restores them.
- `predict` classifies each character once and memoises it, instead of three
  passes with a Python predicate per character. **23,639 to 28,149 lines/s** on
  an Apple M5, identical output. Batching the fastText call was measured at 1.1x
  and left alone.
- `build_dataset` and `train_model` are keyword-only. Both took long runs of
  same-typed arguments where a transposition is silent and type-correct.
- Model size is reported in MB rather than MiB. It was computed as MiB and
  labelled MB, which put 7.72 in the README for an 8.10 MB file.

## 0.1.0

Initial release. Accuracy withdrawn — see 0.2.0.

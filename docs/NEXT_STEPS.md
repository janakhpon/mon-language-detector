# Next steps

Forward work for this repository. The findings and their closures live in
[AUDIT-2026-08-08.md](AUDIT-2026-08-08.md), which carries a status table; this
file is what is left and in what order.

**Re-verified 2026-08-11:** `make check` green — ruff format, ruff check, mypy
and **74 passed**. All eight findings from the 2026-08-08 audit are closed, and
so is the retrain that was blocking everything.

---

## 1. Closed: the retrain

Done 2026-08-11. `src/mon_language_detector/data/langid_mon_mya_eng_compressed.ftz`
is trained on the fixed pipeline, and the README states the figures again with
their denominators and the command that reproduces them.

| | measured on 34,988 held-out lines |
|---|---|
| detector accuracy | **0.9565** |
| accuracy where `reliable` | **0.9888** over 29,102 lines (83.2%) |
| per class | `eng` 1.0000 · `mya` 0.9675 · `mnw` 0.9107 |

Three things the retrain taught, all now in code:

- **A corpus directory is not a language label.** Four of the sibling project's
  Mon directories are English or Burmese by content — it buckets for OCR, where
  the label is the text. `datasets.py` holds the exclusions and their
  measurements.
- **Neither is a corpus line.** 6.51% of Mon-labelled validation lines were over
  85% Latin; the detector called them English and the evaluation scored it
  wrong. `MIN_SCRIPT_DOMINANCE` gates class membership on the same threshold the
  detector uses.
- **An aggregate hides a class.** Quantization raised Precision@1 from 0.9549 to
  0.9561 while dropping Burmese precision from 0.8143 to 0.7645. `train.py` now
  reports per class, for the quantized artifact as well as the full model.

**Still open from this section:** throughput has never been measured, on any
hardware. It stays absent from the README rather than guessed.

**Publishing to PyPI is now unblocked on the artifact** and blocked on nothing
else engineering can see. The corpus's own licence and personal-data questions
are the sibling project's `LIMITATIONS.md` §upstream, and they govern whether a
model trained on it may be distributed at all. That is a decision, not a task.

---

## 2. The one thing a retrain will not fix


**The detector answers a three-way question, and the Myanmar script is shared by
more than three languages.**

fastText has classes `eng`, `mnw` and `mya` and no way to say "none of these", so
Shan, Khamti, Aiton, Karen and Palaung must land on one of the three. Measured on
the shipped model, Shan returned `mnw` at confidence **1.0000** with
`reliable=True`. For the stated use — filtering a scraped Mon corpus, which is
exactly where those languages turn up — that is contamination marked reliable.

`2a00811` closes the part that engineering can close: text carrying a character
exclusive to one of those languages, and no Mon-exclusive character, is returned
`unknown` with `basis="other-myanmar-script"`. Measured cost on the corpora:
0.051% of 306,564 Mon lines, 0% of 9,065 Burmese lines.

**What remains open is the part characters cannot reach.** Shan written without
any of those 124 codepoints is still labelled Mon. Closing it needs one of:

| | Option | Cost |
|---|---|---|
| 2.1 | A fourth class | Shan training data nobody currently has |
| 2.2 | An `other` class trained on the Myanmar-script neighbours | The same data problem, one language at a time |
| 2.3 | A posterior-margin threshold, so a confident-but-wrong answer becomes `unknown` | Cheap, but the measured posterior on Shan was 1.0000 — a margin rule would not have caught it |

2.3 is worth measuring on real Shan text before assuming it fails; one confident
sample is not a distribution. **Nothing here is a bug to fix.** It is a scope
limit, recorded in the README under "What the detector cannot do" so a caller
reads it before trusting `reliable`.

---

## 3. Mon against Burmese, which is where the error now lives

Every miss in the 2026-08-11 evaluation is that pair: **1,367 `mnw` lines
labelled `mya` and 151 the other way**, with no `eng` confusion in either
direction. The two languages share the script, and only about half of Mon lines
carry a Mon-exclusive character for the hard signal to find.

| | Work | Why it might help |
|---|---|---|
| 3.1 | **More Burmese prose.** 46,407 unique lines against Mon's 450,986, and 34,089 of those are dictionary headwords rather than sentences | The class is both the smallest and the least prose-like. Its training side is repeated 3.6× where Mon's is repeated 1.1× |
| 3.2 | Measure whether the errors concentrate in short lines | The reliable-only accuracy is 0.9888 against 0.9565 overall, so most of them already fall outside what the detector vouches for. If the rest are short, the length floor is the lever |
| 3.3 | A Mon/Burmese-only second stage, run when the script is Myanmar | Only worth building if 3.1 and 3.2 do not close it. A second model is a second thing to keep true |

Do 3.2 before 3.1: it is an afternoon with the existing split and it says
whether more data is the answer.

---

## 4. CI

`make check` exists and is green. Nothing runs it but a person.

Copy `mon_tokenizer/.github/workflows/ci.yml`, where the actions are already
pinned to commit SHAs. This is the last item from the 2026-08-09 tooling list —
declaring the linters, fixing the 97 findings and adding `make check` all closed
in `32ed3d5`. Running them on push is what is left.

---

## What we are deliberately not doing

| Not doing | Why |
|---|---|
| **Replacing fastText** | It is the right boring choice for language identification at this size, and nothing in either audit pointed at the model family. The failures were the split, the character set and the missing out-of-domain answer — none of them the classifier |
| **Adding languages to the label set** | Mon, Burmese and English with mixed-script labels covers the ecosystem's actual need. A fourth *class* is §2.1 and is a data problem; a fourth *label* without data would be a worse lie than `unknown` |
| **Publishing to PyPI or Hugging Face** | No longer blocked on the artifact — §1 closed that. It is blocked on the corpus's licence and personal-data questions, which govern whether a model trained on it may be distributed at all. That is the maintainer's decision, not a task |
| **Restating the old 0.925** | It was measured on memorised rows. It does not become true by being cited carefully |
| **Deriving `MON_EXCLUSIVE_CODEPOINTS` at import time** | The test re-derives it from Unicode names on every run, which is the same guarantee without paying ~500 `unicodedata.name` calls at import and without the set changing silently when Python's Unicode data updates |

---

## How this file stays true

Every measurement above carries the date it was taken, and every figure in the
README comes from `make evaluate` rather than from this file.

When a section closes, move its figures into the README with their denominators,
record the closure in `AUDIT-2026-08-08.md`'s status table, and replace the
section here with what the closure taught — §1 is the worked example. A
next-steps document that only ever grows is a backlog; one that records what
each closure cost is a map.

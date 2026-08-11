# Next steps

Forward work for this repository. The findings and their closures live in
[AUDIT-2026-08-08.md](AUDIT-2026-08-08.md), which carries a status table; this
file is what is left and in what order.

**Re-verified 2026-08-11:** `make check` green — ruff format, ruff check, mypy
and **68 passed**. All eight findings from the 2026-08-08 audit are closed. Two
new defects were found on 2026-08-11 and are closed; one new limitation is open
and cannot be closed by engineering alone.

---

## 1. The retrain, which is still the only blocking item

C2 — train/valid leakage by construction — is closed in the **pipeline**
(`7a9e4ca`). It is not closed in the **artifact**.
`src/mon_language_detector/data/langid_mon_mya_eng_compressed.ftz` was built
before the fix, so the model this package ships was trained on a split where
upsampled Mon lines appeared on both sides.

The README handles this correctly: Precision@1 is *"withdrawn pending a
retrain"*, with the reason, and throughput is *"not currently measured"* rather
than repeated from an old run. **Withdrawing a number instead of re-asserting it
from a contaminated split is the right call.**

| | Work |
|---|---|
| 1.1 | Retrain on the clean split. The pipeline deduplicates, splits before upsampling, upsamples the train side only, derives code-switched samples per split, seeds every shuffle, and `build_dataset` raises if a sample appears in both files |
| 1.2 | Measure Precision@1 on the held-out split and put it back in the README **with its denominator and date** |
| 1.3 | Measure throughput once, on named hardware. It has never been measured |
| 1.4 | Ship the new `.ftz`, and record the corpus digest it was trained on |

**Blocked on data, not on code.** `datasets/{mon,burmese,english}` hold only
`.gitkeep`; the corpora live elsewhere. Nothing in this repository can run 1.1
until they are placed.

**Until 1.4, do not publish to PyPI.** The package is GitHub-only, which is what
keeps the stale artifact from becoming a distribution problem.

**Exit:** the README states an accuracy figure again, and the number, the
pipeline and the shipped artifact all come from the same run.

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

## 3. CI

`make check` exists and is green. Nothing runs it but a person.

Copy `mon_tokenizer/.github/workflows/ci.yml`, where the actions are already
pinned to commit SHAs. This is the last item from the 2026-08-09 tooling list;
3.1 through 3.3 closed in `32ed3d5`.

---

## What we are deliberately not doing

| Not doing | Why |
|---|---|
| **Replacing fastText** | It is the right boring choice for language identification at this size, and nothing in either audit pointed at the model family. The failures were the split, the character set and the missing out-of-domain answer — none of them the classifier |
| **Adding languages to the label set** | Mon, Burmese and English with mixed-script labels covers the ecosystem's actual need. A fourth *class* is §2.1 and is a data problem; a fourth *label* without data would be a worse lie than `unknown` |
| **Publishing to PyPI or Hugging Face before §1** | GitHub-only is currently a feature |
| **Restating the old 0.925** | It was measured on memorised rows. It does not become true by being cited carefully |
| **Deriving `MON_EXCLUSIVE_CODEPOINTS` at import time** | The test re-derives it from Unicode names on every run, which is the same guarantee without paying ~500 `unicodedata.name` calls at import and without the set changing silently when Python's Unicode data updates |

---

## How this file stays true

Every measurement above carries the date it was taken. When §1 closes, move the
new figures into the README with their denominators, record the closure in
`AUDIT-2026-08-08.md`'s status table, and delete §1 from this file.

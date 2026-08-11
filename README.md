# Mon Language Detector

Detects Mon (`mnw`), Burmese (`mya`) and English (`eng`), including mixed-script
text. Built for filtering scraped corpora.

fastText over character n-grams, with Unicode rules on top: Mon-exclusive
characters override the model, other Myanmar-script languages are refused, and
every result carries a reliability flag.

## Use

```python
from mon_language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.predict("ပ္ဍဲသၞာံ ၁၉၉၀")

if result.reliable and result.label.startswith("mnw"):
    keep(result)
```

`result.basis` says where `confidence` came from — `posterior`,
`mon-exclusive`, `other-myanmar-script`, `ambiguous-myanmar`, `too-short`,
`no-script` or `empty`. Threshold on `reliable`, not on `confidence`.

| Label | Meaning |
| :--- | :--- |
| `mnw` `mya` `eng` | One language |
| `mnw-eng` `mya-eng` | Mixed with English |
| `mnw-mya` | Myanmar script, too short to tell which |
| `unknown` | Empty, no script, or another Myanmar-script language |

## Accuracy

Retrained 2026-08-11. `make evaluate` reproduces these, once a split exists —
`data/` is not in the repository, so build one first (below).

| | |
|---|---|
| Where `reliable` | **0.9980** over 27,085 lines, 77.4% of the split |
| Overall | 0.9565 over 34,988 held-out lines |
| Per class | `eng` 1.0000 · `mya` 0.9675 · `mnw` 0.9107 |
| Model | 8.10 MB `.ftz`, quantized |
| Throughput | 28,149 lines/s, single-threaded, Apple M5 |

Corpus filtering keeps the reliable rows and drops the rest, so the first row is
the one that describes the workload.

Every remaining error is Mon against Burmese, and it is a length problem: 97.7%
of them are in lines of 40 characters or fewer. Lines carrying a Mon-exclusive
character had none. `MIN_UNAMBIGUOUS_MYANMAR_LEN` is set at 30 on that evidence,
tuned on the split it is scored on — so 0.9980 is optimistic by an unknown
margin.

## What it cannot do

Three classes, and the Myanmar script is shared by more. Shan, Khamti, Aiton,
Karen and Palaung have nowhere to land, and a Mon scrape collects them.

Text carrying a character exclusive to one of those languages now returns
`unknown`. That covers the ones a character can prove; Shan written without them
is still labelled Mon. A fourth class needs Shan data nobody has.

## Building a model

```bash
uv run datasets --explain                      # what the selection keeps and drops
uv run datasets --corpus-root /path/to/corpus  # link a corpus into datasets/
uv run pipeline --target-mon 150000 --target-mya 150000 --target-eng 150000
uv run train --epoch 20 --dim 128
uv run evaluate
```

Point `--corpus-root` at a corpus laid out as a directory per source, each
holding `.txt` files. A directory records where a line came from, not what
language it is, so the selection step is where that gets fixed; `--explain`
shows the reasoning.

Targets are 150,000 because Burmese has 46,407 unique lines. More buys
repetition.

`uv run wrangle` cleans raw files first. `uv run preview` spot-checks a model.
`make check` runs ruff, mypy and the tests.

## Deployment

The `.ftz` works with the standard fastText bindings on Android, iOS and WASM.

Install with `pip install mon-language-detector`. The `wrangle` CLI needs
`[wrangle]` on top; detection does not.

## Documents

- [CHANGELOG.md](CHANGELOG.md) — what changed in each version, with the numbers
- [docs/AUDIT-2026-08-08.md](docs/AUDIT-2026-08-08.md) — what was found, and which commit closed it
- [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) — what is left, and what is deliberately not being done

# Mon Language Detector

Identification tool for Mon (mnw), Burmese (mya), and English (eng) text. Optimized for corpus filtering and production-scale detection.

## Architecture

- **Engine**: fastText (Character n-grams 2-5).
- **Inference Strategy**: Hybrid model prediction + Unicode script-ratio heuristics.
- **Categorization**: Identification of mixed-language text (e.g., `mnw-eng`) and reliability guarding.
- **Normalization**: Mandatory NFC normalization and zero-width character stripping.

## Classification Labels

| Label | Context |
| :--- | :--- |
| `mnw` | Pure Mon text |
| `mya` | Pure Burmese text |
| `eng` | Pure English text |
| `mnw-eng` | Mixed Mon and English |
| `mya-eng` | Mixed Burmese and English |
| `mnw-mya` | Ambiguous shared Myanmar script (e.g., single characters) |
| `unknown` | Empty or non-identifiable input |

## Performance Data

Retrained 2026-08-11 on a clean split. Reproduce with `make evaluate`.

| | |
|---|---|
| **Detector accuracy** | **0.9565** over 34,988 held-out lines |
| **Accuracy where `reliable`** | **0.9888** over 29,102 lines (83.2% of the split) |
| **Model size** | 7.72 MB (`.ftz`), quantized |
| **Throughput** | still not measured |

Per class, and the aggregate does not speak for all three:

| class | accuracy | n |
|---|---:|---:|
| `eng` | 1.0000 | 15,000 |
| `mya` | 0.9675 | 4,640 |
| `mnw` | 0.9107 | 15,348 |

**Read the reliable-only row.** The documented use is corpus filtering, which
keeps `reliable` rows and drops the rest, so accuracy over the whole split
describes a workload nobody runs.

**Every remaining error is Mon against Burmese** — 1,367 `mnw` lines labelled
`mya` and 151 the other way. There is no `eng` confusion left in either
direction. The two languages share the script, and only about half of Mon lines
carry a Mon-exclusive character for the hard signal to find.

Two numbers that are *not* this one: `make train` reports fastText's
Precision@1 (0.9561), which is the classifier without the detector's hard
signal, out-of-domain guard and reliability flag. And the model was trained on
150,000 lines per class — Burmese has only 46,407 unique lines available, so its
training side is repeated about 3.6×, while Mon and English are near their
natural size.

### How the split is kept honest

Lines are deduplicated, each language is split before any upsampling, upsampling
applies to the train side only, and synthetic code-switched samples are derived
per split. Every shuffle is seeded. `build_dataset` raises if a sample reaches
both files.

An earlier release published 0.925 measured on a split that shared rows with
training. That number is withdrawn and is not comparable with the one above.

### What the detector cannot do

**It answers a three-way question.** Mon, Burmese and English are the only
classes; fastText has no way to say "none of these", so anything else must land
on one of the three.

The Myanmar script is shared with Shan, Khamti, Aiton, Karen and Palaung, and a
Mon corpus scrape collects them. Measured on the shipped model, Shan came back
`mnw` at confidence 1.0000 with `reliable=True`. Since 2026-08-11 text carrying a
character exclusive to one of those languages, and no Mon-exclusive character, is
returned as `unknown` with `basis="other-myanmar-script"` instead — see
`OTHER_MYANMAR_LANGUAGE_CODEPOINTS`. Measured cost on the corpora: the guard
fires on 0.051% of 306,564 Mon lines and 0% of 9,065 Burmese lines.

**That guard is characters, not comprehension.** Shan written without any of
those 124 codepoints is still labelled Mon, and a fourth class would need Shan
training data nobody has. Filter on `reliable` and check `basis`.

### Where the reasoning lives

| Document | What it answers |
|---|---|
| [docs/AUDIT-2026-08-08.md](docs/AUDIT-2026-08-08.md) | What was found, with the evidence, and which commit closed each finding |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | What is left, in order, and what is deliberately not being done |

## CLI Reference

Run in this order. `make check` gates all of it.

### 1. Corpus selection
```bash
uv run datasets --explain                      # what is kept, what is dropped, why
uv run datasets --corpus-root /path/to/corpus  # symlink the selection into datasets/
```
The corpus is not in this repository. Point `--corpus-root` at one laid out as
a directory per source, each holding `.txt` files with one line per sentence.

A directory records where a line came from, not what language it is. The corpus
this model was trained on is organised for OCR, where the label is the text on
the image, and four of its Mon directories are English or Burmese by content.
This step is where they are excluded; the tables in `datasets.py` are that
corpus's names and your own will differ.

### 2. Dataset pipeline
```bash
uv run pipeline --target-mon 150000 --target-mya 150000 --target-eng 150000
```
150,000 and not a million: Burmese has 46,407 unique lines, so a larger target
buys repetition rather than data. Raises if any sample reaches both splits.

### 3. Model training
```bash
uv run train --epoch 20 --dim 128
```
Reports per-class precision, recall and F1 for the full model and again for the
quantized one — quantization is lossy and the loss is not spread evenly.

### 4. Evaluation
```bash
uv run evaluate              # scores the DETECTOR; this is what the README cites
```

### Data wrangling
```bash
uv run wrangle --input <dir> --output <dir>
```

### Model Preview
```bash
uv run preview
```

## Production Deployment

The quantized model (`data/*.ftz`) is designed for cross-platform production usage.

### 1. Python SDK
Use the provided `LanguageDetector` class for high-throughput server-side filtering.
```python
from mon_language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.predict("ပ္ဍဲသၞာံ ၁၉၉၀")

# Filter on `reliable`, and read `basis` when you need to know why.
if result.reliable and result.label.startswith("mnw"):
    keep(result)
```

`result.basis` says where `confidence` came from: `posterior` (the model's
probability), `mon-exclusive` (a character decided it, and the number is a
constant), `other-myanmar-script` (refused — Shan, Khamti, Karen, Aiton or
Palaung), or one of `ambiguous-myanmar` / `too-short` / `no-script` / `empty`.

### 2. Mobile (Android/iOS)
The `.ftz` format is compatible with the standard fastText mobile libraries.
- **Android**: Use `com.github.facebookresearch:fastText`.
- **iOS**: Use the fastText C++ interface via Objective-C++.

### 3. Web (WASM)
Deploy to the browser using `fasttext-wasm` or similar wrappers. The 7.72 MB footprint allows for efficient edge-side detection without server round-trips.


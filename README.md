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

- **Precision@1**: withdrawn pending a retrain — see below
- **Model Size**: 7.4 MB (`.ftz`)
- **Throughput**: not currently measured

### Why the accuracy figure was withdrawn

The published 0.925 was measured on a validation split that shared rows with
training. `pipeline.py` upsampled short languages by repeating lines and only
then shuffled and split 90/10, so every repeated line landed on both sides. Mon
was the language most likely to fall short of its target, so it was the most
duplicated and the worst affected. The number was therefore measured partly on
memorised training rows, and its true value is unknown.

The pipeline is fixed as of 2026-08-08: lines are deduplicated, each language is
split before any upsampling, upsampling applies to the train side only, synthetic
code-switched samples are derived per split, and `build_dataset` raises if any
sample appears in both files. Every shuffle is seeded, so a run is reproducible.

The shipped `.ftz` predates that fix. A retrain on a clean split is needed before
any accuracy figure goes back in this README. Throughput was never measured and
is removed rather than repeated.

## CLI Reference

### Data Wrangling
```bash
uv run wrangle --input <dir> --output <dir>
```

### Dataset Pipeline
```bash
uv run pipeline --target-mon 1000000 --target-mya 1000000 --target-eng 1000000
```

### Model Training
```bash
uv run train --epoch 20 --dim 128
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
```

### 2. Mobile (Android/iOS)
The `.ftz` format is compatible with the standard fastText mobile libraries.
- **Android**: Use `com.github.facebookresearch:fastText`.
- **iOS**: Use the fastText C++ interface via Objective-C++.

### 3. Web (WASM)
Deploy to the browser using `fasttext-wasm` or similar wrappers. The 7.4MB footprint allows for efficient edge-side detection without server round-trips.

---
*Built for technical robustness and linguistic accuracy.*

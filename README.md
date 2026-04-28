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

- **Precision@1**: 0.925 (Quantized Model)
- **Model Size**: 7.4 MB (`.ftz`)
- **Throughput**: ~2000 samples/sec (single thread)

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

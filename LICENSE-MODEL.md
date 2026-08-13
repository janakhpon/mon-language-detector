# Model licence and attribution

This file governs the trained model shipped inside the package:

    src/mon_language_detector/data/langid_mon_mya_eng_compressed.ftz

The code is separate. `LICENSE` is an MIT grant scoped to the Python sources and
the `Makefile`, and it does not reach this file.

**There is no single licence for this model, and this file does not invent one.**
A fastText model is a derived work of the text it was fitted to. Its terms follow
its training data, and that data has no single set of terms either.

## What it was trained on

The corpus is not in this repository. `uv run datasets --corpus-root <path>`
links it in, and `datasets.py` records which directories may train which class.
The shipped artifact was fitted on 2026-08-11 against
[MonOCR](https://github.com/MonDevHub/monocr)'s corpus at
`data/raw/corpus/`.

Shares below are **bytes of raw input**, before cleaning, deduplication and
sampling. `wc -c datasets/mon/*.txt` gives the 131,687,389-byte denominator and
the same command per prefix gives each row. They describe what fed the class, not
what survived into the split.

| Source | Share of Mon input | Terms | Status |
| :--- | ---: | :--- | :--- |
| `mon_shards/wikipedia_shard_*` — [Mon Wikipedia](https://mnw.wikipedia.org) | 54.7% | **CC BY-SA 4.0** | Established |
| `mon_shards/monnews_shard_*` — [Independent Mon News Agency](https://monnews.org) | 26.6% | **None established** | **Unresolved** |
| `mon`, `custom`, `atula_chan`, `documents` — scraped and collected Mon text | 18.3% | **None established** | **Unresolved** |
| `mon_shards/{handwritten,telegram,facebook}_*` | 0.4% | **None established** | **Unresolved** |

The `mon_shards/*` files carry the same names as the shards in
[MonCorpusCollection](https://github.com/MonDevHub/MonCorpusCollection); that
repository's `LICENSE-CORPUS.md` is the authority on their terms, keyed to the
same filename prefix.

The Burmese and English classes come from the same corpus root —
`burmese`, `burmese2`, `burmese3` and `english`, the last of which is
Alpaca-derived by its filenames. **Their terms are not recorded anywhere in this
repository either**, and they are named here so the omission is visible rather
than implied to be resolved.

## What follows

- **The model is not redistributable under MIT.** Nothing in this repository
  grants rights over it, because no such rights were obtained.
- **CC BY-SA 4.0 is the strongest established term in the mix.** Whether fitted
  model weights are Adapted Material under §2(a) — and therefore whether
  ShareAlike propagates to the `.ftz` — is an open question the maintainers have
  not taken advice on. It is flagged here rather than answered.
- **The unresolved rows are the harder problem.** IMNA is a working news agency
  and a quarter of the Mon input. No licence has been obtained and this project
  has no authority to grant one.

This is why nothing is published to PyPI or Hugging Face. The engineering is
done; see `docs/NEXT_STEPS.md` §1.

## Personal data

MonCorpusCollection removed 21 lines of personal data from its shards on
2026-08-12. This model was fitted on 2026-08-11, against a copy of those shards
taken before that pass. Whether any removed line reached the training split is
not recorded — selection, deduplication and sampling sit between the corpus and
`data/train.txt`, and that intermediate is not tracked.

## Attribution, if you use it

Attribute **Mon Language Detector** and the underlying sources of the training
data. For the Wikipedia portion, CC BY-SA 4.0 requires a licence link and an
indication of changes:

> Trained in part on text from [Mon Wikipedia](https://mnw.wikipedia.org), by its
> contributors, used under
> [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The text was
> normalized, filtered and resampled before training.

## Reporting a problem

If you hold rights in material used here and it should not have been, open an
issue on
[janakhpon/mon-language-detector](https://github.com/janakhpon/mon-language-detector)
and the model will be withdrawn pending a retrain.

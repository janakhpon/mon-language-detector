"""Score the shipped `LanguageDetector`, not the raw fastText model.

`train.py` reports fastText's Precision@1. That is the classifier's number, and
it is not what a caller gets: `LanguageDetector` adds a Mon-exclusive hard
signal, an out-of-domain guard, mixed-script labelling and a reliability flag on
top. Those change the answer in both directions, and the first clean run
measured the gap at 2.1 points — the detector scored 0.9267 where the model
scored 0.9482, because the Mon class still held English reference lines.

Every accuracy figure in the README comes from this command, so it exists rather
than living in someone's shell history.

    uv run evaluate

The reliable-only figure is the one that matters for the documented use. Corpus
filtering keeps `reliable` rows and drops the rest, so accuracy over the whole
split describes a workload nobody runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .detector import LanguageDetector
from .utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassScore:
    n: int
    accuracy: float


@dataclass(frozen=True)
class Report:
    """The evaluation result, typed.

    A `dict[str, object]` was the first shape, and mypy was right to reject it:
    every read then needs a cast or an ignore, and `report["reliable_acc"]`
    misspelled is a KeyError at print time rather than an error at check time.
    The README quotes these fields, so they are worth naming.
    """

    n: int
    accuracy: float
    reliable_n: int
    reliable_share: float
    reliable_accuracy: float
    per_class: dict[str, ClassScore]
    basis: dict[str, int]
    errors: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "reliable_n": self.reliable_n,
            "reliable_share": round(self.reliable_share, 4),
            "reliable_accuracy": round(self.reliable_accuracy, 4),
            "per_class": {
                k: {"n": v.n, "accuracy": round(v.accuracy, 4)} for k, v in self.per_class.items()
            },
            "basis": self.basis,
            "errors": self.errors,
        }


def score(detector: LanguageDetector, valid_file: Path) -> Report:
    """Accuracy overall, per class, and restricted to `reliable` predictions."""
    total = correct = reliable_total = reliable_correct = 0
    per_class: dict[str, list[int]] = {}
    errors: Counter[tuple[str, str]] = Counter()
    bases: Counter[str] = Counter()

    with open(valid_file, encoding="utf-8") as handle:
        for line in handle:
            gold_label, _, text = line.partition(" ")
            text = text.strip()
            if not text:
                continue
            gold = gold_label.replace("__label__", "")
            result = detector.predict(text)

            # `mnw-eng` on a Mon sentence with an injected English word is the
            # label the synthesiser built, not a miss. Scoring the base language
            # is the comparison the gold labels can actually support — they carry
            # one class each, and the detector's label space is larger.
            base = result.label.split("-")[0]
            hit = base == gold

            total += 1
            correct += hit
            bucket = per_class.setdefault(gold, [0, 0])
            bucket[0] += 1
            bucket[1] += hit
            bases[result.basis] += 1
            if result.reliable:
                reliable_total += 1
                reliable_correct += hit
            if not hit:
                errors[(gold, result.label)] += 1

    if not total:
        raise RuntimeError(f"{valid_file} produced no scorable lines")

    return Report(
        n=total,
        accuracy=correct / total,
        reliable_n=reliable_total,
        reliable_share=reliable_total / total,
        reliable_accuracy=(reliable_correct / reliable_total) if reliable_total else 0.0,
        per_class={
            k: ClassScore(n=v[0], accuracy=v[1] / v[0]) for k, v in sorted(per_class.items())
        },
        basis=dict(bases.most_common()),
        errors={f"{g}->{p}": c for (g, p), c in errors.most_common()},
    )


def _print(report: Report) -> None:
    logger.info("LanguageDetector on the held-out split")
    logger.info(f"  examples                : {report.n:,}")
    logger.info(f"  accuracy                : {report.accuracy:.4f}")
    logger.info(
        f"  accuracy where reliable : {report.reliable_accuracy:.4f}"
        f"   n={report.reliable_n:,} ({report.reliable_share:.1%} of the split)"
    )
    for name, stats in report.per_class.items():
        logger.info(f"    {name:<4} {stats.accuracy:.4f}  n={stats.n:,}")
    logger.info("  basis of each prediction:")
    for name, count in report.basis.items():
        logger.info(f"    {name:<22} {count:>7,}")
    if report.errors:
        logger.info("  errors (gold -> label):")
        for pair, count in list(report.errors.items())[:8]:
            logger.info(f"    {pair:<18} {count:>7,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Score the detector on a held-out split")
    p.add_argument("--valid-file", type=Path, default=PROJECT_ROOT / "data/valid.txt")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="defaults to the model bundled in the package — i.e. what callers get",
    )
    p.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = p.parse_args()

    if not args.valid_file.exists():
        raise FileNotFoundError(
            f"{args.valid_file} does not exist. Run `make datasets && make pipeline` first."
        )
    report = score(LanguageDetector(model_path=args.model), args.valid_file)
    _print(report)
    if args.json:
        args.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        logger.info(f"wrote {args.json}")


if __name__ == "__main__":
    main()

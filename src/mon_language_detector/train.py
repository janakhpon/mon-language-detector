import argparse
import time
from pathlib import Path

import fasttext

from .utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)


def _report(model: object, valid_file: Path, tag: str) -> dict[str, float]:
    """Aggregate AND per-class metrics, plus the confusion that matters.

    A single Precision@1 over an imbalanced validation set is the number that got
    this repository into trouble once already. The split here is roughly
    45k Mon / 4.7k Burmese / 26k English by natural size, so an aggregate is
    dominated by the two large classes and a collapsed Burmese class would move
    it by a couple of points at most.

    Mon and Burmese share the script and are the only pair this model can
    plausibly confuse, so that confusion is printed as a count rather than left
    to be inferred from recall.
    """
    n, precision, recall = model.test(str(valid_file))  # type: ignore[attr-defined]
    logger.info(f"--- {tag} ---")
    logger.info(f"  validation examples : {n:,}")
    logger.info(f"  Precision@1         : {precision:.4f}")
    logger.info(f"  Recall@1            : {recall:.4f}")

    per_label = model.test_label(str(valid_file))  # type: ignore[attr-defined]
    logger.info(f"  {'class':<12}{'precision':>11}{'recall':>9}{'f1':>9}")
    for label in sorted(per_label):
        m = per_label[label]
        logger.info(
            f"  {label.replace('__label__', ''):<12}"
            f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1score']:>9.4f}"
        )

    confusion: dict[tuple[str, str], int] = {}
    with open(valid_file, encoding="utf-8") as f:
        for line in f:
            gold, _, text = line.partition(" ")
            if not text.strip():
                continue
            (pred,), _ = model.predict(text.strip(), k=1)  # type: ignore[attr-defined]
            if pred != gold:
                key = (gold.replace("__label__", ""), pred.replace("__label__", ""))
                confusion[key] = confusion.get(key, 0) + 1
    if confusion:
        logger.info("  misclassifications (gold -> predicted):")
        for (gold, pred), count in sorted(confusion.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {gold:>4} -> {pred:<4} {count:>7,}")
    else:
        logger.info("  no misclassification on the validation split")

    return {"n": float(n), "precision": float(precision), "recall": float(recall)}


def validate_data_files(train_file: Path, valid_file: Path) -> None:
    """Ensure training data exists before starting."""
    if not train_file.exists() or not valid_file.exists():
        raise FileNotFoundError(
            f"Training data files missing. Run pipeline first.\nTrain: {train_file}\nValid: {valid_file}"
        )


def train_model(  # noqa: PLR0913 — eleven CLI flags, each one a hyperparameter
    *,
    train_file: Path,
    valid_file: Path,
    model_output: Path,
    quantized_output: Path,
    lr: float,
    epoch: int,
    word_ngrams: int,
    minn: int,
    maxn: int,
    dim: int,
    thread: int,
) -> None:
    """Train the fastText language identification model.

    Keyword-only for the same reason as `build_dataset`: `word_ngrams, minn,
    maxn, dim, thread` are five consecutive `int`s, and swapping `minn` with
    `maxn` or `dim` with `thread` positionally raises nothing, type-checks, and
    silently trains a different model. A run is minutes to hours; the mistake is
    only visible in the metric.
    """
    logger.info("Initializing fastText model training...")
    validate_data_files(train_file, valid_file)

    start_time = time.time()

    try:
        # Hyperparameters for maximum reliability on sub-word orthographic patterns
        model = fasttext.train_supervised(
            input=str(train_file),
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            minn=minn,
            maxn=maxn,
            dim=dim,
            loss="softmax",
            thread=thread,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"Training completed successfully in {duration:.2f} seconds.")

    logger.info("Evaluating on validation set...")
    _report(model, valid_file, "full model")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_model(str(model_output))
        size_mb = model_output.stat().st_size / (1024 * 1024)
        logger.info(f"Full model saved to {model_output} ({size_mb:.2f} MB)")
    except OSError as e:
        logger.error(f"Failed to save model to {model_output}: {e}")
        raise

    logger.info("Quantizing model for production deployment...")
    try:
        model.quantize(input=str(train_file), qnorm=True, retrain=True, cutoff=100000)
        model.save_model(str(quantized_output))

        q_size_mb = quantized_output.stat().st_size / (1024 * 1024)
        logger.info(f"Compressed model saved to {quantized_output} ({q_size_mb:.2f} MB)")
        # The quantized artifact is the one that ships, so it gets the same
        # report as the full model rather than a bare Precision@1. Quantization
        # is lossy and the loss is not spread evenly across classes.
        _report(model, valid_file, "quantized model — THIS IS WHAT SHIPS")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        # Not raising, since full model succeeded


def main():
    parser = argparse.ArgumentParser(description="Train fastText language detector")
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "train.txt")
    parser.add_argument("--valid-file", type=Path, default=PROJECT_ROOT / "data" / "valid.txt")
    parser.add_argument(
        "--model-output", type=Path, default=PROJECT_ROOT / "data" / "langid_mon_mya_eng.bin"
    )
    parser.add_argument(
        "--quantized-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "langid_mon_mya_eng_compressed.ftz",
    )

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--word-ngrams", type=int, default=1)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--thread", type=int, default=4)

    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        valid_file=args.valid_file,
        model_output=args.model_output,
        quantized_output=args.quantized_output,
        lr=args.lr,
        epoch=args.epoch,
        word_ngrams=args.word_ngrams,
        minn=args.minn,
        maxn=args.maxn,
        dim=args.dim,
        thread=args.thread,
    )


if __name__ == "__main__":
    main()

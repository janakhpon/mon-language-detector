import argparse
import random
from pathlib import Path

from .utils import MIN_RELIABLE_LEN, PROJECT_ROOT, clean_and_normalize, get_logger

logger = get_logger(__name__)

VALID_FRACTION = 0.1
DEFAULT_SEED = 20260808


def _extract_lines(path: Path, min_len: int = MIN_RELIABLE_LEN) -> list[str]:
    """Read and normalize lines from a single file.

    `min_len` is inclusive, and shares its default with the detector's
    reliability guard. Keeping a line here is what entitles the detector to
    vouch for a text of that length.
    """
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                cl = clean_and_normalize(line)
                if len(cl) >= min_len:
                    lines.append(cl)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
    return lines


def _collect(dirs: list[Path], target: int) -> list[str]:
    """Collect DEDUPLICATED lines from all .txt files in the given directories.

    Shuffles files before reading to avoid source-order bias.

    Deduplication and upsampling both moved out of here deliberately. This used
    to upsample by repeating lines, and `build_dataset` then shuffled and split
    the result 90/10 -- so every repeated line landed on both sides of the split
    and the reported Precision@1 was measured partly on memorised training rows.
    Mon is the language most likely to fall short of its target, so it was the
    most duplicated and the worst affected.

    Upsampling now happens after the split, on the train side only.
    """
    files: list[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(d.glob("**/*.txt"))
        elif d.is_file():
            files.append(d)

    random.shuffle(files)
    lines: list[str] = []
    for f in files:
        lines.extend(_extract_lines(f))
        if len(lines) >= target * 2:  # collect a 2x buffer before trimming
            break

    random.shuffle(lines)

    # Corpora contain naturally repeated lines -- boilerplate, headers, stock
    # phrases -- and those straddle the split just as upsampled copies would.
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    return unique[:target]


def _split(lines: list[str], valid_fraction: float) -> tuple[list[str], list[str]]:
    """Split into (train, valid). Input must already be deduplicated."""
    n_valid = int(len(lines) * valid_fraction)
    return lines[n_valid:], lines[:n_valid]


def _upsample(lines: list[str], target: int) -> list[str]:
    """Repeat lines up to `target`. Only ever applied to a train split.

    Upsampling an evaluation set would measure the same rows repeatedly, so
    valid is deliberately left at its natural size.
    """
    if 0 < len(lines) < target:
        factor = (target // len(lines)) + 1
        return (lines * factor)[:target]
    return lines[:target]


def _make_mixed(mon: list[str], eng: list[str], mya: list[str]) -> list[str]:
    """
    Synthesize code-switched Mon samples by injecting a foreign word mid-sentence.
    Labelled mnw since the base sentence is Mon-dominant.
    """
    if not (mon and eng and mya):
        return []

    eng_words = [w for line in eng[:10_000] for w in line.split() if len(w) > 3]
    mya_words = [w for line in mya[:10_000] for w in line.split() if len(w) > 2]

    if not eng_words or not mya_words:
        logger.warning("Insufficient donor words for code-switching synthesis.")
        return []

    samples: list[str] = []
    target = len(mon) // 10
    for _ in range(target):
        words = random.choice(mon).split()
        if len(words) < 5:
            continue
        donor = random.choice(eng_words if random.random() > 0.5 else mya_words)
        words.insert(random.randint(1, len(words) - 1), donor)
        samples.append(" ".join(words))

    logger.info(f"Generated {len(samples)} synthetic mixed samples.")
    return samples


def build_dataset(  # noqa: PLR0913 — nine CLI flags, each one a separate decision
    *,
    eng_dirs: list[Path],
    mya_dirs: list[Path],
    mon_dirs: list[Path],
    out_train: Path,
    out_valid: Path,
    target_eng: int,
    target_mya: int,
    target_mon: int,
    seed: int = DEFAULT_SEED,
) -> None:
    """Compile a labelled, shuffled fastText training dataset from raw corpora.

    Keyword-only, and not to satisfy a linter. Three of these are parallel
    `list[Path]` and three more are parallel `int`, so transposing `target_eng`
    and `target_mya` positionally is silent, type-correct, and produces a corpus
    with the wrong language balance — a defect that would surface as a confusing
    Precision@1 weeks later. The `*` makes that call unwritable.
    """
    # Every shuffle below draws from this. Without it the split, the file order
    # and the synthetic mixed samples all differ per run, so a reported metric
    # cannot be reproduced and a regression is indistinguishable from the RNG.
    random.seed(seed)
    logger.info(f"Starting data pipeline (seed={seed})...")

    # Split each language before upsampling, so no repeated line can appear on
    # both sides. _collect returns deduplicated lines for the same reason.
    eng_all = _collect(eng_dirs, target_eng)
    mya_all = _collect(mya_dirs, target_mya)
    mon_all = _collect(mon_dirs, target_mon)
    logger.info(
        f"unique  English: {len(eng_all):,}  Burmese: {len(mya_all):,}  Mon: {len(mon_all):,}"
    )

    eng_train, eng_valid = _split(eng_all, VALID_FRACTION)
    mya_train, mya_valid = _split(mya_all, VALID_FRACTION)
    mon_train, mon_valid = _split(mon_all, VALID_FRACTION)

    # Synthetic mixed samples are derived from real lines, so they are built
    # from the train side only. Building them from the whole set would put a
    # sentence in valid and its one-word-different derivative in train.
    mixed_train = _make_mixed(mon_train, eng_train, mya_train)
    mixed_valid = _make_mixed(mon_valid, eng_valid, mya_valid)

    eng_train = _upsample(eng_train, target_eng)
    mya_train = _upsample(mya_train, target_mya)
    mon_train = _upsample(mon_train, target_mon)

    def _label(pairs: list[tuple[str, list[str]]]) -> list[str]:
        return [f"__label__{tag} {line}" for tag, lines in pairs for line in lines]

    train = _label(
        [("eng", eng_train), ("mya", mya_train), ("mnw", mon_train), ("mnw", mixed_train)]
    )
    valid = _label(
        [("eng", eng_valid), ("mya", mya_valid), ("mnw", mon_valid), ("mnw", mixed_valid)]
    )
    random.shuffle(train)
    random.shuffle(valid)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)

    out_train.write_text("\n".join(train) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(train):,} samples  -> {out_train}")

    out_valid.write_text("\n".join(valid) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(valid):,} samples -> {out_valid}")

    overlap = set(train) & set(valid)
    if overlap:
        raise RuntimeError(
            f"{len(overlap):,} samples appear in both splits. The dataset is "
            f"contaminated and any metric measured on it is meaningless."
        )
    logger.info("verified: no sample appears in both splits")


def main() -> None:
    p = argparse.ArgumentParser(description="Build fastText training dataset")
    p.add_argument("--eng-dirs", nargs="+", default=[str(PROJECT_ROOT / "datasets/english")])
    p.add_argument("--mya-dirs", nargs="+", default=[str(PROJECT_ROOT / "datasets/burmese")])
    p.add_argument("--mon-dirs", nargs="+", default=[str(PROJECT_ROOT / "datasets/mon")])
    p.add_argument("--out-train", type=Path, default=PROJECT_ROOT / "data/train.txt")
    p.add_argument("--out-valid", type=Path, default=PROJECT_ROOT / "data/valid.txt")
    p.add_argument("--target-eng", type=int, default=1_000_000)
    p.add_argument("--target-mya", type=int, default=1_000_000)
    p.add_argument("--target-mon", type=int, default=1_000_000)
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seeds every shuffle, so the split is reproducible",
    )
    args = p.parse_args()

    build_dataset(
        eng_dirs=[Path(d) for d in args.eng_dirs],
        mya_dirs=[Path(d) for d in args.mya_dirs],
        mon_dirs=[Path(d) for d in args.mon_dirs],
        out_train=args.out_train,
        out_valid=args.out_valid,
        target_eng=args.target_eng,
        target_mya=args.target_mya,
        target_mon=args.target_mon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

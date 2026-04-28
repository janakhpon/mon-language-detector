import argparse
import random
from pathlib import Path
from typing import List, Optional

from .utils import get_logger, clean_and_normalize, PROJECT_ROOT

logger = get_logger(__name__)


def _extract_lines(path: Path, min_len: int = 10) -> List[str]:
    """Read and normalize lines from a single file."""
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                cl = clean_and_normalize(line)
                if len(cl) > min_len:
                    lines.append(cl)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
    return lines


def _collect(dirs: List[Path], target: int) -> List[str]:
    """
    Collect lines from all .txt files in given directories.
    Shuffles files before reading to avoid source-order bias.
    Upsamples if collected lines fall short of the target.
    """
    files: List[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(d.glob("**/*.txt"))
        elif d.is_file():
            files.append(d)

    random.shuffle(files)
    lines: List[str] = []
    for f in files:
        lines.extend(_extract_lines(f))
        if len(lines) >= target * 2:  # collect a 2× buffer before trimming
            break

    random.shuffle(lines)

    if 0 < len(lines) < target:
        factor = (target // len(lines)) + 1
        lines = (lines * factor)[:target]

    return lines[:target]


def _make_mixed(mon: List[str], eng: List[str], mya: List[str]) -> List[str]:
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

    samples: List[str] = []
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


def build_dataset(
    eng_dirs: List[Path],
    mya_dirs: List[Path],
    mon_dirs: List[Path],
    out_train: Path,
    out_valid: Path,
    target_eng: int,
    target_mya: int,
    target_mon: int,
) -> None:
    """Compile a labelled, shuffled fastText training dataset from raw corpora."""
    logger.info("Starting data pipeline...")

    eng = _collect(eng_dirs, target_eng)
    logger.info(f"English:  {len(eng):,}")
    mya = _collect(mya_dirs, target_mya)
    logger.info(f"Burmese:  {len(mya):,}")
    mon = _collect(mon_dirs, target_mon)
    logger.info(f"Mon:      {len(mon):,}")

    mixed = _make_mixed(mon, eng, mya)

    dataset = (
        [f"__label__eng {l}" for l in eng]
        + [f"__label__mya {l}" for l in mya]
        + [f"__label__mnw {l}" for l in mon]
        + [f"__label__mnw {l}" for l in mixed]
    )
    random.shuffle(dataset)

    split = int(len(dataset) * 0.9)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)

    out_train.write_text("\n".join(dataset[:split]) + "\n", encoding="utf-8")
    logger.info(f"Wrote {split:,} samples  → {out_train}")

    out_valid.write_text("\n".join(dataset[split:]) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(dataset) - split:,} samples → {out_valid}")


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
    )


if __name__ == "__main__":
    main()

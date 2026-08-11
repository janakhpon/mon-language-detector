"""Select which corpus directories may train which class.

Expects a corpus root holding one directory per source, each with `.txt` files
of one line per sentence. The source corpus this project uses is organised for
**OCR**, where a line's label is the text on the image. For language
identification the label is the *language*, and those are not the same question:
a Mon-English dictionary line is Mon data for a renderer and half English for a
classifier.

Taking its Mon grouping wholesale would put roughly 219,000 lines of English and
Burmese into the Mon class. This module is the difference, stated once, with the
measurement behind each decision. `--explain` prints it.

Point `--corpus-root` at your own corpus and adjust the two tables below; the
directory names are this corpus's, the reasoning is general.

All figures measured 2026-08-11 on lines surviving `clean_and_normalize` at
`MIN_RELIABLE_LEN`, over script-bearing characters only.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

# Directories under the corpus root, by the class they may train.
#
# Mon. `mon` and `mon_shards` are the corpus proper; `custom`, `atula_chan` and
# `documents` are smaller real-Mon sets.
MON_DIRS: tuple[str, ...] = ("mon", "mon_shards", "custom", "atula_chan", "documents")
BURMESE_DIRS: tuple[str, ...] = ("burmese", "burmese2", "burmese3")
ENGLISH_DIRS: tuple[str, ...] = ("english",)

# Files inside an included directory that are excluded anyway, and why.
EXCLUDED_FILES: dict[str, str] = {
    # 101,185 lines at 43.2% Latin — the same bilingual shape as `dict/` below,
    # not the Mon prose the rest of mon_shards holds. Every other shard in that
    # directory measures between 1.1% and 22.3%.
    "custom_shard_001.txt": "43.2% Latin over 101,185 lines; bilingual, not Mon prose",
}

# Directories grouped as Mon upstream that must NOT train the Mon class here.
# Kept as data rather than as a comment so `--explain` can print it and the
# decision is reviewable without reading the source.
EXCLUDED_DIRS: dict[str, str] = {
    "dict": (
        "94,963 lines at 45.6% Latin. Each line is a Mon headword followed by an "
        "ENGLISH definition — 'က  1 the first consonant of the Mon alphabet.' "
        "Training on it teaches English -> mnw."
    ),
    "mon_dict_examples": (
        "22,976 lines, 100% Myanmar script, and roughly half of them BURMESE: it "
        "is a Mon-Burmese dictionary, 'ဂဥုဲဖျေံဂမ္တဴ = အပူချဆေး၊ ကိုယ်အေးဆေး။'. "
        "The script ratio cannot see this because Mon and Burmese share the "
        "script, and 97.8% of its lines contain a Mon-exclusive character — from "
        "the quoted headword, not from the sentence. It is the worst available "
        "contamination for the Mon/Burmese boundary, which is the hardest call "
        "this detector makes."
    ),
    "proper_nouns": (
        "676 lines at 33.6% Latin — country names, one per line, in Mon, Burmese "
        "and English alike ('Afghanistan'). Grouped as Mon upstream on purpose, "
        "because the mixed-script spelling is what a renderer wants."
    ),
    "mon_generated": (
        "795 machine-authored lines (Gemini), admitted upstream as a documented "
        "exception to a 'not machine-generated' ingestion bar. A detector trained "
        "on them learns a model's idea of Mon."
    ),
}

_CLASSES: dict[str, tuple[str, ...]] = {
    "mon": MON_DIRS,
    "burmese": BURMESE_DIRS,
    "english": ENGLISH_DIRS,
}


def explain() -> str:
    """The selection, as prose, for a reviewer who is not reading the source."""
    lines = ["Included:"]
    for cls, dirs in _CLASSES.items():
        lines.append(f"  {cls:<8} {', '.join(dirs)}")
    lines.append("")
    lines.append("Excluded, with the measurement:")
    for name, why in {**EXCLUDED_DIRS, **EXCLUDED_FILES}.items():
        lines.append(f"  {name}")
        lines.append(f"      {why}")
    return "\n".join(lines)


def link_datasets(corpus_root: Path, out_root: Path, *, copy: bool = False) -> dict[str, int]:
    """Populate `out_root/{mon,burmese,english}` from the selected corpus dirs.

    Symlinks by default. The corpus is ~60 MB and lives in a sibling repository
    that has its own provenance records; copying it here would create a second
    copy with no provenance, which is how a corpus becomes unattributable.
    """
    counts: dict[str, int] = {}
    for cls, dirs in _CLASSES.items():
        target = out_root / cls
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("*.txt"):
            stale.unlink()

        selected: list[Path] = []
        for name in dirs:
            source = corpus_root / name
            if not source.is_dir():
                raise FileNotFoundError(
                    f"{source} does not exist under {corpus_root}. Pass --corpus-root "
                    f"pointing at a corpus laid out as one directory per source."
                )
            for path in sorted(source.rglob("*.txt")):
                if path.name in EXCLUDED_FILES:
                    logger.info(f"  skipping {path.name}: {EXCLUDED_FILES[path.name]}")
                    continue
                selected.append(path)

        for path in selected:
            # Flattened and prefixed: two directories both hold `001.txt`, and a
            # flat destination would silently keep one of them.
            link = target / f"{path.parent.name}__{path.name}"
            if copy:
                shutil.copyfile(path, link)
            else:
                link.symlink_to(path.resolve())
        counts[cls] = len(selected)
        logger.info(f"{cls:<8} {len(selected):>3} files")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required, with no default. It used to default to a path under the author's
    # home directory, which is not a corpus location for anyone else and is not a
    # thing a published package should assume.
    p.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="corpus root: one directory per source, each holding .txt files",
    )
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "datasets")
    p.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    p.add_argument("--explain", action="store_true", help="print the selection and exit")
    args = p.parse_args()

    if args.explain:
        print(explain())
        return

    logger.info(f"selecting from {args.corpus_root}")
    link_datasets(args.corpus_root, args.out_root, copy=args.copy)
    logger.info("done. Next: make pipeline")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - exercised by installing without the extra
    # pandas, pyarrow and tqdm moved to the `wrangle` extra in 0.2.0: they are
    # over 100 MB of wheels and nothing in the detection path imports them.
    # A traceback here would read as a broken package rather than a missing
    # option, so say which one.
    raise ImportError(
        "`wrangle` needs the optional dependencies it is named after:\n"
        "    uv sync --extra wrangle        # or: pip install 'mon-language-detector[wrangle]'\n"
        "Detection itself does not need them."
    ) from exc

from .utils import clean_and_normalize, get_logger

logger = get_logger(__name__)

_COMMON_TEXT_COLS = ("text", "sentence", "transcription", "line", "content", "raw_text")


def _guess_text_col(df: pd.DataFrame) -> str | int:
    """Return the most likely text column name or integer index."""
    for col in _COMMON_TEXT_COLS:
        if col in df.columns:
            return col
    return 1 if len(df.columns) >= 2 else 0


def _save(lines, output: Path) -> None:
    """Normalize and write lines to disk."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc=output.name, leave=False):
            cl = clean_and_normalize(str(line))
            if cl:
                f.write(cl + "\n")
                count += 1
    logger.info(f"Wrote {count:,} lines → {output}")


class CorpusCleaner:
    """Transforms raw corpus files (Parquet, TSV, CSV, TXT) into clean text."""

    def process_file(
        self,
        src: Path,
        dst: Path,
        text_col: str | None = None,
        header: bool | None = None,
    ) -> None:
        ext = src.suffix.lower()
        try:
            if ext == ".parquet":
                df = pd.read_parquet(src)
                col = text_col or _guess_text_col(df)
                texts = df[col].astype(str)

            elif ext in (".tsv", ".csv"):
                sep = "\t" if ext == ".tsv" else ","
                if header is None:
                    probe = pd.read_csv(src, sep=sep, nrows=0)
                    header = any(c.lower() in _COMMON_TEXT_COLS for c in probe.columns)
                df = pd.read_csv(src, sep=sep, header=0 if header else None, on_bad_lines="skip")
                col = text_col or _guess_text_col(df)
                texts = (df.iloc[:, int(col)] if isinstance(col, int) else df[col]).astype(str)

            elif ext == ".txt":
                texts = src.read_text(encoding="utf-8").splitlines()

            else:
                logger.warning(f"Unsupported format '{ext}': {src}")
                return

            _save(texts, dst)

        except Exception as e:
            logger.error(f"Failed to process {src}: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch clean and normalize corpus files")
    p.add_argument("--input", type=Path, required=True, help="File or directory")
    p.add_argument("--output", type=Path, required=True, help="Output file or directory")
    p.add_argument("--text-col", type=str, help="Text column name (auto-detected if omitted)")
    p.add_argument("--header", choices=["true", "false"], help="Override header detection")
    args = p.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    header = {"true": True, "false": False}.get(args.header)
    cleaner = CorpusCleaner()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = [
            f
            for ext in ("*.parquet", "*.tsv", "*.csv", "*.txt")
            for f in args.input.glob(ext)
            if "_cleaned" not in f.name
        ]
        if not files:
            logger.warning(f"No supported files found in {args.input}")
            return
        for src in files:
            dst = args.output / f"{src.stem}_cleaned.txt"
            cleaner.process_file(src, dst, text_col=args.text_col, header=header)
    else:
        cleaner.process_file(args.input, args.output, text_col=args.text_col, header=header)


if __name__ == "__main__":
    main()

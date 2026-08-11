import argparse
import sys
from pathlib import Path

from .detector import LanguageDetector
from .utils import get_logger

logger = get_logger(__name__)

# Representative test cases for default preview
DEFAULT_SAMPLES = [
    "ပ္ဍဲသၞာံ ၁၉၉၀ ဂှ် ဒှ်သၞာံ",  # Pure Mon
    "နေကောင်းလားခင်ဗျာ",  # Pure Burmese
    "The project is ready for production.",  # Pure English
    "Computer သွက်ဂွံစကာ",  # Mon-Eng Mixed
    "Hello နေကောင်းလား",  # Burmese-Eng Mixed
    "ၝ",  # Mon-exclusive single char → mnw
    "အ",  # Shared Myanmar script, short → mnw-mya
]


def main():
    parser = argparse.ArgumentParser(description="Mon Language Detector Preview Tool")
    parser.add_argument("text", type=str, nargs="?", help="Text to identify")
    parser.add_argument("--file", type=Path, help="Batch process text file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")

    args = parser.parse_args()

    try:
        detector = LanguageDetector()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    def show(text: str):
        res = detector.predict(text)
        warn = " [!]" if not res.reliable else ""
        print(f"[{res.label:^7}] ({res.confidence:6.2%}){warn} {text.strip()}")

    # 1. Interactive Mode
    if args.interactive:
        print("Mon Detector REPL (Type 'exit' to quit)")
        while True:
            try:
                line = input("> ")
                if line.lower() in ("exit", "quit"):
                    break
                show(line)
            except (EOFError, KeyboardInterrupt):
                break
        return

    # 2. File Mode
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                show(line)
        return

    # 3. Single String Mode
    if args.text:
        show(args.text)
        return

    # 4. Pipe Mode (STDIN)
    if not sys.stdin.isatty():
        for line in sys.stdin:
            show(line)
        return

    # 5. Default: Run hardcoded test samples
    print(f"--- Running Default Test Samples ({len(DEFAULT_SAMPLES)}) ---")
    for sample in DEFAULT_SAMPLES:
        show(sample)
    print("\nTip: Run with text argument or --interactive for custom testing.")


if __name__ == "__main__":
    main()

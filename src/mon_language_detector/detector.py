import re
import unicodedata
from pathlib import Path
from typing import List, NamedTuple, Optional

import fasttext

from .utils import (
    MIN_RELIABLE_LEN,
    MIN_UNAMBIGUOUS_MYANMAR_LEN,
    clean_and_normalize,
    default_model_path,
    get_logger,
)

logger = get_logger(__name__)


def _is_script_bearing(c: str) -> bool:
    """Letters and combining marks only.

    Myanmar text is dense with medials and vowel signs (U+103B MEDIAL YA and
    friends), which are category Mn, so filtering on `str.isalpha()` alone would
    discard most of the Myanmar signal. Spaces, digits, punctuation and symbols
    carry no language signal in any of the three languages and are excluded.
    """
    return unicodedata.category(c)[0] in ("L", "M")


def _is_latin(c: str) -> bool:
    return "A" <= c <= "Z" or "a" <= c <= "z" or "À" <= c <= "ɏ"


def _is_myanmar(c: str) -> bool:
    return "က" <= c <= "႟" or "ꩠ" <= c <= "ꩿ"


class Detection(NamedTuple):
    # No path produces "mixed"; it was listed here and never emitted.
    label: str        # mnw | mya | eng | mnw-eng | mya-eng | mnw-mya | unknown
    confidence: float
    reliable: bool


class LanguageDetector:
    """
    Language identifier for Mon (mnw), Burmese (mya), and English (eng).

    Decision order:
      1. Empty / blank → unknown
      2. Mon-exclusive Unicode chars → mnw  (hard linguistic signal; length-independent)
      3. Short Myanmar-only text with no Mon-exclusive chars → mnw-mya  (ambiguous, not unknown)
      4. Too short for non-Myanmar text → unknown
      5. Neural prediction (fastText)
      6. Script-ratio analysis (mixed-language labelling)
      7. Reliability guard
    """

    # Characters that appear in Mon but not standard Burmese.
    _MON_RE = re.compile(
        r"[\u105A-\u1060"   # Mon medials
        r"\u106E-\u1070"    # Mon finals
        r"\u1075-\u107C"    # Mon vowels
        r"\u1085\u1086"     # Mon-specific signs
        r"\u109A-\u109D"    # Mon asat/vowel marks
        r"\uAA60-\uAA7B]"   # Mon Extensions block
    )
    _FASTTEXT_LABELS = {"__label__eng": "eng", "__label__mnw": "mnw", "__label__mya": "mya"}

    def __init__(self, model_path: Optional[Path] = None) -> None:
        path = Path(model_path) if model_path else default_model_path()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(str(path))

    def predict(self, text: str) -> Detection:
        """Classify a single text string."""
        cleaned = clean_and_normalize(text)
        if not cleaned:
            return Detection("unknown", 0.0, False)

        has_mon = bool(self._MON_RE.search(cleaned))

        # Hard Mon signal: Mon-exclusive chars are definitive regardless of length.
        # We still continue below to apply mixed-script labelling where applicable.
        if has_mon and len(cleaned) < 5:
            return Detection("mnw", 0.95, True)

        # Short Myanmar-only text with no Mon-exclusive chars:
        # We know it's Myanmar script but can't distinguish Mon from Burmese.
        # Return mnw-mya (ambiguous) rather than unknown — that's the truth.
        myanmar_only = all(
            "\u1000" <= c <= "\u109F" or "\uAA60" <= c <= "\uAA7F" or c in (" ", "\t")
            for c in cleaned
        )
        if len(cleaned) < 5 and myanmar_only:
            return Detection("mnw-mya", 0.0, False)

        # General length guard for non-Myanmar text.
        if len(cleaned) < 3:
            return Detection("unknown", 0.0, False)

        # Script ratios, over script-bearing characters only.
        #
        # This previously counted U+0000-U+024F across the whole string as
        # "Latin", a range that includes the space, every ASCII digit and every
        # ASCII punctuation mark. So "1234567890" and "!!! ,,, ???" each scored
        # latin=1.0 and came back as English, confidence 1.0, reliable=True, and
        # Mon text whose only non-Myanmar content was a year like 1990 came back
        # code-switched. For the stated use of corpus filtering, that silently
        # mislabels every numeric table row and citation block in a scrape.
        scripted = [c for c in cleaned if _is_script_bearing(c)]
        if not scripted:
            # Digits, punctuation or symbols only. There is no language here.
            return Detection("unknown", 0.0, False)

        # Neural prediction
        (raw_label,), (conf,) = self.model.predict(cleaned, k=1)
        lang = self._FASTTEXT_LABELS.get(raw_label, "unknown")
        # fastText posteriors can exceed 1.0 by a float epsilon. Clamp it, so a
        # field documented as a confidence always reads as one.
        conf = min(float(conf), 1.0)

        total = len(scripted)
        latin = sum(1 for c in scripted if _is_latin(c)) / total
        myanmar = sum(1 for c in scripted if _is_myanmar(c)) / total

        # Label synthesis
        label = lang
        if latin > 0.15 and myanmar > 0.15:
            # Mixed script
            label = "mnw-eng" if (has_mon or lang == "mnw") else "mya-eng"
        elif latin > 0.85:
            # Keep the model's posterior. Assigning the script ratio here made
            # `confidence` a probability on some paths and a ratio on others,
            # while the reliability guard below thresholds both against 0.80.
            label = "eng"
        elif has_mon and lang != "mnw":
            # Correct model miss via hard signal
            label, conf = "mnw", max(conf, 0.85)

        # Reliability guard. Both lengths are named in utils.py, and
        # MIN_RELIABLE_LEN is the same constant the training pipeline filters on
        # -- the detector does not vouch for a length the model never saw.
        reliable = conf > 0.80 and len(cleaned) >= MIN_RELIABLE_LEN
        if has_mon:
            # A Mon-exclusive character is a hard signal, not a posterior, so it
            # stands on its own at any length.
            reliable = True
        elif label in ("mnw", "mya") and len(cleaned) < MIN_UNAMBIGUOUS_MYANMAR_LEN:
            reliable = False

        return Detection(label, conf, reliable)

    def predict_batch(self, texts: List[str]) -> List[Detection]:
        return [self.predict(t) for t in texts]

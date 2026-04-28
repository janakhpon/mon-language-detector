import fasttext
import re
from pathlib import Path
from typing import List, Optional, NamedTuple

from .utils import get_logger, clean_and_normalize, PROJECT_ROOT

logger = get_logger(__name__)


class Detection(NamedTuple):
    label: str        # mnw | mya | eng | mnw-eng | mya-eng | mnw-mya | mixed | unknown
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
        path = model_path or PROJECT_ROOT / "data" / "langid_mon_mya_eng_compressed.ftz"
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

        # Neural prediction
        (raw_label,), (conf,) = self.model.predict(cleaned, k=1)
        lang = self._FASTTEXT_LABELS.get(raw_label, "unknown")
        conf = float(conf)

        # Script-ratio analysis
        total = len(cleaned)
        latin   = sum(1 for c in cleaned if "\u0000" <= c <= "\u024F") / total
        myanmar = sum(1 for c in cleaned if "\u1000" <= c <= "\u109F" or "\uAA60" <= c <= "\uAA7F") / total

        # Label synthesis
        label = lang
        if latin > 0.15 and myanmar > 0.15:
            # Mixed script
            label = "mnw-eng" if (has_mon or lang == "mnw") else "mya-eng"
        elif latin > 0.85:
            label, conf = "eng", latin
        elif has_mon and lang != "mnw":
            # Correct model miss via hard signal
            label, conf = "mnw", max(conf, 0.85)

        # Reliability guard
        # Short Myanmar-only text with no Mon-exclusive chars is inherently ambiguous.
        # Threshold is 20 chars (not 15) to avoid incorrectly flagging everyday Burmese phrases.
        reliable = conf > 0.80 and len(cleaned) >= 10
        if has_mon:
            reliable = True
        elif label in ("mnw", "mya") and len(cleaned) < 20 and not has_mon:
            reliable = False

        return Detection(label, conf, reliable)

    def predict_batch(self, texts: List[str]) -> List[Detection]:
        return [self.predict(t) for t in texts]

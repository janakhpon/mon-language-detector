import re
import unicodedata
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple

import fasttext

from .utils import (
    MIN_RELIABLE_LEN,
    MIN_UNAMBIGUOUS_MYANMAR_LEN,
    clean_and_normalize,
    default_model_path,
    get_logger,
)

logger = get_logger(__name__)


def _unicode_name(cp: int) -> str:
    """The character's Unicode name, or "" if the codepoint is unassigned."""
    try:
        return unicodedata.name(chr(cp))
    except ValueError:
        return ""


# Characters that occur in Mon and not in standard Burmese.
#
# **This set is a hard signal, not a hint.** A match overrides the model's label
# and sets `reliable=True` at any length, so a wrong member is a confident wrong
# answer and a missing member is evidence thrown away. It gets the scrutiny a
# posterior does not need.
#
# It was six hand-written ranges commented "Mon medials / Mon finals / Mon vowels
# / Mon-specific signs / Mon asat / Mon Extensions block". **Of the 45 codepoints
# they matched, 7 were Mon.** The rest were other languages that share the
# Myanmar script — Eastern Pwo Karen (U+106E-U+1070), Shan (U+1075-U+107C,
# U+1085, U+1086), Khamti (U+109A, U+109B, most of Myanmar Extended-A), Aiton
# (U+109C, U+109D, U+AA7A) and Pao Karen (U+AA7B). Shan text was therefore
# returned as `mnw` with confidence 1.000 and `reliable=True`, verified live.
# Three genuinely Mon-exclusive characters were missing at the same time.
#
# The rule is now stated rather than approximated: **a codepoint belongs here iff
# its Unicode name carries MON as a word.** That is what
# `test_the_set_is_exactly_what_unicode_calls_mon` re-derives, over all four
# Myanmar blocks — a range list cannot be checked that way, which is how the Shan
# and Karen entries survived.
#
# Verified against real corpora (`mon_OCR/data/raw/corpus`, 2026-08-11):
# 4,792,030 Mon characters and 552,394 Burmese. **Every one of these ten occurs
# in Mon and none occurs in Burmese**, from 360 (U+1028) to 95,104 (U+105A).
# At line level over 306,564 Mon lines, the corrected set fires on 39.99% against
# the old 36.62% — 10,395 lines that carried a hard signal nothing could see.
#
# Deliberately NOT here: U+103A ASAT and U+1035 VOWEL SIGN E ABOVE. `mon_OCR`
# lists both under `MON_REQUIRED_CODEPOINTS`, which answers a different question
# — whether a *font* can draw Mon — and U+103A appears throughout Burmese.
# Exclusivity and capability are not the same set.
MON_EXCLUSIVE_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x1028,  # MYANMAR LETTER MON E                    (Burmese uses U+1027)
        0x1033,  # MYANMAR VOWEL SIGN MON II
        0x1034,  # MYANMAR VOWEL SIGN MON O
        0x105A,  # MYANMAR LETTER MON NGA
        0x105B,  # MYANMAR LETTER MON JHA
        0x105C,  # MYANMAR LETTER MON BBA
        0x105D,  # MYANMAR LETTER MON BBE
        0x105E,  # MYANMAR CONSONANT SIGN MON MEDIAL NA
        0x105F,  # MYANMAR CONSONANT SIGN MON MEDIAL MA
        0x1060,  # MYANMAR CONSONANT SIGN MON MEDIAL LA
    }
)

# Built from the set, so there is one place to change and no second list to drift.
_MON_EXCLUSIVE_RE = re.compile(f"[{''.join(chr(c) for c in sorted(MON_EXCLUSIVE_CODEPOINTS))}]")


# The other languages written in the Myanmar script, and the reason this model
# needs a guard it did not have.
#
# These are the 124 codepoints exclusive to Shan, Khamti, Aiton, Sgaw and Pwo
# Karen, Rumai Palaung and Tai Laing — including the 38 the old regex called
# "Mon". They are the exact opposite signal: their presence is evidence the text
# is **not** Mon, Burmese or English.
#
# The guard is needed because fixing the regex does not fix the model. fastText
# has three classes and no out-of-domain option, so Shan — Myanmar script,
# orthographically nearest to Mon — has nowhere else to go. Measured on the
# shipped `.ftz` after the regex fix:
#
#     "ၵၸၺၼႁလိၵ်ႈတႆး ၵႂၢမ်းတႆး ၼႆႉပဵၼ်ၽႃႇသႃႇ"   -> mnw, 1.0000, reliable=True
#     "မိူင်းတႆး ပဵၼ်မိူင်းၼိုင်ႈ ၼႂ်းမိူင်းႁူမ်ႈတုမ်"  -> mnw, 0.9998, reliable=True
#
# For the stated use — filtering a scraped corpus — that is Shan entering a Mon
# dataset marked reliable, which is the failure this whole detector exists to
# prevent. No retrain fixes it either: a fourth class needs Shan data nobody has.
#
# The rule is **other-language characters present AND no Mon-exclusive character
# present**, not merely the first. A Mon document quoting a Shan word keeps its
# Mon signal and its label; only text with no Mon evidence at all is refused.
# Measured on 306,564 Mon lines and 9,065 Burmese: the guard fires on 156 Mon
# lines (0.051%) and 0 Burmese, against catching every Shan and Khamti sample
# tested. The 156 contain Shan characters and no Mon ones, so refusing to vouch
# for them is the correct answer rather than the cost.
OTHER_MYANMAR_LANGUAGE_CODEPOINTS: frozenset[int] = frozenset(
    cp
    for block in (range(0x1000, 0x10A0), range(0xA9E0, 0xAA00), range(0xAA60, 0xAA80))
    for cp in block
    for name in (_unicode_name(cp),)
    if name
    and "MON" not in name.split()
    and {"SHAN", "KHAMTI", "KAREN", "AITON", "PALAUNG", "LAING"} & set(name.split())
)

_OTHER_MYANMAR_RE = re.compile(
    f"[{''.join(chr(c) for c in sorted(OTHER_MYANMAR_LANGUAGE_CODEPOINTS))}]"
)


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


Basis = Literal[
    "posterior",  # confidence is the model's probability
    "mon-exclusive",  # confidence is a hand-chosen constant; a character decided this
    "other-myanmar-script",  # refused: Shan/Khamti/Karen/Aiton/Palaung, no Mon signal
    "ambiguous-myanmar",  # Myanmar script, too short to separate Mon from Burmese
    "too-short",  # below the length any evidence supports
    "no-script",  # digits, punctuation or symbols only
    "empty",
]


class Detection(NamedTuple):
    # No path produces "mixed"; it was listed here and never emitted.
    label: str  # mnw | mya | eng | mnw-eng | mya-eng | mnw-mya | unknown
    confidence: float
    reliable: bool

    # What KIND of number `confidence` is. Audit finding H1, and the reason it
    # stayed open: `confidence` is a fastText posterior on one path and a
    # hand-chosen 0.95 or 0.85 on another, so a caller thresholding at 0.9 was
    # selecting for branch rather than for certainty and had no way to tell.
    #
    # Returning the posterior everywhere was the alternative. It would be a lie
    # in the other direction — a Mon-exclusive character is not 0.95 likely to be
    # Mon, it is a categorical fact, and flattening it into a probability throws
    # away the strongest evidence this detector has.
    #
    # Appended with a default, so `Detection(label, conf, reliable)` and every
    # existing attribute read keep working unchanged.
    basis: Basis = "posterior"


class LanguageDetector:
    """
    Language identifier for Mon (mnw), Burmese (mya), and English (eng).

    Decision order:
      1. Empty / blank → unknown
      2. Another Myanmar-script language with no Mon signal → unknown  (see
         OTHER_MYANMAR_LANGUAGE_CODEPOINTS: the model has three classes and
         cannot say "none of these", so this branch says it instead)
      3. Mon-exclusive Unicode chars → mnw  (hard linguistic signal; length-independent)
      4. Short Myanmar-only text with no Mon-exclusive chars → mnw-mya  (ambiguous, not unknown)
      5. Too short for non-Myanmar text → unknown
      6. Neural prediction (fastText)
      7. Script-ratio analysis (mixed-language labelling)
      8. Reliability guard

    Every `Detection` carries a `basis` saying which of these produced it, and
    therefore whether `confidence` is a probability or a constant.

    **This detector answers a three-way question.** Text outside Mon, Burmese
    and English is `unknown` where a character proves it and mislabelled where
    nothing does — the guard covers the Myanmar-script neighbours because they
    are the ones a Mon corpus scrape actually collects.
    """

    _MON_RE = _MON_EXCLUSIVE_RE
    _OTHER_RE = _OTHER_MYANMAR_RE
    _FASTTEXT_LABELS: ClassVar[dict[str, str]] = {
        "__label__eng": "eng",
        "__label__mnw": "mnw",
        "__label__mya": "mya",
    }

    def __init__(self, model_path: Path | None = None) -> None:
        path = Path(model_path) if model_path else default_model_path()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(str(path))

    def _mixed_label(self, cleaned: str, has_mon: bool) -> tuple[str, Basis, bool]:
        """Label a text that carries both Latin and Myanmar in quantity.

        Returns `(label, basis, myanmar_is_judgeable)`. The third is whether the
        Myanmar half rests on enough characters to separate Mon from Burmese;
        `predict` turns a False into `reliable=False`.

        Extracted because it stopped being one expression. It was:

            "mnw-eng" if (has_mon or lang == "mnw") else "mya-eng"

        which conflates two questions — what the whole string is, and which
        Myanmar language is inside it. See the caller for how that produced
        `mya-eng` on Mon text with nothing suggesting Burmese.
        """
        if has_mon:
            # A Mon-exclusive character settles it without a second prediction.
            return "mnw-eng", "mon-exclusive", True

        fragment = "".join(c for c in cleaned if _is_myanmar(c))
        (fragment_label,), _ = self.model.predict(fragment, k=1)
        side = self._FASTTEXT_LABELS.get(fragment_label, "mya")
        label = f"{side}-eng" if side in ("mnw", "mya") else "mya-eng"
        return label, "posterior", len(fragment) >= MIN_UNAMBIGUOUS_MYANMAR_LEN

    def predict(self, text: str) -> Detection:  # noqa: PLR0911
        """Classify a single text string.

        Seven returns, one per rung of the decision order above. Collapsing them
        into a single exit would mean carrying the outcome in mutable locals
        through every branch that cannot apply to it, and the flat form is what
        lets each `return` sit under the comment explaining why that rung exists.
        `basis` names which one fired.
        """
        cleaned = clean_and_normalize(text)
        if not cleaned:
            return Detection("unknown", 0.0, False, "empty")

        has_mon = bool(self._MON_RE.search(cleaned))

        # Another Myanmar-script language, with no Mon evidence to outweigh it.
        #
        # This runs before every other branch because the model cannot express
        # the answer: it has three classes and returns `mnw` at confidence 1.0000
        # for Shan. A character exclusive to Shan, Khamti, Karen, Aiton, Palaung
        # or Tai Laing is categorical evidence that none of the three is right,
        # and saying so is the only honest output available.
        if not has_mon and self._OTHER_RE.search(cleaned):
            return Detection("unknown", 0.0, False, "other-myanmar-script")

        # Hard Mon signal: Mon-exclusive chars are definitive regardless of length.
        if has_mon and len(cleaned) < 5:
            return Detection("mnw", 0.95, True, "mon-exclusive")

        # Short Myanmar-only text with no Mon-exclusive chars:
        # We know it's Myanmar script but can't distinguish Mon from Burmese.
        # Return mnw-mya (ambiguous) rather than unknown — that's the truth.
        myanmar_only = all(
            "\u1000" <= c <= "\u109f" or "\uaa60" <= c <= "\uaa7f" or c in (" ", "\t")
            for c in cleaned
        )
        if len(cleaned) < 5 and myanmar_only:
            return Detection("mnw-mya", 0.0, False, "ambiguous-myanmar")

        # General length guard for non-Myanmar text.
        if len(cleaned) < 3:
            return Detection("unknown", 0.0, False, "too-short")

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
            return Detection("unknown", 0.0, False, "no-script")

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
        basis: Basis = "posterior"
        # Only the mixed branch can leave the Myanmar side undecided; every other
        # path either has the whole string in one script or a Mon-exclusive
        # character to settle it.
        myanmar_is_judgeable = True
        if latin > 0.15 and myanmar > 0.15:
            # Mixed script. Which Myanmar language is a SEPARATE question from
            # what the whole string is, and this branch used to conflate them:
            #
            #     "mnw-eng" if (has_mon or lang == "mnw") else "mya-eng"
            #
            # `lang` is the verdict on the whole string, so on a sentence that is
            # 82% Latin it is `eng` — correctly — and a Mon fragment carrying no
            # Mon-exclusive character fell to the `else` and came back `mya-eng`
            # with nothing in the input suggesting Burmese. It stayed invisible
            # while the model returned `mnw` for English text, which it did until
            # the training class was cleaned.
            #
            # Ask the question that is actually open: classify the Myanmar
            # characters on their own. Measured on the case that exposed this,
            # the sentence reads `eng` at 1.000 and its Myanmar substring reads
            # `mnw` at 1.000.
            label, basis, myanmar_is_judgeable = self._mixed_label(cleaned, has_mon)
        elif latin > 0.85:
            # Keep the model's posterior. Assigning the script ratio here made
            # `confidence` a probability on some paths and a ratio on others,
            # while the reliability guard below thresholds both against 0.80.
            label = "eng"
        elif has_mon and lang != "mnw":
            # Correct the model's miss from the hard signal. `conf` stops being a
            # posterior at this line — 0.85 is a floor someone chose — and that is
            # exactly what `basis` exists to say.
            label, conf, basis = "mnw", max(conf, 0.85), "mon-exclusive"

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
        elif not myanmar_is_judgeable:
            # A mixed-script label whose Myanmar side was decided from a fragment
            # too short to separate Mon from Burmese. The `-eng` half is solid;
            # the half a caller filtering a Mon corpus cares about is not.
            reliable = False

        return Detection(label, conf, reliable, basis)

    def predict_batch(self, texts: list[str]) -> list[Detection]:
        return [self.predict(t) for t in texts]

"""Classification tests against the real committed model.

The rest of the suite runs `LanguageDetector` against a two-line `DummyModel`
that returns 0.99 for everything, so it could not fail on a label bug. Both
defects these tests cover were live and passing:

  "1234567890"  -> eng, confidence 1.0, reliable True
  "!!! ,,, ???" -> eng, confidence 1.0, reliable True

The cause was `latin = sum(1 for c in cleaned if "\\u0000" <= c <= "\\u024F")`,
computed over the whole string. That range contains the space, every ASCII digit
and every ASCII punctuation mark, so text with no letters at all scored latin=1.0.

These load the actual `.ftz`. They are slower than the stubbed tests and worth it.
"""

from __future__ import annotations

import pytest

from mon_language_detector.detector import LanguageDetector

MON = "ဘာသာမန်ကို ဂၠာဲကေတ်"
BURMESE = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
ENGLISH = "The Mon language is spoken by about a million people."


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


@pytest.mark.parametrize(
    "text",
    ["1234567890", "!!! ,,, ???", "12 34 56", "-- ... --", "()[]{}"],
    ids=["digits", "punctuation", "spaced digits", "dashes", "brackets"],
)
def test_text_with_no_letters_is_not_a_language(detector: LanguageDetector, text: str):
    """Digits and punctuation carry no language signal in any of the three."""
    result = detector.predict(text)
    assert result.label == "unknown", f"{text!r} classified as {result.label}"
    assert not result.reliable


def test_mon_with_a_western_year_is_not_code_switched(detector: LanguageDetector):
    """A date is not English. This returned `mnw-eng` because the digits and the
    spaces were counted as Latin script."""
    assert detector.predict("ဘာသာမန် ၁၉၉၀ 1990").label == "mnw"


@pytest.mark.parametrize(
    "text,expected",
    [(MON, "mnw"), (BURMESE, "mya"), (ENGLISH, "eng")],
    ids=["mon", "burmese", "english"],
)
def test_monolingual_text_is_classified_correctly(
    detector: LanguageDetector, text: str, expected: str
):
    assert detector.predict(text).label == expected


def test_genuine_code_switching_is_still_detected(detector: LanguageDetector):
    """The fix must not make the detector blind to real mixing."""
    assert detector.predict("ဘာသာမန် is spoken in Myanmar and Thailand today").label == "mnw-eng"


@pytest.mark.parametrize(
    "text",
    [MON, BURMESE, ENGLISH, "1234567890", "ဘာသာမန် 1990", "a", ""],
    ids=["mon", "burmese", "english", "digits", "mixed", "single char", "empty"],
)
def test_confidence_is_a_probability(detector: LanguageDetector, text: str):
    """It is documented as a confidence, so it must be in [0, 1].

    fastText posteriors can exceed 1.0 by a float epsilon, and the value was
    returned unclamped: pure Mon came back at 1.0000098. Separately, the
    `latin > 0.85` branch assigned a script *ratio* to the same field, so one
    threshold at 0.80 was comparing two different quantities.
    """
    confidence = detector.predict(text).confidence
    assert 0.0 <= confidence <= 1.0, f"{text!r} -> {confidence!r}"


def test_the_bundled_model_resolves_outside_the_repository(detector: LanguageDetector):
    """The model used to sit at the repo root, found by walking up from __file__.

    That resolved only under an editable install; a built wheel contained no
    model at all. It now lives inside the package and is found via
    importlib.resources.
    """
    from mon_language_detector.utils import default_model_path

    path = default_model_path()
    assert path.exists()
    assert "mon_language_detector" in path.parts

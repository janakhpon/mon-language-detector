"""Zawgyi is refused, because every rule after it would misread it.

Zawgyi stores Burmese in the order it is drawn and reuses Myanmar codepoints for
display forms, four of them Mon-exclusive in Unicode. Before the gate:

    "ျမန္မာႏိုင္ငံ ရန္ကုန္ၿမိဳ႕"   -> mnw, 1.0000, reliable=True

U+1033 there is Zawgyi's tall u, not Mon's II, and the model agreed with the
character rule. The gate reads storage order instead, which a codepoint cannot
fake. The Mon cases below are the reason it is as narrow as it is: each is a
pattern from real Mon text that a looser rule flags.
"""

from __future__ import annotations

import pytest

from mon_language_detector.detector import LanguageDetector, looks_like_zawgyi

# The reported sample, "Myanmar, Yangon", in Zawgyi.
REPORTED = "ျမန္မာႏိုင္ငံ ရန္ကုန္ၿမိဳ႕"

ZAWGYI_BURMESE = [
    REPORTED,
    # Carries U+1033 (Zawgyi tall u), so it used to come back reliable `mnw`.
    "ရန္ကုန္ၿမိဳ႕ကို ကြၽန္ေတာ္ သြားခဲ့သည္။",
    "ေက်ာင္းသားမ်ား စာေမးပြဲ ေျဖဆိုၾကသည္။",
    "ျမန္မာႏိုင္ငံသည္ အေရွ႕ေတာင္အာရွတြင္ တည္ရွိသည္။",
    "ေက်းဇူးတင္ပါတယ္",
]

# Mon typed in a legacy font that follows Zawgyi's storage order. Not Unicode
# either, so the model has never seen it and nothing downstream can read it.
LEGACY_MON = "ဘာတန္ေဒါဝ္ေကာန္ဂကူမန္"

UNICODE_MON = [
    "ဘာသာမန်ကို ဂၠာဲကေတ် ကွာန်ဗၟာ",
    "ပ္ဍဲသၞာံ ၁၉၉၀",
    # U+103A before U+1031. Ill-formed by the letter of Unicode, written this
    # way thousands of times in the Mon corpus.
    "ယုတ္တိစှ်ေစှ်ေ ပြုပြာ",
    # The same, next to a Burmese word typed with Zawgyi's dot below (U+1095),
    # which counts as supporting evidence. Two signals if U+103A were not a base.
    "ဟွံစှ်ေအာ ပ္ဍဲဍုၚ် (ဆက်စပ်ဖွဲ႕စည်း)",
    # U+103A U+1039 before a non-letter: how Mon writes a kinzi-like final.
    "အၚ်္ဂလိက်၊ သချာၚ်္၊ ဓာတုဗေဒ",
    "ညံင်္ ပုန် ကဵုရာင်္ ဆာဲ",
    # U+1039 before a vowel sign or medial: OCR damage, twice in one line.
    "ကောန်ဍ္ာ်ကေုာံ ပစ္ွပွန်",
    # One U+1031 typed before its consonant and nothing else wrong. The gate
    # needs more than a lone U+1031, or Mon with one typo would be refused.
    "ကျာ်တြုဲေဂါတမ ကဵုဏာ ဓါတ်သာ်",
    # One stray U+1039 where the asat belongs: a typo, not an encoding.
    "ဟိုတ်မစွံပတန်လဝ္ ဗွဲမလအ်ဂှ်ရ",
]

BURMESE = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
SHAN = "ၵၸၺၼႁလိၵ်ႈတႆး ၵႂၢမ်းတႆး ၼႆႉပဵၼ်ၽႃႇသႃႇ"
ENGLISH = "The Mon language is spoken by about a million people."


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def test_the_reported_sample_is_refused_rather_than_called_mon(detector):
    result = detector.predict(REPORTED)
    assert result == ("unknown", 0.0, False, "zawgyi")


@pytest.mark.parametrize("text", [*ZAWGYI_BURMESE, LEGACY_MON])
def test_zawgyi_order_text_is_refused(detector, text):
    result = detector.predict(text)
    assert result.label == "unknown", f"{text!r} came back {result}"
    assert not result.reliable
    assert result.basis == "zawgyi"


@pytest.mark.parametrize("text", UNICODE_MON)
def test_unicode_mon_is_untouched(detector, text):
    """Measured on 211,685 Mon corpus lines: the gate flags none that are
    well-formed Unicode. These are the patterns that made looser rules fail."""
    assert not looks_like_zawgyi(text)
    result = detector.predict(text)
    assert result.basis != "zawgyi"
    assert result.label == "mnw"


@pytest.mark.parametrize("text", [BURMESE, ENGLISH])
def test_unicode_burmese_and_english_are_untouched(detector, text):
    assert not looks_like_zawgyi(text)
    assert detector.predict(text).basis != "zawgyi"


def test_shan_is_still_refused_as_another_language(detector):
    """Shan Unicode is full of U+1061-U+1097, which the gate counts as
    supporting evidence. It must not fire without a storage-order signal, or
    Shan would be reported under the wrong reason."""
    assert not looks_like_zawgyi(SHAN)
    assert detector.predict(SHAN).basis == "other-myanmar-script"


def test_codepoints_alone_never_prove_zawgyi():
    """U+1061-U+1097 are Shan, Karen and others in Unicode. Without a
    storage-order signal they are not Zawgyi evidence, however many there are."""
    assert not looks_like_zawgyi("ႏႏႏႏ ၿၿၿ ႕႕႕")


def test_a_lone_e_first_is_not_enough_even_when_repeated():
    """Unicode Mon typed with U+1031 first and nothing else wrong exists in the
    corpus, sometimes several times in one line. The gate needs a signal of
    another kind."""
    assert not looks_like_zawgyi("ကျာ် ေဂါ ေလာန် ေသၞာဝ်")
    assert looks_like_zawgyi("ကျာ် ေဂါ ေလာန္")

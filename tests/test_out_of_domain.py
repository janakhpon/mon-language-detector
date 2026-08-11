"""A three-class model has to answer, so something else has to be able to refuse.

fastText was trained on `eng`, `mnw` and `mya`. Shan is written in the Myanmar
script and is orthographically nearest to Mon, so it has nowhere else to land —
and it lands hard. Measured on the shipped `.ftz`, **after** the Mon-exclusive
regex was corrected so the character rule was no longer implicated:

    "ၵၸၺၼႁလိၵ်ႈတႆး ၵႂၢမ်းတႆး ၼႆႉပဵၼ်ၽႃႇသႃႇ"    -> mnw, 1.0000, reliable=True
    "မိူင်းတႆး ပဵၼ်မိူင်းၼိုင်ႈ ၼႂ်းမိူင်းႁူမ်ႈတုမ်"   -> mnw, 0.9998, reliable=True

For the stated use — filtering a scraped corpus — that is Shan entering a Mon
dataset marked reliable. No retrain fixes it: a fourth class needs Shan data
nobody has, and the same argument covers Khamti, Karen, Aiton and Palaung.

So the guard is characters, not probability. The 124 codepoints exclusive to
those languages are categorical evidence that none of the three classes is
right, and `unknown` is the only honest output.
"""

from __future__ import annotations

import unicodedata

import pytest

from mon_language_detector.detector import (
    MON_EXCLUSIVE_CODEPOINTS,
    OTHER_MYANMAR_LANGUAGE_CODEPOINTS,
    LanguageDetector,
)

SHAN = "ၵၸၺၼႁလိၵ်ႈတႆး ၵႂၢမ်းတႆး ၼႆႉပဵၼ်ၽႃႇသႃႇ"
SHAN_WIKI = "မိူင်းတႆး ပဵၼ်မိူင်းၼိုင်ႈ ၼႂ်းမိူင်းႁူမ်ႈတုမ်"
KHAMTI = "ꩠꩡꩢꩣꩤꩥꩦ ꩧꩨꩩꩪ ꩫꩬꩭ"
MON = "ဘာသာမန်ကို ဂၠာဲကေတ် ကွာန်ဗၟာ"
BURMESE = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။"
ENGLISH = "The Mon language is spoken by about a million people."


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def test_the_two_character_sets_are_disjoint():
    """A codepoint that is both Mon evidence and not-Mon evidence would make the
    guard's outcome depend on which branch ran first."""
    assert not (MON_EXCLUSIVE_CODEPOINTS & OTHER_MYANMAR_LANGUAGE_CODEPOINTS)


def test_the_guard_covers_the_scripts_the_old_regex_called_mon():
    """The 38 wrongly-claimed codepoints are exactly the ones worth guarding on —
    the defect and its fix are the same list read with the opposite sign."""
    formerly_claimed_as_mon = {
        *range(0x106E, 0x1071),  # Eastern Pwo Karen
        *range(0x1075, 0x107D),  # Shan
        0x1085,
        0x1086,  # Shan
        *range(0x109A, 0x109E),  # Khamti, Aiton
        *range(0xAA60, 0xAA7C),  # Khamti, Aiton, Pao Karen
    }
    uncovered = formerly_claimed_as_mon - OTHER_MYANMAR_LANGUAGE_CODEPOINTS
    assert not uncovered, (
        "these were matched as Mon and are not covered by the guard either: "
        + ", ".join(f"U+{cp:04X} {unicodedata.name(chr(cp))}" for cp in sorted(uncovered))
    )


@pytest.mark.parametrize(
    ("name", "text"), [("Shan", SHAN), ("Shan wiki", SHAN_WIKI), ("Khamti", KHAMTI)]
)
def test_other_myanmar_languages_are_refused_rather_than_called_mon(detector, name, text):
    result = detector.predict(text)
    assert result.label == "unknown", f"{name} came back as {result.label}"
    assert not result.reliable
    assert result.basis == "other-myanmar-script"


@pytest.mark.parametrize(
    ("name", "text"), [("Mon", MON), ("Burmese", BURMESE), ("English", ENGLISH)]
)
def test_the_three_real_classes_are_untouched(detector, name, text):
    """The guard must cost nothing on the languages the detector is for. Measured
    on the corpora it fires on 0.051% of Mon lines and 0% of Burmese."""
    result = detector.predict(text)
    assert result.basis != "other-myanmar-script", f"{name} was refused by the guard"


def test_mon_evidence_outweighs_a_borrowed_shan_word(detector):
    """The rule is 'other-language characters AND no Mon character', not just the
    first. A Mon sentence quoting a Shan word keeps its label — otherwise the
    guard would throw away real Mon lines to catch Shan ones."""
    result = detector.predict(MON + " ၵၸၺ")
    assert result.basis != "other-myanmar-script"
    assert result.label == "mnw"


# ---------------------------------------------------------------------------
# basis — audit finding H1
# ---------------------------------------------------------------------------


def test_a_posterior_and_a_constant_are_distinguishable(detector):
    """H1: `confidence` was a fastText probability on one path and a hand-chosen
    0.95 or 0.85 on another, with nothing to tell them apart. A caller
    thresholding at 0.9 was selecting for branch, not for certainty."""
    assert detector.predict(ENGLISH).basis == "posterior"
    # Under five characters with a Mon-exclusive character: the 0.95 literal.
    short_mon = detector.predict("ၚ")
    assert short_mon.confidence == 0.95
    assert short_mon.basis == "mon-exclusive"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("1234567890", "no-script"),
        ("!!! ,,, ???", "no-script"),
        ("ab", "too-short"),
        # Four characters: inside the `len < 5` window the ambiguous branch
        # covers. "မြန်မာ" is six and reaches the model, which is the point of
        # the window — the threshold is the behaviour, so the test states a
        # length rather than "some Myanmar text".
        ("မြန်", "ambiguous-myanmar"),
    ],
)
def test_every_early_return_names_its_reason(detector, text, expected):
    """`unknown` had four distinct causes and one indistinguishable output, so a
    caller could not tell "too short to judge" from "not a language at all"."""
    assert detector.predict(text).basis == expected


def test_the_legacy_three_field_shape_still_works(detector):
    """`basis` is appended with a default, so nothing that reads the first three
    fields has to change."""
    result = detector.predict(ENGLISH)
    label, confidence, reliable, _ = result
    assert (label, confidence, reliable) == (result.label, result.confidence, result.reliable)

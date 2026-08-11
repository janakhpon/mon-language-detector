"""What counts as a Mon-exclusive character, and why it is not a range list.

The detector treats a Mon-exclusive character as a **hard signal**: it overrides
the model's label, and it sets `reliable=True` unconditionally at any length. So
the membership of this set is load-bearing in a way a posterior is not — a wrong
member is a confident wrong answer, and a missing member is a hard signal thrown
away.

The original set was six hand-written codepoint ranges, commented "Mon medials /
Mon finals / Mon vowels / Mon-specific signs / Mon asat / Mon Extensions block".
**Of the 45 codepoints they matched, 7 were Mon.** The other 38 were Eastern Pwo
Karen, Shan, Khamti, Aiton and Pao Karen — and three genuinely Mon-exclusive
characters were missing entirely.

Measured 2026-08-11 on `mon_OCR/data/raw/corpus` (4,792,030 Mon characters,
552,394 Burmese):

    U+1028 MYANMAR LETTER MON E                     360 in Mon, 0 in Burmese
    U+1033 MYANMAR VOWEL SIGN MON II             11,942 in Mon, 0 in Burmese  <- was missed
    U+1034 MYANMAR VOWEL SIGN MON O              15,348 in Mon, 0 in Burmese  <- was missed
    U+105A..U+1060 (the seven it did have)     161,981 in Mon, 0 in Burmese

At line level, over 306,564 Mon lines: the old set fired on 36.62%, the correct
one on 39.99%. **10,395 Mon lines carried a hard signal the detector could not
see**, and 66 fired only on Shan characters inside the Mon shard.
"""

from __future__ import annotations

import unicodedata

import pytest

from mon_language_detector.detector import MON_EXCLUSIVE_CODEPOINTS, LanguageDetector

# Every Myanmar-script block, so the derivation below cannot miss a codepoint by
# looking in the wrong place: Myanmar, Myanmar Extended-A, -B and -C.
_MYANMAR_BLOCKS = (
    range(0x1000, 0x10A0),
    range(0xA9E0, 0xAA00),
    range(0xAA60, 0xAA80),
    range(0x116D0, 0x11700),
)


def _codepoints_named_mon() -> set[int]:
    """Myanmar codepoints whose Unicode name carries MON as a whole word.

    `in name` would be wrong: it matches nothing here today, but "MON" is a
    substring waiting to happen. Splitting on words is the rule that says what
    is meant.
    """
    found = set()
    for block in _MYANMAR_BLOCKS:
        for cp in block:
            try:
                name = unicodedata.name(chr(cp))
            except ValueError:
                continue
            if "MON" in name.split():
                found.add(cp)
    return found


def test_the_set_is_exactly_what_unicode_calls_mon():
    """The set is explicit in the source for speed and greppability; this is the
    derivation it has to keep agreeing with. A range list cannot be checked this
    way, which is how 38 Shan and Karen codepoints survived in it."""
    assert MON_EXCLUSIVE_CODEPOINTS == _codepoints_named_mon()


@pytest.mark.parametrize(
    ("cp", "script"),
    [
        (0x106E, "Eastern Pwo Karen"),
        (0x1070, "Eastern Pwo Karen"),
        (0x1075, "Shan"),
        (0x107C, "Shan"),
        (0x1085, "Shan"),
        (0x1086, "Shan"),
        (0x109A, "Khamti"),
        (0x109D, "Aiton"),
        (0xAA60, "Khamti"),
        (0xAA7A, "Aiton"),
        (0xAA7B, "Pao Karen"),
    ],
)
def test_other_myanmar_script_languages_are_not_mon(cp, script):
    """Each of these was matched as Mon-exclusive, so text in these languages was
    labelled `mnw` with `reliable=True` and the model's own answer discarded."""
    assert cp not in MON_EXCLUSIVE_CODEPOINTS, (
        f"U+{cp:04X} ({unicodedata.name(chr(cp))}) is {script}, not Mon"
    )


@pytest.mark.parametrize("cp", [0x1028, 0x1033, 0x1034])
def test_the_three_that_were_missing_are_present(cp):
    """Absent from the old ranges, and not rare: U+1033 and U+1034 alone account
    for 27,290 occurrences in a 4.8M-character Mon corpus."""
    assert cp in MON_EXCLUSIVE_CODEPOINTS


def test_no_burmese_letter_is_ever_mon_exclusive():
    """The set's whole purpose is to break the Mon/Burmese tie, so anything
    Burmese in it defeats the point. U+103A ASAT is the specific trap: a sibling
    repository lists it under Mon *font capability*, which is a different
    question — every Burmese text uses it."""
    for cp in (0x1000, 0x1001, 0x1039, 0x103A, 0x103B, 0x103C, 0x1031, 0x102C):
        assert cp not in MON_EXCLUSIVE_CODEPOINTS, (
            f"U+{cp:04X} ({unicodedata.name(chr(cp))}) occurs throughout Burmese"
        )


# ---------------------------------------------------------------------------
# The behaviour the set controls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def test_shan_text_is_not_confidently_labelled_mon(detector):
    """The live defect. Shan sentence, no Mon character in it.

    Before: `mnw`, confidence 1.000, reliable True — the hard-signal branch fired
    on U+1075..U+107C and overrode everything.
    """
    shan = "ၵၸၺၼႁလိၵ်ႈတႆး ၵႂၢမ်းတႆး ၼႆႉပဵၼ်ၽႃႇသႃႇ"
    result = detector.predict(shan)
    assert not (result.label == "mnw" and result.reliable), (
        f"Shan text came back {result.label} reliable={result.reliable}; the "
        "Mon-exclusive branch is matching Shan characters again"
    )


def test_a_mon_only_signal_still_wins(detector):
    """The fix must not cost the true positives it exists for. U+105A MON NGA
    appears 95,104 times in the Mon corpus and never in the Burmese one."""
    result = detector.predict("ဘာသာမန်ကို ဂၠာဲကေတ်ၚ")
    assert result.label == "mnw"
    assert result.reliable

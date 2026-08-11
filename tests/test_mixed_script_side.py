"""Which Myanmar language is in a mixed-script line, and how the code guessed.

The mixed branch read:

    label = "mnw-eng" if (has_mon or lang == "mnw") else "mya-eng"

`lang` is the model's verdict on the **whole** string. On a sentence that is 82%
Latin, that verdict is `eng` — correctly — so a Mon fragment with no
Mon-exclusive character fell through to the `else` and came back **`mya-eng`,
asserted with no evidence at all.** Nothing in the input said Burmese.

It was invisible until 2026-08-11 because the previous model returned `mnw` at
confidence 1.000 for whole English sentences: 6.5% of its Mon training class was
English reference lines from Mon Wikipedia. `lang == "mnw"` was therefore true
for almost any Mon-adjacent text and the `else` was rarely reached. Cleaning the
training data exposed the defect — the retrained model says `eng` for the
sentence and `mnw` for its Myanmar substring, both right.

The fix asks the question the branch actually needs answered: run the classifier
on the Myanmar-script characters alone.
"""

from __future__ import annotations

import pytest

from mon_language_detector.detector import LanguageDetector


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def test_a_mon_fragment_in_an_english_sentence_is_mon(detector):
    """The regression that surfaced this. `ဘာသာမန်` carries no Mon-exclusive
    character, so the hard signal cannot help and the substring has to be
    classified on its own."""
    result = detector.predict("ဘာသာမန် is spoken in Myanmar and Thailand today")
    assert result.label == "mnw-eng"


def test_a_burmese_fragment_in_an_english_sentence_is_burmese(detector):
    """The other side, which the old code got right only by defaulting to it."""
    result = detector.predict("မြန်မာနိုင်ငံသည် is a country in Southeast Asia today")
    assert result.label == "mya-eng"


def test_a_mon_exclusive_character_still_short_circuits(detector):
    """The hard signal is cheaper and stronger than a second prediction, so it
    stays first — `ဂၠာဲ` carries U+1060."""
    result = detector.predict("ဂၠာဲကေတ် is the Mon word used here in this sentence")
    assert result.label == "mnw-eng"
    assert result.basis == "mon-exclusive"


def test_the_myanmar_side_is_decided_from_myanmar_characters_only(detector):
    """Not from the whole string. The two verdicts genuinely differ here: the
    sentence is `eng` at 1.000 and its Myanmar substring is `mnw` at 1.000."""
    sentence = "ဘာသာမန် is spoken in Myanmar and Thailand today"
    (whole,), _ = detector.model.predict(sentence, k=1)
    fragment = "".join(c for c in sentence if "က" <= c <= "႟")
    (part,), _ = detector.model.predict(fragment, k=1)
    assert whole == "__label__eng", "premise changed; the sentence no longer reads as English"
    assert part == "__label__mnw", "premise changed; the fragment no longer reads as Mon"
    assert detector.predict(sentence).label == "mnw-eng"


def test_an_unjudgeable_fragment_is_not_vouched_for(detector):
    """A ten-character Myanmar fragment cannot separate Mon from Burmese.

    The label still has to be one of the two, so `reliable` carries the doubt —
    the alternative is inventing a `mnw-mya-eng` label for a case the caller can
    already detect.

    The consistency this enforces: the same fragment alone returns
    `reliable=False`, because ten characters is under
    `MIN_UNAMBIGUOUS_MYANMAR_LEN`. Prepending an English word cannot raise
    confidence about which Myanmar language follows, and until 2026-08-11 it did.
    """
    mixed = detector.predict("Computer သွက်ဂွံစကာ")
    alone = detector.predict("သွက်ဂွံစကာ")
    assert mixed.label == "mnw-eng"
    assert alone.reliable is False
    assert mixed.reliable is False, "an English prefix made the Myanmar side vouchable"


def test_a_long_enough_fragment_is_vouched_for(detector):
    """The guard is the fragment length, not the presence of mixing — otherwise
    every mixed-script line would be unreliable and the label would be useless."""
    result = detector.predict("Computer မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။")
    assert result.label.endswith("-eng")
    assert result.reliable

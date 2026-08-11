"""Why the Myanmar floor is 30, expressed as behaviour rather than a constant.

Mon against Burmese is where every remaining error lives, and the measurement
that settled it is that the errors are a **length** problem, not a data problem.
Over the 19,988 Myanmar-script lines of the 2026-08-11 held-out split:

    length   n        error rate   share of all errors
    11-20    9,154    12.33%       74.2%
    21-40    4,657     7.67%       23.5%
    41-80    2,395     0.63%        1.0%
    81+      3,782     0.53%        1.3%

97.7% of the errors sit in lines of 40 characters or fewer. And of the 8,855
lines carrying a Mon-exclusive character, zero were misclassified, against 13.66%
of the 11,133 without one.

So the floor governs exactly the band where the classifier is weak and the hard
signal is absent. `tests/test_length_floor.py` probes the boundary itself; this
file pins the reasoning that put the boundary there, so raising or lowering it
has to argue with the evidence.
"""

from __future__ import annotations

import pytest

from mon_language_detector.detector import LanguageDetector
from mon_language_detector.utils import MIN_RELIABLE_LEN, MIN_UNAMBIGUOUS_MYANMAR_LEN


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def test_the_floor_covers_the_band_where_the_errors_are():
    """74.2% of errors are under 20 characters and 23.5% more are 21-40. A floor
    of 20 left the second band vouched for; 30 covers most of it."""
    assert MIN_UNAMBIGUOUS_MYANMAR_LEN >= 30, (
        "at 20 the 21-40 band was marked reliable at a 7.67% error rate"
    )
    assert MIN_UNAMBIGUOUS_MYANMAR_LEN > MIN_RELIABLE_LEN, (
        "the Myanmar floor is stricter than the general one, because Mon and "
        "Burmese share the script and English does not share it with either"
    )


def test_a_mon_exclusive_character_is_still_exempt(detector):
    """The floor is about absent evidence, not about length as such. Zero of the
    8,855 lines carrying a Mon-exclusive character were misclassified, so gating
    them on length would cost coverage and buy nothing."""
    short_mon = "ဂၠာဲကေတ်"  # under the floor, carries U+1060
    result = detector.predict(short_mon)
    assert len(short_mon) < MIN_UNAMBIGUOUS_MYANMAR_LEN
    assert result.label == "mnw"
    assert result.reliable, "a hard signal is not subject to the length floor"


def test_a_short_myanmar_line_without_the_signal_is_not_vouched_for(detector):
    """The case the floor exists for: Myanmar script, no Mon-exclusive character,
    inside the band where one line in eight is wrong."""
    result = detector.predict("မြန်မာနိုင်ငံသည် အရှေ့")
    assert len(result.label) > 0
    assert not result.reliable


def test_a_long_myanmar_line_is_vouched_for(detector):
    """Above the floor the measured error rate is under 1%, so the guard has to
    let these through or the flag is useless."""
    result = detector.predict(
        "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံတစ်ခုဖြစ်ပြီး လူဦးရေ ငါးဆယ်သန်းကျော် ရှိပါသည်။"
    )
    assert result.reliable

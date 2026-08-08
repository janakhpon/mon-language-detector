"""Training and serving must agree on the shortest text this model handles.

They did not. `pipeline._extract_lines` kept lines of `len > 10`, so eleven
characters and up, while `detector.predict` marked a result reliable at
`len >= 10`. A ten-character input was therefore vouched for at a length the
model had never seen a single training example of -- an off-by-one, but on the
boundary where the classifier is weakest and the guard exists precisely to
protect.

Both now read `MIN_RELIABLE_LEN`. These tests pin the two ends to the same
constant so the pair cannot drift apart again silently.
"""

from pathlib import Path

from mon_language_detector.pipeline import _extract_lines
from mon_language_detector.utils import MIN_RELIABLE_LEN, MIN_UNAMBIGUOUS_MYANMAR_LEN
from test_detector import get_test_detector


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_training_filter_is_inclusive_at_the_floor(tmp_path):
    """Exactly MIN_RELIABLE_LEN characters is kept, one fewer is dropped.

    Inclusivity is the whole fix. `>` and `>=` differ by one line length, and
    that one length is the one the detector was willing to answer for.
    """
    at_floor = "a" * MIN_RELIABLE_LEN
    below = "a" * (MIN_RELIABLE_LEN - 1)

    kept = _extract_lines(_write(tmp_path, at_floor, below))

    assert at_floor in kept, "a line at the floor must be trained on"
    assert below not in kept, "a line below the floor must not be"


def test_the_detector_does_not_vouch_below_the_training_floor(tmp_path):
    """The boundary from the serving side, using Latin text.

    Latin avoids MIN_UNAMBIGUOUS_MYANMAR_LEN, which is a separate and stricter
    guard -- Myanmar text would fail this for the other reason and prove
    nothing about the floor.
    """
    detector = get_test_detector()

    below_text = "This is " + "x" * (MIN_RELIABLE_LEN - 9)
    at_floor_text = "This is " + "x" * (MIN_RELIABLE_LEN - 8)
    assert len(below_text) == MIN_RELIABLE_LEN - 1
    assert len(at_floor_text) == MIN_RELIABLE_LEN

    assert detector.predict(below_text).reliable is False
    assert detector.predict(at_floor_text).reliable is True


def test_the_two_floors_agree_when_probed(tmp_path):
    """Measure both floors instead of reading either one.

    The tests above build their strings from MIN_RELIABLE_LEN, so they pin the
    comparison but not the number: set the constant wrong and they still pass
    together. This one asks each side, in its own terms, for the shortest input
    it accepts, and compares the two answers. It is the test that actually
    describes the defect -- a gap between the two -- rather than one side of it.

    "s" repeated is deliberate: DummyModel answers eng for anything containing
    T, h, i or s, and an all-Latin string stays clear of the separate and
    stricter Myanmar guard.
    """
    detector = get_test_detector()
    probe = range(1, 40)

    guard_floor = min(n for n in probe if detector.predict("s" * n).reliable)
    filter_floor = min(n for n in probe if ("s" * n) in _extract_lines(_write(tmp_path, "s" * n)))

    assert guard_floor == filter_floor, (
        f"the detector vouches from {guard_floor} characters but the pipeline trains "
        f"from {filter_floor}. Whichever moved, the other has to follow."
    )
    assert guard_floor == MIN_RELIABLE_LEN


def test_the_pipeline_default_is_the_shared_constant():
    """A copy that happens to match is how the two drifted the first time."""
    import inspect

    assert inspect.signature(_extract_lines).parameters["min_len"].default == MIN_RELIABLE_LEN


def test_a_mon_exclusive_character_is_reliable_below_the_floor():
    """The floor is about the classifier's evidence, not about length as such.

    A Mon-exclusive character identifies the language outright, so the guard is
    deliberately bypassed. Pinned because tightening the floor by one character
    must not quietly take this path with it.
    """
    assert len("ၝောအ်") < MIN_RELIABLE_LEN

    result = get_test_detector().predict("ၝောအ်")
    assert result.label == "mnw"
    assert result.reliable is True


def test_myanmar_carries_the_stricter_floor():
    """Myanmar-only text needs MIN_UNAMBIGUOUS_MYANMAR_LEN, not MIN_RELIABLE_LEN.

    Mon and Burmese share nearly every character, so length past the general
    floor still is not enough to separate them without a hard signal.
    """
    assert MIN_UNAMBIGUOUS_MYANMAR_LEN > MIN_RELIABLE_LEN

    detector = get_test_detector()
    between = "မ" * ((MIN_RELIABLE_LEN + MIN_UNAMBIGUOUS_MYANMAR_LEN) // 2)

    assert MIN_RELIABLE_LEN <= len(between) < MIN_UNAMBIGUOUS_MYANMAR_LEN
    assert detector.predict(between).reliable is False

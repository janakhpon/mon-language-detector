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


def test_the_two_ends_read_one_constant():
    """The default is the constant itself, not a copy that happens to match.

    A literal in either place is how the two drifted the first time.
    """
    import inspect

    from mon_language_detector import detector as detector_module

    assert (
        inspect.signature(_extract_lines).parameters["min_len"].default == MIN_RELIABLE_LEN
    ), "the pipeline default has been unpinned from the shared constant"

    source = inspect.getsource(detector_module.LanguageDetector.predict)
    assert "MIN_RELIABLE_LEN" in source, "the reliability guard no longer reads the constant"
    assert "len(cleaned) >= 10" not in source, "the literal floor is back"


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

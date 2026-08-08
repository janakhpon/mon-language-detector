import pytest
import pathlib
from mon_language_detector.detector import LanguageDetector

class DummyModel:
    def predict(self, text, k=1):
        if any(c in text for c in "This"): return (["__label__eng"], [0.99])
        return (["__label__mnw"], [0.99])

def get_test_detector():
    # Both patches must be undone. `fasttext.load_model` used to be replaced and
    # never restored, so every later test in the session got DummyModel instead
    # of the real artifact -- which silently broke tests/test_detector_real_model.py
    # when it was added: 18 passed alone, 2 failed in the full suite, because a
    # stub that always answers 0.99 was standing in for the classifier.
    import fasttext
    import pathlib

    original_load = fasttext.load_model
    original_exists = pathlib.Path.exists
    fasttext.load_model = lambda x: DummyModel()
    pathlib.Path.exists = lambda x: True
    try:
        return LanguageDetector(pathlib.Path("dummy.ftz"))
    finally:
        pathlib.Path.exists = original_exists
        fasttext.load_model = original_load

def test_basic_detection():
    d = get_test_detector()
    assert d.predict("This is English").label == "eng"
    # ၝ is Mon-exclusive
    assert d.predict("ၝောအ်ကိုတ်").label == "mnw"

def test_mixed_detection():
    d = get_test_detector()
    # Mixed English and Mon
    res = d.predict("Computer သွက်ဂွံစကာ")
    assert res.label == "mnw-eng"
    assert res.reliable is True

def test_unreliable_short():
    d = get_test_detector()
    res = d.predict("အ")
    assert res.reliable is False

def test_empty():
    d = get_test_detector()
    assert d.predict("").label == "unknown"

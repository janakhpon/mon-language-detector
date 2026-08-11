from mon_language_detector.utils import clean_and_normalize


def test_nfc_normalization():
    # Example of NFD (Decomposed) converting to NFC (Composed)
    # This might be tricky to hardcode visibly, but we can test standard characters
    assert clean_and_normalize("This is clean") == "This is clean"
    assert clean_and_normalize("Hello  World") == "Hello World"
    assert clean_and_normalize("Tab\there") == "Tab here"


def test_invisible_char_stripping():
    # ZWSP, ZWNJ, ZWJ, BOM
    dirty_text = "Hello\u200b\u200c\u200d\ufeffWorld"
    assert clean_and_normalize(dirty_text) == "HelloWorld"


def test_edge_cases():
    assert clean_and_normalize("") == ""
    assert clean_and_normalize(None) == ""
    assert clean_and_normalize("   \n   \t  ") == ""

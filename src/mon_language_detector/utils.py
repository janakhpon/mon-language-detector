import logging
import unicodedata
from importlib import resources
from pathlib import Path

# Provide a root project path based on the location of this file
# This assumes the file is in src/mon_language_detector/utils.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_FILENAME = "langid_mon_mya_eng_compressed.ftz"


def default_model_path() -> Path:
    """Path to the bundled fastText model.

    The model used to live at the repository root, resolved by walking up three
    parents from __file__. That worked only because `uv sync` installs this
    project editable, so the walk landed back in the checkout. A real install put
    PROJECT_ROOT inside site-packages and the constructor raised FileNotFoundError
    -- and `uv build` produced a wheel with no model in it at all, because the
    file sat outside the package directory.

    It now lives inside the package and is resolved through importlib.resources,
    so it ships in the wheel and is found wherever the package is installed.
    """
    return Path(str(resources.files("mon_language_detector") / "data" / MODEL_FILENAME))

def get_logger(name: str) -> logging.Logger:
    """Return a logger with structured SE Brain compliant formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

def clean_and_normalize(text: str) -> str:
    """
    NFC normalize and clean a string.
    Removes zero-width characters and normalizes whitespaces.
    Matches training and evaluation pipeline requirements.
    """
    if not isinstance(text, str) or not text:
        return ""
        
    text = text.strip()
    try:
        # Standard SE Brain constraints for Mon/Burmese scripts
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u200b', '') # ZWSP
        text = text.replace('\u200c', '') # ZWNJ
        text = text.replace('\u200d', '') # ZWJ
        text = text.replace('\ufeff', '') # BOM
        
        # Clean whitespaces
        text = text.replace('\n', ' ').replace('\t', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text.strip()
    except Exception as e:
        logger.warning(f"Normalization failed for text: {e}")
        return ""

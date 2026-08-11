import logging
import unicodedata
from importlib import resources
from pathlib import Path

# Provide a root project path based on the location of this file
# This assumes the file is in src/mon_language_detector/utils.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_FILENAME = "langid_mon_mya_eng_compressed.ftz"

# The shortest text the detector is willing to vouch for, and the shortest line
# the training pipeline keeps. One constant, because they are one decision.
#
# They were two: training kept `len > 10`, so 11 characters and up, while the
# reliability guard accepted `len >= 10`. Text of exactly ten characters was
# therefore marked reliable at a length the model had never been trained on.
# Serving now matches training rather than the reverse -- the model's evidence
# is what sets the floor.
MIN_RELIABLE_LEN = 11

# Myanmar script alone does not separate Mon from Burmese: the two share almost
# every character, and Mon-exclusive characters are what break the tie. Below
# this length a Myanmar-only string with no Mon-exclusive character carries too
# little signal to call, whatever the posterior says.
#
# **20 -> 30, measured 2026-08-11.** Mon against Burmese is where every remaining
# error lives, and it is a length problem rather than a data problem. Over the
# 19,988 Myanmar-script lines of the held-out split:
#
#     length   n        error rate   share of all errors
#     11-20    9,154    12.33%       74.2%
#     21-40    4,657     7.67%       23.5%
#     41-80    2,395     0.63%        1.0%
#     81+      3,782     0.53%        1.3%
#
# **97.7% of the errors are in lines of 40 characters or fewer**, and above 40
# the rate collapses under 1%. The other half of the same measurement: of 8,855
# lines carrying a Mon-exclusive character, **zero** were misclassified, against
# 13.66% of the 11,133 without one. The hard signal is doing its job; the gap is
# everything it cannot reach.
#
# Raising the floor trades coverage for correctness, and 30 is the knee:
#
#     threshold   coverage   accuracy where reliable
#     20 (was)      83.2%    0.9888
#     30            77.4%    0.9980
#     40            75.5%    0.9990
#     60            73.2%    0.9992
#
# 20 -> 30 costs 5.8 points of coverage and cuts the error rate 5.6x. 30 -> 40
# costs another 1.9 for a tenth of a point. For the documented use — filtering a
# corpus, where candidate lines are plentiful and a contaminated one is
# expensive — that first trade is worth making and the second is not.
#
# **This threshold was selected on the same split it is scored on**, because
# there is no held-out test set (AUDIT-2026-08-08, Medium). The reported 0.9980
# is therefore optimistic by an unknown margin. One threshold chosen off a smooth
# monotone curve is close to the mildest form that bias takes, but it is not zero
# and the number should be read with that attached.
MIN_UNAMBIGUOUS_MYANMAR_LEN = 30

# The share of a line's script-bearing characters that must belong to a class's
# own script before that line may train or evaluate the class.
#
# One constant, because it is one decision made in two places: `detector.py`
# labels a text `eng` when its Latin share exceeds this, and the pipeline now
# refuses to put a line in a single-language class unless that class's script
# clears the same bar. Without the second half the first is punished for being
# right — the Mon class held Mon-Wikipedia reference lines like
# "Heinz, L.C. (6 March 1962)." labelled `mnw`, so the detector called them
# English and the evaluation scored it wrong.
#
# Measured 2026-08-11 on the deduplicated corpus: at 0.85 the gate keeps 89.6% of
# Mon lines, 99.2% of Burmese and 99.6% of English. The 10.4% it drops from Mon
# are lines where Myanmar is not the majority script; they are mixed or
# mislabelled, and a single-language class is the wrong home for both. Code
# switching is served by the `mnw-eng` / `mya-eng` labels and by the synthesised
# mixed samples, not by mislabelled corpus rows.
MIN_SCRIPT_DOMINANCE = 0.85


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
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\u200b", "")  # ZWSP
        text = text.replace("\u200c", "")  # ZWNJ
        text = text.replace("\u200d", "")  # ZWJ
        text = text.replace("\ufeff", "")  # BOM

        # Clean whitespaces
        text = text.replace("\n", " ").replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()
    except Exception as e:
        logger.warning(f"Normalization failed for text: {e}")
        return ""

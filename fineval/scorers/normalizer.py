"""
normalizer.py
Standalone normalization layer for FinEval scoring.
Converts model outputs and expected answers into comparable floats
before scoring. Handles percentages, decimals, currency, commas,
extra words, and yes/no answers.
"""
import re


def extract_number_from_text(text: str) -> tuple[float | None, bool]:
    """
    Extract a number from free-form text.
    Also returns whether the number was expressed as a percentage.
    Returns (value, is_percentage).
    """
    text = str(text).strip()

    # handle yes/no first
    lower = text.lower().strip().rstrip(".")
    if lower in ["yes", "no"]:
        return None, False

    # remove currency symbols and commas
    text = text.replace("$", "").replace("£", "").replace("€", "").replace(",", "")
    text = text.strip()

    # check for explicit percentage sign
    is_percentage = "%" in text
    text = text.replace("%", "")

    # handle billion
    if "billion" in text.lower():
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            return float(match.group()) * 1_000_000_000, False

    # handle million
    if "million" in text.lower():
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            return float(match.group()) * 1_000_000, False

    # extract first number from text
    # handles cases like "The answer is 53%" or "approximately 0.53"
    match = re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group()), is_percentage
        except ValueError:
            return None, False

    return None, False


def normalize_value(text: str) -> tuple[float | None, bool]:
    """
    Normalize a value to a float and flag if it was a percentage.
    Returns (normalized_float, was_percentage).
    """
    return extract_number_from_text(text)


def is_yes_no(text: str) -> bool:
    """Check if a string is a yes or no answer."""
    return str(text).lower().strip().rstrip(".") in ["yes", "no"]


def normalize_yes_no(text: str) -> str | None:
    """Normalize yes/no answer to lowercase string."""
    clean = str(text).lower().strip().rstrip(".")
    if clean in ["yes", "no"]:
        return clean
    return None


def values_are_close(
    pred: float,
    exp: float,
    tolerance: float = 0.01
) -> bool:
    """
    Check if two floats are close enough to be considered equal.
    Uses absolute tolerance for small numbers and relative
    tolerance for large numbers.
    For large numbers (>= 1000), allows a small relative difference
    so that 41932 counts as correct for 41932.20339.
    """
    if exp == 0:
        return abs(pred) <= tolerance
    # for large numbers use a relative tolerance of 0.1%
    if abs(exp) >= 1000:
        relative_diff = abs(pred - exp) / abs(exp)
        return relative_diff <= 0.001
    return abs(pred - exp) <= tolerance


def normalize_and_compare(
    predicted: str,
    expected: str,
    tolerance: float = 0.01
) -> tuple[float, bool]:
    """
    Core normalization and comparison logic.

    Handles:
    - Percentages vs decimals (53% == 0.53232 within tolerance)
    - Rounding tolerance (0.27743 == 0.2774)
    - Currency symbols ($1,000 == 1000)
    - Extra words ("The answer is 53%" == 0.53)
    - Negative numbers
    - Yes/No answers

    Does NOT treat bare integers as percentages:
    - 53 does NOT equal 0.53 unless explicitly marked as percentage
    - 53% DOES equal 0.53

    Returns (score, is_correct).
    """
    predicted = str(predicted).strip()
    expected = str(expected).strip()

    # handle yes/no
    if is_yes_no(expected):
        pred_yn = normalize_yes_no(predicted)
        exp_yn = normalize_yes_no(expected)
        if pred_yn is None or exp_yn is None:
            return 0.0, False
        is_correct = pred_yn == exp_yn
        return (1.0 if is_correct else 0.0), is_correct

    # extract numbers and percentage flags
    pred_value, pred_is_pct = normalize_value(predicted)
    exp_value, exp_is_pct = normalize_value(expected)

    if pred_value is None or exp_value is None:
        return 0.0, False

    # convert percentage to decimal if explicitly marked with %
    if pred_is_pct and not exp_is_pct:
        pred_as_decimal = pred_value / 100.0
        if values_are_close(pred_as_decimal, exp_value, tolerance):
            return 1.0, True

    if exp_is_pct and not pred_is_pct:
        exp_as_decimal = exp_value / 100.0
        if values_are_close(pred_value, exp_as_decimal, tolerance):
            return 1.0, True

# implicit percentage detection
    # if expected is a decimal between -1 and 1, and predicted looks like a percentage version of it, mark correct. handles models that answer 53.2 
    # when the expected is 0.53232.
    # uses a tighter tolerance to avoid crediting coincidental 100x matches.
    if not pred_is_pct and not exp_is_pct:
        if -1.0 <= exp_value <= 1.0 and abs(pred_value) > 1.0:
            pred_as_decimal = pred_value / 100.0
            relative_tolerance = max(abs(exp_value) * 0.02, 0.002)
            if abs(pred_as_decimal - exp_value) <= relative_tolerance:
                return 1.0, True

    # million scale mismatch detection
    # if the model answered in full units (3800000) but expected is in millions (3.8), or vice versa, check if dividing by 1 million matches.
    # handles "$3.8 million" being read as 3800000 when expected is 3.8
    if pred_value != 0 and exp_value != 0:
        if values_are_close(pred_value / 1_000_000, exp_value, tolerance):
            return 1.0, True
        if values_are_close(pred_value * 1_000_000, exp_value, tolerance):
            return 1.0, True

    # direct comparison
    is_correct = values_are_close(pred_value, exp_value, tolerance)

    # soft score - if marked correct by any rule, score is 1.0
    if is_correct:
        soft_score = 1.0
    else:
        error = abs(pred_value - exp_value)
        if tolerance > 0:
            soft_score = max(0.0, 1.0 - error / (10 * tolerance))
        else:
            soft_score = 0.0

    return soft_score, is_correct
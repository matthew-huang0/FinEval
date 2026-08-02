"""
NumericalScorer.
Uses the standalone normalizer to compare predicted
and expected answers before scoring.
"""

from fineval.scorers.normalizer import normalize_and_compare, is_yes_no


class NumericalScorer:
    """
    Scores numerical and yes/no QA responses.
    Delegates all normalization to normalizer.py.
    Returns (score, is_correct).
    """

    def score(
        self,
        predicted: str,
        expected: str,
        tolerance: float = 0.01
    ) -> tuple[float, bool]:
        return normalize_and_compare(predicted, expected, tolerance)



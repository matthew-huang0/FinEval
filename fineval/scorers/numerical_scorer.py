"""
NumericalScorer.
Uses the standalone normalizer to compare predicted
and expected answers before scoring.
"""
import sys

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


if __name__ == "__main__":
    scorer = NumericalScorer()

    test_cases = [
        # percentage vs decimal
        ("53%", "0.53232", True),
        ("53%", "0.53", True),
        ("0.53", "53%", True),
        # rounding tolerance
        ("0.2774", "0.27743", True),
        # currency and commas
        ("1000", "$1,000", True),
        # extra words
        ("The answer is 1,000.", "1000", True),
        # bare integer should NOT equal decimal
        ("53", "0.53", False),
        # yes/no
        ("Yes.", "yes", True),
        ("no", "yes", False),
        # standard numerical
        ("3.8", "3.8", True),
        ("206588", "206588.0", True),
        ("$1.5 million", "1500000", True),
        ("1,500,000", "1500000", True),
        ("1.5e6", "1500000", True),
        ("-6.9", "6.9", False),
        ("-2620", "-2620.0", True),
    ]

    print("NumericalScorer test cases:\n")
    all_passed = True
    for predicted, expected, should_be_correct in test_cases:
        score, is_correct = scorer.score(predicted, expected)
        passed = is_correct == should_be_correct
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"{status} predicted={predicted!r} expected={expected!r}")
        print(f"   score={score:.2f} is_correct={is_correct} "
              f"expected_correct={should_be_correct}")
        print()

    print("=" * 40)
    print("All tests passed." if all_passed else "Some tests FAILED.")
    print("=" * 40)
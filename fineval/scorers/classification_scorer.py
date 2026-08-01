"""
ClassificationScorer.
Compares a predicted label against the expected label.
Used for Financial PhraseBank sentiment classification.
"""


class ClassificationScorer:
    """
    Scores classification responses.
    Returns a score of 1.0 and True if the predicted label matches
    the expected label, 0.0 and False if it does not.
    """

    def score(self, predicted: str, expected: str) -> tuple[float, bool]:
        """
        Compare predicted label to expected label.
        Case insensitive — "Positive" and "positive" are treated the same.
        Returns (score, is_correct).
        """
        predicted_clean = predicted.lower().strip()
        expected_clean = expected.lower().strip()

        # check if the expected label appears anywhere in the prediction
        # this handles cases where the model says "The sentiment is positive"
        # instead of just "positive"
        is_correct = expected_clean in predicted_clean

        return (1.0 if is_correct else 0.0), is_correct


if __name__ == "__main__":
    scorer = ClassificationScorer()

    test_cases = [
        ("positive", "positive", True),
        ("Positive", "positive", True),
        ("The sentiment is positive", "positive", True),
        ("neutral", "positive", False),
        ("negative", "negative", True),
        ("Negative", "negative", True),
        ("neutral", "neutral", True),
    ]

    print("ClassificationScorer test cases:\n")
    for predicted, expected, expected_correct in test_cases:
        score, is_correct = scorer.score(predicted, expected)
        status = "PASS" if is_correct == expected_correct else "FAIL"
        print(f"{status} predicted={predicted!r} expected={expected!r}")
        print(f"   score={score:.2f} is_correct={is_correct}")
        print()
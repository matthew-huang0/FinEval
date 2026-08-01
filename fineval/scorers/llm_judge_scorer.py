"""
LLMJudgeScorer.
Uses a second AI model to grade the first AI model's answer.
Useful for open ended questions where programmatic scoring falls short.
Already defined in DataStructures.py as LLMJudgeScorer.
"""
import os
import sys

from openai import OpenAI


def get_judge_client(
    base_url: str = "https://integrate.api.nvidia.com/v1"
) -> OpenAI:
    """Set up the judge model client."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Set NVIDIA_API_KEY env var first")
    return OpenAI(base_url=base_url, api_key=api_key)


def build_judge_prompt(
    question: str,
    expected: str,
    predicted: str
) -> str:
    """
    Build the prompt sent to the judge model.
    Asks the judge to score the predicted answer against the expected answer.
    """
    return (
        f"You are an expert financial evaluator. Your job is to grade "
        f"whether a predicted answer is correct given the expected answer.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Predicted answer: {predicted}\n\n"
        f"Instructions:\n"
        f"- If the predicted answer is correct or means the same thing "
        f"as the expected answer, respond with: SCORE: 1.0\n"
        f"- If the predicted answer is partially correct, respond with "
        f"a score between 0 and 1, for example: SCORE: 0.5\n"
        f"- If the predicted answer is wrong, respond with: SCORE: 0.0\n"
        f"- Always start your response with SCORE: followed by the number.\n"
        f"- After the score you may add a brief one sentence explanation."
    )


def parse_judge_response(response: str) -> float:
    """
    Extract the score from the judge model's response.
    Looks for SCORE: followed by a number.
    Returns a float between 0.0 and 1.0.
    """
    import re
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", response)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0)
    return 0.0


class LLMJudgeScorer:
    """
    Scores open ended responses using a second AI model as judge.
    The judge model reads the question, expected answer, and predicted
    answer and returns a score between 0 and 1.
    """

    def __init__(
        self,
        judge_model: str = "meta/llama-3.3-70b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1"
    ):
        self.judge_model = judge_model
        self.client = get_judge_client(base_url)

    def score(
        self,
        question: str,
        expected: str,
        predicted: str
    ) -> tuple[float, bool]:
        """
        Score a predicted answer using the judge model.
        Returns (score, is_correct).
        is_correct is True if score >= 0.5.
        """
        prompt = build_judge_prompt(question, expected, predicted)

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw_response = response.choices[0].message.content.strip()
            score = parse_judge_response(raw_response)
            is_correct = score >= 0.5
            return score, is_correct

        except Exception as e:
            print(f"Judge scorer error: {e}")
            return 0.0, False


if __name__ == "__main__":
    print("LLMJudgeScorer test cases:\n")
    print("Note: these require a live API connection and will be slow.")
    print("Skipping live tests — judge scorer is ready to use.")
    print()
    print("Example usage:")
    print()
    print("from fineval.scorers.llm_judge_scorer import LLMJudgeScorer")
    print("judge = LLMJudgeScorer()")
    print("score, correct = judge.score(")
    print("    question='What caused the revenue decline?',")
    print("    expected='Lower sales volume',")
    print("    predicted='Sales dropped causing revenue to fall'")
    print(")")
    print("print(score, correct)")
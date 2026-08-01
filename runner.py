"""
FinEval Runner.
Sends questions to an AI model in a continuous loop,
scores the responses, and returns a list of results.
"""
from email.mime import message
import os
import sys
import time
from urllib import response
sys.path.append('~/fineval')

from openai import OpenAI
from datetime import datetime
from DataStructures import EvalItem, EvalResult, LLMResponse, TaskType
from fineval.scorers.numerical_scorer import NumericalScorer
from fineval.scorers.classification_scorer import ClassificationScorer

# set up scorers
numerical_scorer = NumericalScorer()
classification_scorer = ClassificationScorer()

# set up the NVIDIA API client
def get_client(base_url: str = "https://integrate.api.nvidia.com/v1") -> OpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Set CEREBRAS_API_KEY or NVIDIA_API_KEY env var first")
    return OpenAI(base_url=base_url, api_key=api_key)


def build_prompt(item: EvalItem) -> str:
    """Build the right prompt for each task type."""
    if item.task_type == TaskType.CLASSIFICATION:
        return (
            f"Classify the sentiment of this finance news sentence "
            f"as one of: positive, neutral, negative.\n\n"
            f"Sentence: {item.input['text']}\n\n"
            f"Respond with one word only: positive, neutral, or negative."
        )
    elif item.task_type == TaskType.NUMERICAL_QA:
        expected = str(item.expected_output).lower().strip()
        # include the question if the item has one stored separately
        question = item.input.get('question', '')
        question_line = f"\n\nQuestion: {question}" if question else ""
        if expected in ["yes", "no"]:
            return (
                f"{item.input['context']}{question_line}\n\n"
                f"Answer with yes or no only. No explanation."
            )
        return (
            f"{item.input['context']}{question_line}\n\n"
            f"Answer with a single number only. "
            f"No units, no commas, no explanation."
        )
    elif item.task_type == TaskType.MULTITURN_QA:
        return (
            f"{item.input['query']}\n\n"
            f"Answer with a single number only. "
            f"No units, no commas, no explanation."
        )
    else:
        return str(item.input)
    
def call_model(client: OpenAI, model_id: str, prompt: str, retries: int = 10) -> tuple[str, int, int, int]:
    """
    Send a prompt to the model and return the response.
    Retries automatically if rate limited.
    Returns (response_text, latency_ms, tokens_in, tokens_out).
    """
    for attempt in range(retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                 model=model_id,
                 messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            latency_ms = int((time.time() - start) * 1000)
            message = response.choices[0].message
            raw_text = message.content or message.reasoning or ""
            return (
                raw_text.strip(),
                latency_ms,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )
        except Exception as e:
            if attempt < retries - 1:
                print(f"   Error: {e}. Retrying in 5 seconds...")
                time.sleep(15)
            else:
                return f"ERROR: {e}", 0, 0, 0


def score_item(item: EvalItem, predicted: str) -> tuple[float, bool]:
    """Route the item to the right scorer based on task type."""
    if item.task_type == TaskType.CLASSIFICATION:
        return classification_scorer.score(predicted, str(item.expected_output))
    else:
        return numerical_scorer.score(predicted, str(item.expected_output))


def run_eval(
    items: list[EvalItem],
    model_id: str,
    dataset_name: str,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    cost_per_million_input: float = 0.0,
    cost_per_million_output: float = 0.0,
) -> list[EvalResult]:
    """
    Run the evaluation loop.
    Sends every item to the model continuously without stopping.
    Returns a list of EvalResult objects.
    """
    client = get_client(base_url)
    results = []

    print(f"\nRunning eval on {len(items)} items from {dataset_name} using {model_id}\n")

    for i, item in enumerate(items):
        print(f"[{i+1}/{len(items)}] {item.id}")

        prompt = build_prompt(item)
        raw_text, latency_ms, tokens_in, tokens_out = call_model(client, model_id, prompt)
        score, is_correct = score_item(item, raw_text)

        # calculate real cost based on token counts
        input_cost = (tokens_in / 1_000_000) * cost_per_million_input
        output_cost = (tokens_out / 1_000_000) * cost_per_million_output
        total_cost = input_cost + output_cost

        result = EvalResult(
            item_id=item.id,
            dataset_name=dataset_name,
            model_id=model_id,
            score=score,
            is_correct=is_correct,
            response=LLMResponse(
                raw_text=raw_text,
                parsed_value=raw_text,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_cents=total_cost,
            ),
            expected=item.expected_output,
            timestamp=datetime.now(),
        )
        results.append(result)

        if item.task_type.value == "classification":
            print(f"   question: {item.input['text']}")
        elif item.task_type.value == "numerical_qa":
            print(f"   question: {item.input['question']}")
        elif item.task_type.value == "multiturn_qa":
            print(f"   turn: {item.input['turn']}")
        print(f"   expected: {item.expected_output}")
        print(f"   AI answered: {raw_text}")
        print(f"   score: {score:.2f} correct: {is_correct}")
        print(f"   cost: {total_cost:.6f} cents")
        print()

        time.sleep(0.5)

    return results
def print_metrics(results: list[EvalResult], dataset_name: str, model_id: str):
    """
    Calculate and print evaluation metrics after a benchmark run.
    Includes accuracy, F1 score, average latency, and total cost.
    """
    from sklearn.metrics import f1_score
    import numpy as np

    if not results:
        print("No results to evaluate.")
        return

    # calculate accuracy
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total * 100

    # calculate F1 score
    # collect expected and predicted labels
    expected_labels = [str(r.expected).lower().strip() for r in results]
    predicted_labels = [r.response.raw_text.lower().strip() for r in results]

    # get unique labels
    unique_labels = list(set(expected_labels))

    try:
        f1 = f1_score(expected_labels, predicted_labels, labels=unique_labels, average="weighted", zero_division=0)
    except Exception:
        f1 = 0.0

    # calculate average latency
    latencies = [r.response.latency_ms for r in results]
    avg_latency = int(sum(latencies) / len(latencies))

    # calculate total cost
    total_cost = sum(r.response.cost_cents for r in results)

    # print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Dataset:         {dataset_name}")
    print(f"Model:           {model_id}")
    print(f"Total items:     {total}")
    print(f"Correct:         {correct}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"F1 Score:        {f1:.4f}")
    print(f"Avg latency:     {avg_latency}ms")
    print(f"Total cost:      {total_cost:.4f} cents")
    print("=" * 50 + "\n")
"""
Reference runner — read on Day 1, replace by Day 3.

Demonstrates the minimal flow:
  1. Load sample data
  2. Configure an LLM (NVIDIA Build, OpenAI-compatible)
  3. Call the LLM on one item
  4. Score the response
  5. Print the result

You'll replace this with a real CLI, real scorers, real datasets, and
the leaderboard aggregation in `DataStructures.py`. This script exists
so you can verify your API key works on Day 1 before doing anything else.

Usage:
    export NVIDIA_API_KEY=nvapi-...
    pip install openai pydantic
    python3 example_runner.py
"""
import json
import os
import re
import time
from pathlib import Path

# pip install openai
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_ID = "meta/llama-3.3-70b-instruct"  # pick any from build.nvidia.com

SAMPLE_DATA_PATH = Path(__file__).parent / "sample-data.json"


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible client pointed at NVIDIA)
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model_id: str, prompt: str) -> tuple[str, int, int, int]:
    """Returns (raw_text, latency_ms, tokens_in, tokens_out)."""
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    latency_ms = int((time.time() - start) * 1000)
    return (
        response.choices[0].message.content,
        latency_ms,
        response.usage.prompt_tokens if response.usage else 0,
        response.usage.completion_tokens if response.usage else 0,
    )


# ---------------------------------------------------------------------------
# Prompt construction (you'll generalize this per task type)
# ---------------------------------------------------------------------------

def build_numerical_qa_prompt(item: dict) -> str:
    return (
        f"Context:\n{item['input']['context']}\n\n"
        f"Question: {item['input']['question']}\n\n"
        f"Answer with a single number. No units, no commas, no explanation. "
        f"If the answer is a ratio (e.g. growth rate), give it as a decimal (e.g. 0.145 for 14.5%)."
    )


def build_classification_prompt(item: dict) -> str:
    labels = item["input"]["labels"]
    return (
        f"Classify the sentiment of this finance-news sentence as one of: {', '.join(labels)}.\n\n"
        f"Sentence: {item['input']['text']}\n\n"
        f"Respond with one of [{', '.join(labels)}] and nothing else."
    )


# ---------------------------------------------------------------------------
# Scoring (you'll replace with proper Scorer classes)
# ---------------------------------------------------------------------------

def extract_first_number(text: str) -> float | None:
    """Pull the first number out of free-form text."""
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def score_numerical(predicted_text: str, expected: float, tolerance: float) -> tuple[float, bool]:
    pred = extract_first_number(predicted_text)
    if pred is None:
        return 0.0, False
    correct = abs(pred - expected) <= tolerance
    # Soft score: 1.0 at exact, decaying linearly to 0 at 10x tolerance.
    err = abs(pred - expected)
    soft = max(0.0, 1.0 - err / (10 * tolerance)) if tolerance > 0 else (1.0 if pred == expected else 0.0)
    return soft, correct


def score_classification(predicted_text: str, expected: str) -> tuple[float, bool]:
    correct = expected.lower() in predicted_text.lower()
    return (1.0 if correct else 0.0), correct


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set NVIDIA_API_KEY env var. Get a free key at https://build.nvidia.com"
        )

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    # Load sample data.
    data = json.loads(SAMPLE_DATA_PATH.read_text())
    datasets = data["datasets"]

    # Pick one item from each task type for the demo.
    demo_items = []
    seen_types = set()
    for ds in datasets:
        for item in ds["items"]:
            tt = item["task_type"]
            if tt in seen_types:
                continue
            seen_types.add(tt)
            demo_items.append((ds["name"], item))
            break

    print(f"Running {len(demo_items)} demo items against model {MODEL_ID}\n")
    for ds_name, item in demo_items:
        tt = item["task_type"]
        if tt == "numerical_qa":
            prompt = build_numerical_qa_prompt(item)
        elif tt == "classification":
            prompt = build_classification_prompt(item)
        else:
            print(f"[{ds_name}/{item['id']}] task_type={tt} not handled in demo runner; skipping.")
            continue

        try:
            raw, latency, t_in, t_out = call_llm(client, MODEL_ID, prompt)
        except Exception as e:
            print(f"[{ds_name}/{item['id']}] ERROR: {e}")
            continue

        if tt == "numerical_qa":
            tolerance = item["metadata"].get("tolerance", 0.01)
            score, correct = score_numerical(raw, float(item["expected_output"]), tolerance)
        else:
            score, correct = score_classification(raw, str(item["expected_output"]))

        print(f"[{ds_name}/{item['id']}]")
        print(f"  task: {tt}")
        print(f"  expected: {item['expected_output']}")
        print(f"  raw response: {raw.strip()[:120]}")
        print(f"  score={score:.2f} correct={correct} latency={latency}ms tokens_in={t_in} tokens_out={t_out}")
        print()


if __name__ == "__main__":
    main()

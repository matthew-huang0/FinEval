"""
FinEval CLI.
Run a full benchmark from the command line with a single command.

Usage:
    python3 -m fineval.cli --model meta/llama-3.3-70b-instruct --datasets finqa,finpb,convfinqa --max-items 20 --out results/benchmark.json
"""
import argparse
import json
import os
import sys

from fineval.datasets.finpb_loader import load_finpb
from fineval.datasets.finqa_loader import load_finqa
from fineval.datasets.convfinqa_loader import load_convfinqa
from fineval.runner import run_eval, print_metrics


def load_dataset(name: str, max_items: int):
    """Load a dataset by name and return a list of EvalItems."""
    name = name.strip().lower()
    if name == "finpb":
        return load_finpb(max_items=max_items), "finpb"
    elif name == "finqa":
        return load_finqa(max_items=max_items), "finqa"
    elif name == "convfinqa":
        return load_convfinqa(max_items=max_items), "convfinqa"
    elif name == "custom":
        from fineval.datasets.custom_loader import load_custom
        return load_custom(max_items=max_items), "custom"
    else:
        print(f"Unknown dataset: {name}. Valid options are: finpb, finqa, convfinqa, custom")
        sys.exit(1)


def save_results(results, path: str):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = []
    for r in results:
        data.append({
            "item_id": r.item_id,
            "dataset_name": r.dataset_name,
            "model_id": r.model_id,
            "score": r.score,
            "is_correct": r.is_correct,
            "expected": str(r.expected),
            "ai_answer": r.response.raw_text,
            "latency_ms": r.response.latency_ms,
            "tokens_in": r.response.tokens_in,
            "tokens_out": r.response.tokens_out,
            "cost_cents": r.response.cost_cents,
            "timestamp": str(r.timestamp)
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


def get_default_costs(provider: str, model: str) -> tuple[float, float]:
    """
    Return default cost per million tokens (input, output) based on provider and model.
    Returns (0.0, 0.0) for free tier models.
    """
    pricing = {
        "cerebras": {
            "llama-3.3-70b": (0.0, 0.0),
            "llama3.1-8b": (0.0, 0.0),
            "llama-4-scout-17b-16e-instruct": (0.0, 0.0),
            "gpt-oss-120b": (0.0, 0.0),
        },
        "nvidia": {
            "meta/llama-3.3-70b-instruct": (0.0, 0.0),
            "meta/llama-3.2-1b-instruct": (0.0, 0.0),
            "mistralai/mixtral-8x22b-instruct": (0.0, 0.0),
        },
        "openai": {
            "gpt-4o": (500.0, 1500.0),
            "gpt-4o-mini": (15.0, 60.0),
        },
        "anthropic": {
            "claude-sonnet-4-6": (300.0, 1500.0),
        },
    }

    provider_pricing = pricing.get(provider.lower(), {})
    return provider_pricing.get(model, (0.0, 0.0))


def main():
    parser = argparse.ArgumentParser(description="FinEval — Finance LLM Benchmark")

    parser.add_argument(
        "--model",
        type=str,
        default="meta/llama-3.3-70b-instruct",
        help="Model ID to benchmark"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="nvidia",
        help="Provider name e.g. nvidia, cerebras, openai, anthropic"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--cost-per-million-input-tokens",
        type=float,
        default=-1.0,
        help="Cost in cents per million input tokens. If not set, auto-detected from provider and model."
    )
    parser.add_argument(
        "--cost-per-million-output-tokens",
        type=float,
        default=-1.0,
        help="Cost in cents per million output tokens. If not set, auto-detected from provider and model."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="finpb,finqa,convfinqa",
        help="Comma separated list of datasets: finpb, finqa, convfinqa"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Maximum number of items per dataset"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/benchmark.json",
        help="Path to save results JSON file"
    )

    args = parser.parse_args()

    # auto set costs based on provider and model unless explicitly provided
    default_input_cost, default_output_cost = get_default_costs(args.provider, args.model)
    if args.cost_per_million_input_tokens < 0:
        args.cost_per_million_input_tokens = default_input_cost
    if args.cost_per_million_output_tokens < 0:
        args.cost_per_million_output_tokens = default_output_cost

    dataset_names = args.datasets.split(",")
    all_results = []

    print("\nFinEval Benchmark")
    print(f"Model:          {args.model}")
    print(f"Provider:       {args.provider}")
    print(f"Base URL:       {args.base_url}")
    print(f"Input cost:     {args.cost_per_million_input_tokens} cents per million tokens")
    print(f"Output cost:    {args.cost_per_million_output_tokens} cents per million tokens")
    print(f"Datasets:       {args.datasets}")
    print(f"Max items:      {args.max_items}")
    print(f"Output:         {args.out}")
    print("=" * 50)

    for name in dataset_names:
        items, dataset_name = load_dataset(name, args.max_items)
        results = run_eval(
            items,
            args.model,
            dataset_name,
            base_url=args.base_url,
            cost_per_million_input=args.cost_per_million_input_tokens,
            cost_per_million_output=args.cost_per_million_output_tokens,
        )
        print_metrics(results, dataset_name, args.model)
        all_results.extend(results)

    save_results(all_results, args.out)

    print(f"\nTotal items evaluated: {len(all_results)}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
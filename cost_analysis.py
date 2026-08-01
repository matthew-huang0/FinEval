"""
cost_analysis.py
Estimates the market cost of each model based on token usage.

The models were run for free (Cerebras free tier and local Ollama), so
these are reference prices from paid hosts, representing what each model
would cost in production. Two approximations apply: Gemma 4 31B uses its
predecessor Gemma 3 27B's market rate as a proxy, since the newer model
is not yet broadly priced; and the small Llama's output rate is estimated
at roughly twice its input rate, as output pricing was not separately
published.

Prices are per million tokens, in US dollars.
"""
import json
import os

# reference market prices per million tokens (input, output) in USD
PRICING = {
    "gpt-oss-120b":  {"input": 0.35, "output": 0.75, "source": "Cerebras published rate"},
    "gemma-4-31b":   {"input": 0.08, "output": 0.16, "source": "proxy: Gemma 3 27B market rate"},
    "llama3.2:1b":   {"input": 0.02, "output": 0.04, "source": "Meta Llama 3.2 1B cheapest host"},
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def analyze_model(path):
    """Compute cost and accuracy for one model's results file."""
    data = load_results(path)
    if not data:
        return None

    model_id = data[0].get("model_id", os.path.basename(path))
    price = PRICING.get(model_id)
    if price is None:
        print(f"No pricing for {model_id}, skipping.")
        return None

    total_tokens_in = sum(r.get("tokens_in", 0) for r in data)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in data)
    n_items = len(data)
    n_correct = sum(1 for r in data if r.get("is_correct"))
    accuracy = n_correct / n_items * 100

    cost_in = total_tokens_in / 1_000_000 * price["input"]
    cost_out = total_tokens_out / 1_000_000 * price["output"]
    total_cost = cost_in + cost_out

    cost_per_correct = total_cost / n_correct if n_correct else 0.0
    cost_per_1k_items = total_cost / n_items * 1000

    return {
        "model": model_id,
        "n_items": n_items,
        "accuracy": accuracy,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "total_cost_usd": total_cost,
        "cost_per_correct_usd": cost_per_correct,
        "cost_per_1k_items_usd": cost_per_1k_items,
        "source": price["source"],
    }


def main():
    model_files = [
        "results/large-gptoss120b.json",
        "results/medium-gemma31b.json",
        "results/small-llama1b.json",
    ]

    results = []
    for path in model_files:
        if os.path.exists(path):
            r = analyze_model(path)
            if r:
                results.append(r)

    print("=" * 78)
    print("FinEval Model Cost Analysis")
    print("Reference market prices; models were actually run for free.")
    print("=" * 78)
    print()
    print(f"{'Model':<16} {'Acc':>6} {'Tokens In':>12} {'Tokens Out':>12} "
          f"{'Cost USD':>10} {'$/1k items':>11}")
    print("-" * 78)
    for r in results:
        print(f"{r['model']:<16} "
              f"{r['accuracy']:>5.1f}% "
              f"{r['tokens_in']:>12,} "
              f"{r['tokens_out']:>12,} "
              f"${r['total_cost_usd']:>8.4f} "
              f"${r['cost_per_1k_items_usd']:>9.4f}")
    print("-" * 78)
    print()

    print("Cost per correct answer (across all 1500 items):")
    for r in results:
        print(f"  {r['model']:<16} ${r['cost_per_correct_usd']:.6f}")
    print()

    os.makedirs("results", exist_ok=True)
    with open("results/cost_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/cost_analysis.json")


if __name__ == "__main__":
    main()
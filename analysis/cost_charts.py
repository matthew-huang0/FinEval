"""
cost_charts.py
Builds the two cost-related charts my manager requested:
1. Model cost vs accuracy (dollar cost)
2. Token usage vs accuracy (which reveals the reasoning model's verbosity)

Reads results/cost_analysis.json, which is produced by cost_analysis.py.
Run cost_analysis.py first.
"""
import json
import matplotlib.pyplot as plt

# load the cost analysis data
with open("results/cost_analysis.json") as f:
    data = json.load(f)

colors = {
    "gpt-oss-120b": "#2980b9",
    "gemma-4-31b": "#27ae60",
    "llama3.2:1b": "#e67e22",
}


# Chart 1 - Model dollar cost vs accuracy

def chart_cost_vs_accuracy():
    fig, ax = plt.subplots(figsize=(11, 7))

    # manual label offsets so nothing overlaps (x, y in points)
    offsets = {
        "gpt-oss-120b": (0, 18),
        "gemma-4-31b": (0, 18),
        "llama3.2:1b": (0, 18),
    }

    for r in data:
        model = r["model"]
        ax.scatter(
            r["total_cost_usd"],
            r["accuracy"],
            s=400,
            color=colors.get(model, "gray"),
            edgecolors="black",
            alpha=0.85,
            zorder=3,
        )
        dx, dy = offsets.get(model, (0, 18))
        ax.annotate(
            f"{model}\n{r['accuracy']:.1f}% | ${r['total_cost_usd']:.2f}",
            (r["total_cost_usd"], r["accuracy"]),
            fontsize=10,
            xytext=(dx, dy),
            textcoords="offset points",
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("Total cost for 1,500 items (USD) — lower is cheaper")
    ax.set_ylabel("Average accuracy across all datasets (%) — higher is better")
    ax.set_title("Cost vs Accuracy — Reference Market Pricing")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    # pad the x-axis so labels aren't cut off
    costs = [r["total_cost_usd"] for r in data]
    span = max(costs) - min(costs)
    ax.set_xlim(min(costs) - span * 0.15, max(costs) + span * 0.25)

    plt.tight_layout()
    plt.savefig("results/chart-cost-vs-accuracy.png", dpi=150)
    print("Saved results/chart-cost-vs-accuracy.png")
    plt.close()

# Chart 2 - Token usage vs accuracy (reveals verbosity)
def chart_tokens_vs_accuracy():
    fig, ax = plt.subplots(figsize=(11, 7))

    offsets = {
        "gpt-oss-120b": (0, 18),
        "gemma-4-31b": (0, 18),
        "llama3.2:1b": (0, 18),
    }

    for r in data:
        model = r["model"]
       # use OUTPUT tokens only - this is where the verbosity difference lives
        total_k = r["tokens_out"] / 1000
        ax.scatter(
            total_k,
            r["accuracy"],
            s=400,
            color=colors.get(model, "gray"),
            edgecolors="black",
            alpha=0.85,
            zorder=3,
        )
        dx, dy = offsets.get(model, (0, 18))
        ax.annotate(
            f"{model}\n{r['accuracy']:.1f}% | {r['tokens_out']:,} out",
            (total_k, r["accuracy"]),
            fontsize=10,
            xytext=(dx, dy),
            textcoords="offset points",
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("Output tokens generated for 1,500 items (thousands) — lower is more efficient")
    ax.set_ylabel("Average accuracy across all datasets (%) — higher is better")
    ax.set_title("Token Usage vs Accuracy — Output Tokens Labeled")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    totals = [r["tokens_out"] / 1000 for r in data]
    span = max(totals) - min(totals)
    ax.set_xlim(min(totals) - span * 0.15, max(totals) + span * 0.25)

    plt.tight_layout()
    plt.savefig("results/chart-tokens-vs-accuracy.png", dpi=150)
    print("Saved results/chart-tokens-vs-accuracy.png")
    plt.close()


if __name__ == "__main__":
    chart_cost_vs_accuracy()
    chart_tokens_vs_accuracy()
    print("\nBoth cost charts generated in results/ folder.")
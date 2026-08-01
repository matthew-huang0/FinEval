"""
analysis.py
FinEval analysis and visualization.
Loads all benchmark results and generates comparison charts.
Uses median latency to reflect typical speed rather than being
skewed by rare slow retry calls.
"""
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Load all results into a dataframe
# -------------------------------------------------------------------
def load_all_results():
    rows = []
    for path in glob.glob("results/*.json"):
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue
        # skip files that aren't benchmark results (cost_analysis.json, etc.)
        first = data[0]
        if not isinstance(first, dict) or "dataset_name" not in first or "model_id" not in first:
            continue
        model_id = data[0].get("model_id", os.path.basename(path))
        datasets = set(r.get("dataset_name") for r in data)
        for ds in datasets:
            ds_rows = [r for r in data if r.get("dataset_name") == ds]
            n = len(ds_rows)
            correct = sum(1 for r in ds_rows if r.get("is_correct"))
            accuracy = correct / n * 100
            latencies = [r.get("latency_ms", 0) for r in ds_rows]
            median_latency = float(np.median(latencies))
            rows.append({
                "model": model_id,
                "dataset": ds,
                "accuracy": accuracy,
                "median_latency_ms": median_latency,
                "n_items": n,
            })
    return pd.DataFrame(rows)


df = load_all_results()
print("Loaded results:")
print(df.to_string(index=False))
print()

# -------------------------------------------------------------------
# Chart 1 - Accuracy by model and dataset
# -------------------------------------------------------------------
def chart_accuracy_by_model():
    llm_models = ["gpt-oss-120b", "gemma-4-31b", "llama3.2:1b"]
    plot_df = df[df["model"].isin(llm_models)]

    datasets = ["finpb", "finqa", "convfinqa"]
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(llm_models):
        accuracies = []
        for ds in datasets:
            row = plot_df[(plot_df["model"] == model) & (plot_df["dataset"] == ds)]
            accuracies.append(row["accuracy"].values[0] if len(row) else 0)
        ax.bar(x + i * width, accuracies, width, label=model)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Model and Dataset")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Financial PhraseBank", "FinQA", "ConvFinQA"])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/chart-accuracy.png", dpi=150)
    print("Saved results/chart-accuracy.png")
    plt.close()


# -------------------------------------------------------------------
# Chart 2 - LLM vs Classical ML on Financial PhraseBank
# -------------------------------------------------------------------
def chart_llm_vs_baseline():
    finpb = df[df["dataset"] == "finpb"].sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c0392b" if "TF-IDF" in m else "#2980b9" for m in finpb["model"]]
    ax.barh(finpb["model"], finpb["accuracy"], color=colors)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("LLMs vs Classical ML — Financial PhraseBank Sentiment")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", label="LLM"),
        Patch(facecolor="#c0392b", label="Classical ML"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig("results/chart-llm-vs-baseline.png", dpi=150)
    print("Saved results/chart-llm-vs-baseline.png")
    plt.close()


# -------------------------------------------------------------------
# Chart 3 - Average accuracy vs median latency (one point per model)
# -------------------------------------------------------------------
def chart_accuracy_vs_latency():
    llm_models = ["gpt-oss-120b", "gemma-4-31b", "llama3.2:1b"]
    plot_df = df[df["model"].isin(llm_models)]

    summary = plot_df.groupby("model").agg(
        avg_accuracy=("accuracy", "mean"),
        median_latency=("median_latency_ms", "median"),
    ).reset_index()

    colors = {"gpt-oss-120b": "#2980b9", "gemma-4-31b": "#27ae60", "llama3.2:1b": "#e67e22"}

    # manual label offsets so nothing overlaps (x offset, y offset in points)
    label_offsets = {
        "gpt-oss-120b": (0, 28),
        "gemma-4-31b": (0, -45),
        "llama3.2:1b": (20, 12),
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for _, row in summary.iterrows():
        model = row["model"]
        ax.scatter(
            row["median_latency"],
            row["avg_accuracy"],
            s=400,
            color=colors.get(model, "gray"),
            edgecolors="black",
            alpha=0.85,
            zorder=3,
        )
        dx, dy = label_offsets.get(model, (12, 8))
        ax.annotate(
            f"{model}\n{row['avg_accuracy']:.0f}% avg accuracy",
            (row["median_latency"], row["avg_accuracy"]),
            fontsize=10,
            xytext=(dx, dy),
            textcoords="offset points",
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("Median Latency (ms) — lower is faster")
    ax.set_ylabel("Average Accuracy across all datasets (%) — higher is better")
    ax.set_title("Accuracy vs Latency — Average Across All Three Datasets")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)

    xmin = summary["median_latency"].min()
    xmax = summary["median_latency"].max()
    span = max(xmax - xmin, 1)
    ax.set_xlim(xmin - span * 0.25, xmax + span * 0.35)

    plt.tight_layout()
    plt.savefig("results/chart-accuracy-vs-latency.png", dpi=150)
    print("Saved results/chart-accuracy-vs-latency.png")
    plt.close()


if __name__ == "__main__":
    chart_accuracy_by_model()
    chart_llm_vs_baseline()
    chart_accuracy_vs_latency()
    print("\nAll charts generated in results/ folder.")
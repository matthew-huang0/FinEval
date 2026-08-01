"""
leaderboard.py
Auto-discovers every benchmark result file in the results folder and
builds a combined leaderboard. Any model benchmarked through the CLI
or baseline script appears automatically with no code editing needed.
Exports to Markdown and CSV.
"""
import json
import os
import glob
import sys
from datetime import datetime

from fineval.data_structures import Leaderboard, LeaderboardEntry, TaskType


# map dataset names to task types
TASK_TYPE_MAP = {
    "finpb": TaskType.CLASSIFICATION,
    "finqa": TaskType.NUMERICAL_QA,
    "convfinqa": TaskType.MULTITURN_QA,
}


def load_results_from_json(path: str) -> list[dict]:
    """Load benchmark results from a saved JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return []


def build_entry(results: list[dict], model_id: str, dataset_name: str) -> LeaderboardEntry | None:
    """Build one leaderboard entry for a model on a dataset."""
    dataset_results = [r for r in results if r.get("dataset_name") == dataset_name]
    if not dataset_results:
        return None

    n_items = len(dataset_results)
    n_correct = sum(1 for r in dataset_results if r.get("is_correct"))
    accuracy = n_correct / n_items
    mean_score = sum(r.get("score", 0.0) for r in dataset_results) / n_items

    # use f1_override if present (baselines store their real F1 here)
    f1_override = dataset_results[0].get("f1_override")
    display_score = f1_override if f1_override is not None else mean_score

    latencies = [r.get("latency_ms", 0) for r in dataset_results]
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.5)]
    p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]
    total_cost = sum(r.get("cost_cents", 0.0) for r in dataset_results)
    cost_per_correct = total_cost / n_correct if n_correct > 0 else 0.0

    task_type = TASK_TYPE_MAP.get(dataset_name, TaskType.NUMERICAL_QA)

    return LeaderboardEntry(
        model_id=model_id,
        dataset_name=dataset_name,
        task_type=task_type,
        n_items=n_items,
        accuracy=accuracy,
        mean_score=display_score,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_cost_cents=total_cost,
        cost_per_correct_cents=cost_per_correct,
        timestamp=datetime.now(),
    )


def discover_and_build_entries() -> list[LeaderboardEntry]:
    """Scan the results folder and build entries from every JSON file."""
    entries = []
    result_files = glob.glob("results/*.json")

    for path in result_files:
        results = load_results_from_json(path)
        if not results:
            continue

        # skip files that aren't benchmark results (e.g. cost_analysis.json,
        # leaderboard exports). a real result file has these fields on each row.
        first = results[0]
        if not isinstance(first, dict) or "dataset_name" not in first or "model_id" not in first:
            continue

        # get the model id from inside the file
        model_id = results[0].get("model_id", os.path.basename(path))

        # find every dataset present in this file
        datasets_in_file = set(r.get("dataset_name") for r in results)
        for dataset_name in datasets_in_file:
            entry = build_entry(results, model_id, dataset_name)
            if entry:
                entries.append(entry)

    return entries


def leaderboard_to_markdown(leaderboard: Leaderboard) -> str:
    lines = []
    lines.append("# FinEval Leaderboard")
    lines.append(f"Generated: {leaderboard.generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("| Model | Dataset | Task | Items | Accuracy | F1/Mean Score | P50 Latency | Cost (cents) |")
    lines.append("|-------|---------|------|-------|----------|---------------|-------------|--------------|")

    for entry in sorted(leaderboard.entries, key=lambda e: (e.dataset_name, -e.accuracy)):
        lines.append(
            f"| {entry.model_id} "
            f"| {entry.dataset_name} "
            f"| {entry.task_type.value} "
            f"| {entry.n_items} "
            f"| {entry.accuracy*100:.2f}% "
            f"| {entry.mean_score:.4f} "
            f"| {entry.p50_latency_ms}ms "
            f"| {entry.total_cost_cents:.4f} |"
        )

    return "\n".join(lines)


def leaderboard_to_csv(leaderboard: Leaderboard) -> str:
    lines = []
    lines.append(
        "model_id,dataset_name,task_type,n_items,accuracy,"
        "mean_score,p50_latency_ms,p95_latency_ms,"
        "total_cost_cents,cost_per_correct_cents"
    )
    for entry in leaderboard.entries:
        lines.append(
            f"{entry.model_id},"
            f"{entry.dataset_name},"
            f"{entry.task_type.value},"
            f"{entry.n_items},"
            f"{entry.accuracy:.4f},"
            f"{entry.mean_score:.4f},"
            f"{entry.p50_latency_ms},"
            f"{entry.p95_latency_ms},"
            f"{entry.total_cost_cents:.4f},"
            f"{entry.cost_per_correct_cents:.4f}"
        )
    return "\n".join(lines)


def main():
    entries = discover_and_build_entries()

    if not entries:
        print("No result files found in results/. Run a benchmark first.")
        return

    leaderboard = Leaderboard(
        generated_at=datetime.now(),
        entries=entries,
    )

    os.makedirs("results", exist_ok=True)

    markdown = leaderboard_to_markdown(leaderboard)
    with open("results/leaderboard.md", "w") as f:
        f.write(markdown)
    print("Leaderboard saved to results/leaderboard.md")

    csv = leaderboard_to_csv(leaderboard)
    with open("results/leaderboard.csv", "w") as f:
        f.write(csv)
    print("Leaderboard saved to results/leaderboard.csv")

    print()
    print(markdown)


if __name__ == "__main__":
    main()
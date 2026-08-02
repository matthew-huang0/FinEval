"""
custom_loader.py
Loads the custom SEC EDGAR 10-K dataset.
Follows the same EvalItem structure as the other dataset loaders,
so it flows through the runner, scorer, and leaderboard unchanged.
"""
import json
import os
import sys

from fineval.data_structures import EvalItem, TaskType

CUSTOM_DATASET_PATH = os.path.join(os.path.dirname(__file__), "custom", "custom_dataset.json")


def load_custom(max_items: int = None) -> list[EvalItem]:
    """
    Load the custom 10-K dataset and return a list of EvalItems.
    Each item is a numerical_qa task, matching FinQA's format.
    """
    with open(CUSTOM_DATASET_PATH, "r") as f:
        data = json.load(f)

    if max_items is not None:
        data = data[:max_items]

    items = []
    for entry in data:
        items.append(EvalItem(
            id=entry["id"],
            task_type=TaskType.NUMERICAL_QA,
            input={
                "context": entry["context"],
                "question": entry["question"],
            },
            expected_output=str(entry["answer"]),
            metadata={
                "company": entry["company"],
                "source": entry["source"],
            },
        ))

    print(f"Loaded {len(items)} items from Custom 10-K dataset")
    return items


if __name__ == "__main__":
    items = load_custom(max_items=5)
    for item in items:
        print(f"\nID: {item.id}")
        print(f"Question: {item.input['question']}")
        print(f"Expected: {item.expected_output}")
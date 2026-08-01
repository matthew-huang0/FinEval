"""
FinQA loader.
Loads ChanceFocus/flare-finqa from HuggingFace and returns a list of EvalItem objects.
"""
import sys
sys.path.append('~/fineval')

from datasets import load_dataset
from DataStructures import EvalItem, TaskType


def load_finqa(split: str = "train", max_items: int = None) -> list[EvalItem]:
    """
    Load FinQA from HuggingFace.
    Returns a list of EvalItem objects.
    """
    dataset = load_dataset("ChanceFocus/flare-finqa")
    data = dataset[split]

    items = []

    for i, row in enumerate(data):
        if max_items and i >= max_items:
            break

        item = EvalItem(
            id=row["id"],
            task_type=TaskType.NUMERICAL_QA,
            input={
                "context": row["query"],
                "question": row["text"]
            },
            expected_output=row["answer"],
            metadata={"tolerance": 0.01}
        )
        items.append(item)

    print(f"Loaded {len(items)} items from FinQA ({split} split)")
    return items


if __name__ == "__main__":
    items = load_finqa(max_items=3)
    for item in items:
        print(f"ID: {item.id}")
        print(f"Question: {item.input['question']}")
        print(f"Expected answer: {item.expected_output}")
        print()

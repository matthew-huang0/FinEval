"""
ConvFinQA loader.
Loads ChanceFocus/flare-convfinqa from HuggingFace and returns a list of EvalItem objects.
Each item represents one turn in a conversation, with full history included.
"""
import sys

from datasets import load_dataset
from fineval.data_structures import EvalItem, TaskType


def load_convfinqa(split: str = "train", max_items: int = None) -> list[EvalItem]:
    """
    Load ConvFinQA from HuggingFace.
    Returns a list of EvalItem objects, one per conversation turn.
    """
    dataset = load_dataset("ChanceFocus/flare-convfinqa")
    data = dataset[split]

    items = []

    for i, row in enumerate(data):
        if max_items and i >= max_items:
            break

        item = EvalItem(
            id=row["id"],
            task_type=TaskType.MULTITURN_QA,
            input={
                "query": row["query"],
                "dialogue_id": row["dialogue_id"],
                "turn": row["turn"]
            },
            expected_output=row["answer"],
            metadata={
                "dialogue_id": row["dialogue_id"],
                "turn": row["turn"],
                "tolerance": 0.01
            }
        )
        items.append(item)

    print(f"Loaded {len(items)} turns from ConvFinQA ({split} split)")
    return items


if __name__ == "__main__":
    items = load_convfinqa(max_items=4)
    for item in items:
        print(f"ID: {item.id}")
        print(f"Turn: {item.input['turn']}")
        print(f"Dialogue ID: {item.input['dialogue_id']}")
        print(f"Expected answer: {item.expected_output}")
        print()

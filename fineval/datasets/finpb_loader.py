"""
Financial PhraseBank loader.
Reads Sentences_AllAgree.txt and returns a list of EvalItem objects.
"""
import sys

from fineval.data_structures import EvalItem, TaskType


FINPB_PATH = "~/Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"


def load_finpb(max_items: int = None) -> list[EvalItem]:
    """
    Load Financial PhraseBank from local file.
    Returns a list of EvalItem objects.
    """
    items = []

    with open(FINPB_PATH, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # stop early if max_items is set
        if max_items and i >= max_items:
            break

        line = line.strip()
        if not line or "@" not in line:
            continue

        sentence, label = line.rsplit("@", 1)
        sentence = sentence.strip()
        label = label.strip()

        item = EvalItem(
            id=f"finpb-{i}",
            task_type=TaskType.CLASSIFICATION,
            input={
                "text": sentence,
                "labels": ["positive", "neutral", "negative"]
            },
            expected_output=label,
            metadata={}
        )
        items.append(item)

    print(f"Loaded {len(items)} items from Financial PhraseBank")
    return items


if __name__ == "__main__":
    items = load_finpb(max_items=5)
    for item in items:
        print(f"ID: {item.id}")
        print(f"Text: {item.input['text']}")
        print(f"Label: {item.expected_output}")
        print()

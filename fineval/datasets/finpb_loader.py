"""
Financial PhraseBank loader.
Reads the Sentences_AllAgree.txt file and returns a list of EvalItem
objects. Financial PhraseBank is licensed for non-commercial research
use only and cannot be redistributed, so it is not bundled with this
repository. Download it separately and point to it with the FINPB_PATH
environment variable (see the README).
"""
import os

from fineval.data_structures import EvalItem, TaskType


# Path to the local Sentences_AllAgree.txt. Override with the FINPB_PATH
# environment variable; the default is a convenience for local use.
FINPB_PATH = os.environ.get(
    "FINPB_PATH",
    os.path.expanduser("~/Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"),
)


def load_finpb(max_items: int = None) -> list[EvalItem]:
    """
    Load Financial PhraseBank from the local file at FINPB_PATH.
    Returns a list of EvalItem objects.
    """
    if not os.path.exists(FINPB_PATH):
        raise FileNotFoundError(
            f"Financial PhraseBank file not found at: {FINPB_PATH}\n"
            "Financial PhraseBank is a non-commercial research dataset and is "
            "not included in this repository.\n"
            "Download it (Sentences_AllAgree.txt from the FinancialPhraseBank-v1.0 "
            "release) and either place it at the path above or set the FINPB_PATH "
            "environment variable to its location. See the README for details."
        )

    items = []
    with open(FINPB_PATH, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if max_items and i >= max_items:
            break
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        item = EvalItem(
            id=f"finpb-{i}",
            task_type=TaskType.CLASSIFICATION,
            input={
                "text": sentence.strip(),
                "labels": ["positive", "neutral", "negative"]
            },
            expected_output=label.strip(),
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

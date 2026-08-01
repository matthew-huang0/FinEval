from datasets import load_dataset

finqa = load_dataset("ChanceFocus/flare-finqa")

print(f"Total train items: {len(finqa['train'])}")
print(f"Total test items: {len(finqa['test'])}")
print()

item = finqa["train"][0]
print(f"ID: {item['id']}")
print(f"Answer: {item['answer']}")
print(f"Keys in item: {list(item.keys())}")

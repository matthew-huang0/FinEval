from datasets import load_dataset

convfinqa = load_dataset("ChanceFocus/flare-convfinqa")
item = convfinqa["train"][0]
print("Fields available:")
print(list(item.keys()))
print()
print("Full item:")
print(item)

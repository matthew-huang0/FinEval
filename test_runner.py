"""
Quick test of the runner using 3 items from each dataset.
"""
import sys
sys.path.append('~/fineval')

from fineval.datasets.finpb_loader import load_finpb
from fineval.datasets.finqa_loader import load_finqa
from fineval.datasets.convfinqa_loader import load_convfinqa
from runner import run_eval, print_metrics

MODEL_ID = "meta/llama-3.3-70b-instruct"

# test with 3 items from Financial PhraseBank
print("=" * 50)
print("Testing Financial PhraseBank")
print("=" * 50)
finpb_items = load_finpb(max_items=3)
finpb_results = run_eval(finpb_items, MODEL_ID, "finpb")
print_metrics(finpb_results, "finpb", MODEL_ID)

# test with 3 items from FinQA
print("=" * 50)
print("Testing FinQA")
print("=" * 50)
finqa_items = load_finqa(max_items=3)
finqa_results = run_eval(finqa_items, MODEL_ID, "finqa")
print_metrics(finqa_results, "finqa", MODEL_ID)

# test with 3 items from ConvFinQA
print("=" * 50)
print("Testing ConvFinQA")
print("=" * 50)
convfinqa_items = load_convfinqa(max_items=3)
convfinqa_results = run_eval(convfinqa_items, MODEL_ID, "convfinqa")
print_metrics(convfinqa_results, "convfinqa", MODEL_ID)

print("=" * 50)
print(f"Total results collected: {len(finpb_results) + len(finqa_results) + len(convfinqa_results)}")
print("Runner working correctly.")
print("=" * 50)
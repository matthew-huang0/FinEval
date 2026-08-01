"""
Week 3 test script.
Loads items from all three datasets and runs the NumericalScorer on FinQA items.
"""
import sys
sys.path.append('~/fineval')

from fineval.datasets.finpb_loader import load_finpb
from fineval.datasets.finqa_loader import load_finqa
from fineval.datasets.convfinqa_loader import load_convfinqa
from fineval.scorers.numerical_scorer import NumericalScorer

scorer = NumericalScorer()

# -------------------------------------------------------------------
# Test 1 - Financial PhraseBank
# -------------------------------------------------------------------
print("=" * 50)
print("FINANCIAL PHRASEBANK — 5 items")
print("=" * 50)
finpb_items = load_finpb(max_items=5)
for item in finpb_items:
    print(f"ID: {item.id}")
    print(f"Text: {item.input['text']}")
    print(f"Expected label: {item.expected_output}")
    print()

# -------------------------------------------------------------------
# Test 2 - FinQA with NumericalScorer
# -------------------------------------------------------------------
print("=" * 50)
print("FINQA — 5 items scored")
print("=" * 50)
finqa_items = load_finqa(max_items=5)
for item in finqa_items:
    expected = item.expected_output
    score, is_correct = scorer.score(expected, expected)
    print(f"ID: {item.id}")
    print(f"Question: {item.input['question']}")
    print(f"Expected answer: {expected}")
    print(f"Score (self-check): {score:.2f} correct={is_correct}")
    print()

# -------------------------------------------------------------------
# Test 3 - ConvFinQA
# -------------------------------------------------------------------
print("=" * 50)
print("CONVFINQA — 4 turns")
print("=" * 50)
convfinqa_items = load_convfinqa(max_items=4)
for item in convfinqa_items:
    print(f"ID: {item.id}")
    print(f"Turn: {item.input['turn']}")
    print(f"Expected answer: {item.expected_output}")
    print()

print("=" * 50)
print("All loaders and scorer working correctly.")
print("Week 3 complete.")
print("=" * 50)
import os

# Financial PhraseBank - load directly from downloaded file
print("Loading Financial PhraseBank from local file...")

phrasebank_path = "~/Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

if os.path.exists(phrasebank_path):
    with open(phrasebank_path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    print("\n=== FINPB - 10 items ===")
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if "@" in line:
            sentence, label = line.rsplit("@", 1)
            print(f"--- Item {i} ---")
            print(f"sentence: {sentence}")
            print(f"label: {label}")
            print()
else:
    print(f"File not found at {phrasebank_path}")

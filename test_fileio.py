with open("~/Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "r", encoding="latin-1") as f:
    lines = f.readlines()

for line in lines[:3]:
    line = line.strip()
    sentence, label = line.rsplit("@", 1)
    print(f"sentence: {sentence}")
    print(f"label: {label}")
    print()

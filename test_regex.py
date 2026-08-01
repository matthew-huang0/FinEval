import re

def normalize_number(text):
    text = text.lower()
    text = text.replace("$", "").replace(",", "").replace("£", "").replace("€", "")
    text = text.strip()
    
    if "million" in text:
        number = re.search(r"-?\d+\.?\d*", text)
        if number:
            return float(number.group()) * 1_000_000
    
    if "billion" in text:
        number = re.search(r"-?\d+\.?\d*", text)
        if number:
            return float(number.group()) * 1_000_000_000
    
    try:
        return float(text)
    except ValueError:
        return None

# Test cases
print(normalize_number("$1.5 million"))
print(normalize_number("1,500,000"))
print(normalize_number("1.5e6"))
print(normalize_number("0.1449"))
print(normalize_number("wrong"))

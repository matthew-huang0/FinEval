"""
test_normalizer.py
Tests for the standalone normalizer module.

These are hand written unit tests. They prove the normalizer handles
the specific cases below correctly. They do not prove it handles every
possible input correctly.

Run with:
    python3 test_normalizer.py
"""
import sys
sys.path.append('~/fineval')

from fineval.scorers.normalizer import normalize_and_compare

test_cases = [
    # description, predicted, expected, should_pass

    # explicit percentage handling
    ("53% equals 0.53232",              "53%",                  "0.53232",      True),
    ("53% equals 0.53",                 "53%",                  "0.53",         True),
    ("0.53 equals 53%",                 "0.53",                 "53%",          True),

    # implicit percentage detection (Week 6)
    # models often answer percentages without the percent sign
    ("53 equals 0.53 (implicit pct)",   "53",                   "0.53",         True),
    ("53.2 equals 0.53232",             "53.2",                 "0.53232",      True),
    ("-3.22 equals -0.03219",           "-3.22",                "-0.03219",     True),
    ("63.6 equals 0.63634",             "63.6",                 "0.63634",      True),
    ("28 equals 0.27743",               "28",                   "0.27743",      True),
    ("14.14 equals 0.14136",            "14.14",                "0.14136",      True),
    ("-32.82 equals -0.3282",           "-32.82",               "-0.3282",      True),
    ("12.03 equals 0.12032",            "12.03",                "0.12032",      True),
    ("5 equals 0.05 (genuine match)",   "5",                    "0.05",         True),

    # coincidence guards (Week 6)
    # relative tolerance must reject matches that are close but wrong
    ("99 must NOT equal 0.5",           "99",                   "0.5",          False),
    ("7 must NOT equal 0.05",           "7",                    "0.05",         False),

    # large number tolerance (Week 6)
    ("41932 equals 41932.20339",        "41932",                "41932.20339",  True),
    ("22929 equals 22929.0",            "22929",                "22929.0",      True),
    ("41000 must NOT equal 41932.2",    "41000",                "41932.20339",  False),

    # million scale detection (Week 6)
    ("3800000 equals 3.8 (millions)",   "3800000",              "3.8",          True),

    # rounding tolerance
    ("rounding 0.2774 vs 0.27743",      "0.2774",               "0.27743",      True),

    # currency, commas, and extra words
    ("currency $1,000 vs 1000",         "$1,000",               "1000",         True),
    ("extra words in answer",           "The answer is 1,000.", "1000",         True),
    ("million format",                  "$1.5 million",         "1500000",      True),
    ("comma format",                    "1,500,000",            "1500000",      True),
    ("scientific notation",             "1.5e6",                "1500000",      True),

    # yes/no answers
    ("yes/no: Yes. equals yes",         "Yes.",                 "yes",          True),
    ("yes/no: no vs yes",               "no",                   "yes",          False),

    # plain numbers and signs
    ("standard number 3.8",             "3.8",                  "3.8",          True),
    ("large number 206588",             "206588",               "206588.0",     True),
    ("negative number -2620",           "-2620",                "-2620.0",      True),
    ("wrong sign -6.9 vs 6.9",          "-6.9",                 "6.9",          False),
]

print("Normalizer test cases:\n")
all_passed = True
passed_count = 0
failed_count = 0

for desc, predicted, expected, should_pass in test_cases:
    score, is_correct = normalize_and_compare(predicted, expected)
    passed = is_correct == should_pass
    if passed:
        passed_count += 1
    else:
        failed_count += 1
        all_passed = False
    status = "PASS" if passed else "FAIL"
    print(f"{status} {desc}")
    print(f"   predicted={predicted!r} expected={expected!r}")
    print(f"   score={score:.2f} is_correct={is_correct} "
          f"should_pass={should_pass}")
    print()

print("=" * 50)
print(f"Passed: {passed_count}   Failed: {failed_count}   "
      f"Total: {len(test_cases)}")
print("All tests passed." if all_passed else "Some tests FAILED.")
print("=" * 50)
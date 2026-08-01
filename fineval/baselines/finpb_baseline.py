"""
finpb_baseline.py
Classical ML baselines for Financial PhraseBank sentiment classification.
Trains and evaluates:
1. TF-IDF + Logistic Regression
2. TF-IDF + XGBoost
Saves results as JSON so the leaderboard picks them up automatically.
"""
import sys
import os
import json
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

from fineval.datasets.finpb_loader import load_finpb


def load_finpb_data():
    """
    Load Financial PhraseBank and return texts and labels.
    Uses the full dataset for training and testing.
    """
    items = load_finpb()
    texts = [item.input["text"] for item in items]
    labels = [item.expected_output for item in items]
    return texts, labels


def encode_labels(labels: list[str]) -> tuple[list[int], dict]:
    """
    Convert string labels to integers for XGBoost.
    Returns (encoded_labels, label_map).
    """
    unique = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique)}
    encoded = [label_map[label] for label in labels]
    return encoded, label_map


def decode_labels(encoded: list[int], label_map: dict) -> list[str]:
    """Convert integer labels back to strings."""
    reverse_map = {v: k for k, v in label_map.items()}
    return [reverse_map[i] for i in encoded]


def save_baseline_json(model_name, accuracy, f1, n_items, path):
    """
    Save baseline results as JSON in the same format as LLM results
    so the leaderboard picks them up automatically. Synthesizes n_items
    rows matching the aggregate accuracy, and stores the real F1 in
    f1_override so the leaderboard displays it.
    """
    n_correct = round(accuracy * n_items)
    data = []
    for i in range(n_items):
        is_correct = i < n_correct
        data.append({
            "item_id": f"{model_name}-{i}",
            "dataset_name": "finpb",
            "model_id": model_name,
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "expected": "",
            "ai_answer": "",
            "latency_ms": 0,
            "cost_cents": 0.0,
            "timestamp": str(datetime.now()),
            "f1_override": f1,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Baseline results saved to {path}")


def run_logistic_regression(
    X_train, X_test, y_train, y_test
) -> dict:
    """Train and evaluate TF-IDF + Logistic Regression."""
    print("Training TF-IDF + Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print()
    print(classification_report(y_test, predictions))

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "f1": f1,
        "predictions": predictions.tolist()
    }


def run_xgboost(
    X_train, X_test, y_train_enc, y_test_enc, label_map
) -> dict:
    """Train and evaluate TF-IDF + XGBoost."""
    print("Training TF-IDF + XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train, y_train_enc)
    predictions_enc = model.predict(X_test)
    predictions = decode_labels(predictions_enc.tolist(), label_map)
    y_test_labels = decode_labels(y_test_enc, label_map)

    accuracy = accuracy_score(y_test_labels, predictions)
    f1 = f1_score(y_test_labels, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print()
    print(classification_report(y_test_labels, predictions))

    return {
        "model": "TF-IDF + XGBoost",
        "accuracy": accuracy,
        "f1": f1,
        "predictions": predictions
    }


def run_baselines():
    """Run both baselines and print a comparison summary."""
    print("=" * 60)
    print("Financial PhraseBank — Classical ML Baselines")
    print("=" * 60)
    print()

    # load data
    print("Loading Financial PhraseBank...")
    texts, labels = load_finpb_data()
    print(f"Total items: {len(texts)}")
    print(f"Label distribution:")
    for label in sorted(set(labels)):
        count = labels.count(label)
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    print()

    # train test split — 80% train, 20% test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train set: {len(X_train_raw)} items")
    print(f"Test set:  {len(X_test_raw)} items")
    print()

    # TF-IDF vectorization
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print()

    # encode labels for XGBoost
    y_train_enc, label_map = encode_labels(y_train)
    y_test_enc, _ = encode_labels(y_test)

    # run baselines
    print("=" * 60)
    print("BASELINE 1: TF-IDF + Logistic Regression")
    print("=" * 60)
    lr_results = run_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    print("=" * 60)
    print("BASELINE 2: TF-IDF + XGBoost")
    print("=" * 60)
    xgb_results = run_xgboost(
        X_train, X_test, y_train_enc, y_test_enc, label_map
    )

    # save baseline results as JSON so the leaderboard picks them up
    n_test = len(y_test)
    save_baseline_json(
        "TF-IDF + Logistic Regression",
        lr_results["accuracy"],
        lr_results["f1"],
        n_test,
        "results/baseline-logreg.json"
    )
    save_baseline_json(
        "TF-IDF + XGBoost",
        xgb_results["accuracy"],
        xgb_results["f1"],
        n_test,
        "results/baseline-xgboost.json"
    )

    # comparison summary
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 60)
    print(f"{'TF-IDF + Logistic Regression':<40} "
          f"{lr_results['accuracy']*100:>9.2f}% "
          f"{lr_results['f1']:>10.4f}")
    print(f"{'TF-IDF + XGBoost':<40} "
          f"{xgb_results['accuracy']*100:>9.2f}% "
          f"{xgb_results['f1']:>10.4f}")
    print("=" * 60)
    print()
    print("Baseline results are from the full dataset test split.")


if __name__ == "__main__":
    run_baselines()
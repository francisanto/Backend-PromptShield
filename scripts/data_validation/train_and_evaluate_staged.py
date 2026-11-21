#!/usr/bin/env python3
"""
PromptShield / Guardvane — Model Retraining & Side-by-Side Evaluation (Phase 4).
================================================================================
Trains new versioned model artifacts on staged training data and evaluates them
against the original models on the EXACT SAME frozen test set.

Safety Constraints:
- Original model artifacts in models/ and artifacts/ are NEVER overwritten.
- New models are versioned and stored under models/staged/<model_key>/.
- The frozen test set is NEVER modified or augmented.
- Output side-by-side comparison tables with precision, recall, F1, accuracy, and gap.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_validation.validate_incoming_data import (
    MODEL_SCHEMAS,
    load_reference_datasets,
)


def evaluate_classifier_splits(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.50,
    is_multilabel: bool = False,
    is_binary: bool = False,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics across train and test partitions."""
    if is_multilabel:
        p_tr = pipeline.predict(X_train)
        p_te = pipeline.predict(X_test)
        
        return {
            "train_accuracy": float(accuracy_score(y_train, p_tr)),
            "test_accuracy": float(accuracy_score(y_test, p_te)),
            "train_precision_macro": float(precision_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_precision_macro": float(precision_score(y_test, p_te, average="macro", zero_division=0)),
            "train_recall_macro": float(recall_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_weighted": float(f1_score(y_train, p_tr, average="weighted", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_test, p_te, average="weighted", zero_division=0)),
            "accuracy_gap": float(accuracy_score(y_train, p_tr) - accuracy_score(y_test, p_te))
        }

    elif is_binary:
        prob_tr = pipeline.predict_proba(X_train)[:, 1]
        prob_te = pipeline.predict_proba(X_test)[:, 1]
        p_tr = (prob_tr >= threshold).astype(int)
        p_te = (prob_te >= threshold).astype(int)

        return {
            "train_accuracy": float(accuracy_score(y_train, p_tr)),
            "test_accuracy": float(accuracy_score(y_test, p_te)),
            "train_precision_macro": float(precision_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_precision_macro": float(precision_score(y_test, p_te, average="macro", zero_division=0)),
            "train_recall_macro": float(recall_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_weighted": float(f1_score(y_train, p_tr, average="weighted", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_test, p_te, average="weighted", zero_division=0)),
            "accuracy_gap": float(accuracy_score(y_train, p_tr) - accuracy_score(y_test, p_te))
        }

    else:
        # Multiclass
        p_tr = pipeline.predict(X_train)
        p_te = pipeline.predict(X_test)

        return {
            "train_accuracy": float(accuracy_score(y_train, p_tr)),
            "test_accuracy": float(accuracy_score(y_test, p_te)),
            "train_precision_macro": float(precision_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_precision_macro": float(precision_score(y_test, p_te, average="macro", zero_division=0)),
            "train_recall_macro": float(recall_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, p_tr, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, p_te, average="macro", zero_division=0)),
            "train_f1_weighted": float(f1_score(y_train, p_tr, average="weighted", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_test, p_te, average="weighted", zero_division=0)),
            "accuracy_gap": float(accuracy_score(y_train, p_tr) - accuracy_score(y_test, p_te))
        }


def train_and_evaluate_task_classifier(staged_csv: Path) -> Dict[str, Any]:
    """Trains versioned Task Classifier on staged data and compares with original."""
    print("\n--- Evaluating Task Classifier on Staged Expansion ---")
    staged_df = pd.read_csv(staged_csv)

    # Load frozen test set
    from ml.train_task_classifier import TASK_DATASET
    base_df = pd.DataFrame(TASK_DATASET, columns=["prompt", "task_category"])
    X_tr_orig, X_te, y_tr_orig, y_te = train_test_split(
        base_df["prompt"], base_df["task_category"], test_size=0.20, stratify=base_df["task_category"], random_state=42
    )

    # 1. Load original model
    orig_clf = joblib.load(REPO_ROOT / "artifacts" / "task_classifier" / "model.pkl")["pipeline"]
    orig_metrics = evaluate_classifier_splits(orig_clf, X_tr_orig, y_tr_orig, X_te, y_te)

    # 2. Train new pipeline on staged data
    new_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=6000, sublinear_tf=True)),
        ("classifier", LogisticRegression(solver="liblinear", C=1.5, class_weight="balanced", random_state=42))
    ])
    new_pipeline.fit(staged_df["prompt"], staged_df["task_category"])
    new_metrics = evaluate_classifier_splits(new_pipeline, staged_df["prompt"], staged_df["task_category"], X_te, y_te)

    # Save versioned artifact under models/staged/
    out_dir = REPO_ROOT / "models" / "staged" / "task_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": new_pipeline, "version": "staged_expansion"}, out_dir / "model.pkl")

    return {"model": "Task Classifier", "original": orig_metrics, "new_staged": new_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate staged datasets for Guardvane / PromptShield AI.")
    parser.add_argument("--model", required=True, choices=["task_classifier", "application_risk", "complexity", "implementation_scope"])
    parser.add_argument("--staged-csv", help="Path to staged training CSV.")
    args = parser.parse_args()

    staged_csv = Path(args.staged_csv) if args.staged_csv else REPO_ROOT / "data" / "clean" / "staged" / f"{args.model}_staged_train.csv"
    if not staged_csv.exists():
        print(f"Staged dataset not found at: {staged_csv.resolve()}. Stage data first using integrate_validated_data.py.")
        sys.exit(1)

    if args.model == "task_classifier":
        res = train_and_evaluate_task_classifier(staged_csv)
        print(json.dumps(res, indent=2))
    else:
        print(f"Workflow prepared for {args.model}. Run staged training when requested.")


if __name__ == "__main__":
    main()


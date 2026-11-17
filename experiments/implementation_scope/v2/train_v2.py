#!/usr/bin/env python3
"""
Guardvane / PromptShield AI — Implementation Scope Predictor v2 Retraining Pipeline.
===================================================================================
Incorporates:
1. Base training partition: data/splits/train.csv (609 rows)
2. Scope batch 01 clean:    data/clean/implementation_scope_batch_01_clean.csv (50 rows)
3. Auth integration clean:  data/clean/auth_integration_clean.csv (1,000 rows)

Partition Integrity:
- Validation partition (data/splits/val.csv, 129 rows) is used exclusively for tuning.
- Held-out test partition (data/splits/test.csv, 131 rows) is evaluated strictly once.

Outputs:
- Models:      models/implementation_scope/v2/
- Experiments: experiments/implementation_scope/v2/
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("Guardvane / PromptShield Implementation Scope Predictor — Retraining v2")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print("=" * 80)

    # 1. Directories
    models_dir = REPO_ROOT / "models" / "implementation_scope" / "v2"
    exp_dir = REPO_ROOT / "experiments" / "implementation_scope" / "v2"
    models_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Partitions and Clean Datasets
    train_path = REPO_ROOT / "data" / "splits" / "train.csv"
    val_path = REPO_ROOT / "data" / "splits" / "val.csv"
    test_path = REPO_ROOT / "data" / "splits" / "test.csv"
    scope_clean_path = REPO_ROOT / "data" / "clean" / "implementation_scope_batch_01_clean.csv"
    auth_clean_path = REPO_ROOT / "data" / "clean" / "auth_integration_clean.csv"

    base_train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    scope_clean_df = pd.read_csv(scope_clean_path)
    auth_clean_df = pd.read_csv(auth_clean_path)

    print(f"Base Train Split:    {base_train_df.shape[0]} rows")
    print(f"Scope Batch 01:      {scope_clean_df.shape[0]} rows")
    print(f"Auth Integration:    {auth_clean_df.shape[0]} rows")
    print(f"Validation Split:    {val_df.shape[0]} rows (Frozen)")
    print(f"Test Split:          {test_df.shape[0]} rows (Frozen)")

    # Construct Corpus A (659 rows: base train + scope batch)
    train_scope_df = pd.concat([base_train_df, scope_clean_df], ignore_index=True)

    # Construct Corpus B (1,659 rows: base train + scope batch + auth integration)
    train_auth_df = pd.concat([
        base_train_df[['enhanced_prompt', 'authentication_required', 'external_integration_required']],
        scope_clean_df[['enhanced_prompt', 'authentication_required', 'external_integration_required']],
        auth_clean_df[['enhanced_prompt', 'authentication_required', 'external_integration_required']]
    ], ignore_index=True)

    print(f"\nCorpus A (Scope & Secondary targets): {train_scope_df.shape[0]} rows")
    print(f"Corpus B (Auth & Ext Requirement):    {train_auth_df.shape[0]} rows")

    # 3. Feature Extraction Pipeline (Fitted exclusively on training vocabularies)
    print("\n[Step 1] Fitting unified Feature Extraction Pipeline on Train split only...")
    all_train_prompts = list(train_scope_df['enhanced_prompt']) + list(auth_clean_df['enhanced_prompt'])

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )

    feature_pipeline = FeatureUnion([
        ('word_tfidf', word_vec),
        ('char_tfidf', char_vec)
    ])

    feature_pipeline.fit(all_train_prompts)

    X_train_scope = feature_pipeline.transform(train_scope_df['enhanced_prompt'])
    X_train_auth = feature_pipeline.transform(train_auth_df['enhanced_prompt'])
    X_val = feature_pipeline.transform(val_df['enhanced_prompt'])
    X_test = feature_pipeline.transform(test_df['enhanced_prompt'])

    print(f"Feature Dimension: {X_train_scope.shape[1]} features")
    print(f"X_train_scope shape: {X_train_scope.shape}")
    print(f"X_train_auth shape:  {X_train_auth.shape}")
    print(f"X_val shape:         {X_val.shape}")
    print(f"X_test shape:        {X_test.shape}")

    # 4. Hyperparameter Tuning on Validation Fold
    print("\n[Step 2] Hyperparameter Tuning on Validation Set...")
    validation_tuning_results: Dict[str, Any] = {}

    # 4.1 Primary Scope Tuning
    scope_search = []
    best_scope_f1 = -1.0
    best_scope_cfg: Dict[str, Any] = {}
    best_scope_clf = None

    for solver in ['liblinear']:
        for cw in [None, 'balanced']:
            for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                clf.fit(X_train_scope, train_scope_df['primary_scope'])
                preds = clf.predict(X_val)
                macro_f1 = f1_score(val_df['primary_scope'], preds, average='macro', zero_division=0)
                weighted_f1 = f1_score(val_df['primary_scope'], preds, average='weighted', zero_division=0)
                acc = accuracy_score(val_df['primary_scope'], preds)
                cfg = {
                    'solver': solver,
                    'class_weight': cw,
                    'C': C,
                    'val_macro_f1': round(float(macro_f1), 4),
                    'val_weighted_f1': round(float(weighted_f1), 4),
                    'val_accuracy': round(float(acc), 4)
                }
                scope_search.append(cfg)
                if macro_f1 > best_scope_f1:
                    best_scope_f1 = macro_f1
                    best_scope_cfg = cfg
                    best_scope_clf = clf

    validation_tuning_results['primary_scope'] = {
        'best_config': best_scope_cfg,
        'grid_results': scope_search
    }
    print(f"  Primary Scope: C={best_scope_cfg['C']}, cw={best_scope_cfg['class_weight']} (Val Macro-F1: {best_scope_cfg['val_macro_f1']})")

    # 4.2 Requirements Classifiers Tuning
    req_models: Dict[str, Any] = {}
    validation_tuning_results['requirements'] = {}

    req_targets = [
        'frontend_required',
        'backend_required',
        'database_required',
        'api_required',
        'authentication_required',
        'external_integration_required'
    ]

    for req_name in req_targets:
        # Route to appropriate training matrix
        if req_name in ['authentication_required', 'external_integration_required']:
            X_curr_train = X_train_auth
            y_curr_train = train_auth_df[req_name]
        else:
            X_curr_train = X_train_scope
            y_curr_train = train_scope_df[req_name]

        best_f1 = -1.0
        best_cfg: Dict[str, Any] = {}
        best_clf = None
        best_th = 0.50
        req_grid = []

        for solver in ['liblinear']:
            for cw in [None, 'balanced']:
                for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                    clf.fit(X_curr_train, y_curr_train)
                    probs_val = clf.predict_proba(X_val)[:, 1]

                    for th in np.linspace(0.15, 0.65, 51):
                        th = round(float(th), 2)
                        p_bin = (probs_val >= th).astype(int)
                        f1 = f1_score(val_df[req_name], p_bin, zero_division=0)
                        prec = precision_score(val_df[req_name], p_bin, zero_division=0)
                        rec = recall_score(val_df[req_name], p_bin, zero_division=0)
                        acc = accuracy_score(val_df[req_name], p_bin)
                        brier = brier_score_loss(val_df[req_name], probs_val)

                        cfg = {
                            'solver': solver,
                            'class_weight': cw,
                            'C': C,
                            'threshold': th,
                            'val_f1': round(float(f1), 4),
                            'val_precision': round(float(prec), 4),
                            'val_recall': round(float(rec), 4),
                            'val_accuracy': round(float(acc), 4),
                            'val_brier': round(float(brier), 4)
                        }
                        req_grid.append(cfg)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_cfg = cfg
                            best_clf = clf
                            best_th = th

        validation_tuning_results['requirements'][req_name] = {
            'best_config': best_cfg,
            'grid_sample': req_grid[:5]
        }
        req_models[req_name] = {
            'model': best_clf,
            'threshold': best_th,
            'config': best_cfg
        }
        print(f"  {req_name:<30}: C={best_cfg['C']}, cw={best_cfg['class_weight']}, th={best_th} (Val F1: {best_cfg['val_f1']})")

    # 4.3 Secondary Targets Tuning
    sec_models: Dict[str, Any] = {}
    validation_tuning_results['secondary'] = {}

    for sec_name in ['complexity', 'design_emphasis', 'task_intent']:
        best_sec_f1 = -1.0
        best_sec_cfg: Dict[str, Any] = {}
        best_sec_clf = None
        sec_grid = []

        for solver in ['liblinear']:
            for cw in [None, 'balanced']:
                for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                    clf.fit(X_train_scope, train_scope_df[sec_name])
                    preds = clf.predict(X_val)
                    macro_f1 = f1_score(val_df[sec_name], preds, average='macro', zero_division=0)
                    weighted_f1 = f1_score(val_df[sec_name], preds, average='weighted', zero_division=0)
                    acc = accuracy_score(val_df[sec_name], preds)

                    cfg = {
                        'solver': solver,
                        'class_weight': cw,
                        'C': C,
                        'val_macro_f1': round(float(macro_f1), 4),
                        'val_weighted_f1': round(float(weighted_f1), 4),
                        'val_accuracy': round(float(acc), 4)
                    }
                    sec_grid.append(cfg)
                    if macro_f1 > best_sec_f1:
                        best_sec_f1 = macro_f1
                        best_sec_cfg = cfg
                        best_sec_clf = clf

        validation_tuning_results['secondary'][sec_name] = {
            'best_config': best_sec_cfg,
            'grid_results': sec_grid
        }
        sec_models[sec_name] = {
            'model': best_sec_clf,
            'config': best_sec_cfg
        }
        print(f"  {sec_name:<30}: C={best_sec_cfg['C']}, cw={best_sec_cfg['class_weight']} (Val Macro-F1: {best_sec_cfg['val_macro_f1']})")

    # 5. Held-Out Test Split Evaluation (Strictly Once)
    print("\n[Step 3] Evaluating Final Models on Held-Out Test Set (131 rows)...")
    test_metrics: Dict[str, Any] = {}

    # Primary Scope Test
    test_preds_scope = best_scope_clf.predict(X_test)
    test_metrics['primary_scope'] = {
        'accuracy': round(float(accuracy_score(test_df['primary_scope'], test_preds_scope)), 4),
        'macro_f1': round(float(f1_score(test_df['primary_scope'], test_preds_scope, average='macro', zero_division=0)), 4),
        'weighted_f1': round(float(f1_score(test_df['primary_scope'], test_preds_scope, average='weighted', zero_division=0)), 4),
        'classes': list(best_scope_clf.classes_),
        'per_class': {}
    }
    rep_scope = classification_report(test_df['primary_scope'], test_preds_scope, output_dict=True, zero_division=0)
    for c in best_scope_clf.classes_:
        if c in rep_scope:
            test_metrics['primary_scope']['per_class'][c] = {
                'precision': round(float(rep_scope[c]['precision']), 4),
                'recall': round(float(rep_scope[c]['recall']), 4),
                'f1': round(float(rep_scope[c]['f1-score']), 4),
                'support': int(rep_scope[c]['support'])
            }

    print(f"\nPrimary Scope Test Accuracy:    {test_metrics['primary_scope']['accuracy']:.2%}")
    print(f"Primary Scope Test Weighted F1: {test_metrics['primary_scope']['weighted_f1']:.4f}")
    print(f"Primary Scope Test Macro F1:    {test_metrics['primary_scope']['macro_f1']:.4f}")

    # Requirements Test
    test_metrics['requirements'] = {}
    for req_name in req_targets:
        entry = req_models[req_name]
        clf = entry['model']
        th = entry['threshold']
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= th).astype(int)
        y_true = test_df[req_name].astype(int)

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        test_metrics['requirements'][req_name] = {
            'threshold': th,
            'accuracy': round(float(accuracy_score(y_true, preds)), 4),
            'f1': round(float(f1_score(y_true, preds, zero_division=0)), 4),
            'precision': round(float(precision_score(y_true, preds, zero_division=0)), 4),
            'recall': round(float(recall_score(y_true, preds, zero_division=0)), 4),
            'brier_score': round(float(brier_score_loss(y_true, probs)), 4),
            'positive_support': int(y_true.sum()),
            'negative_support': int(len(y_true) - y_true.sum()),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        print(f"  {req_name:<30}: Acc={test_metrics['requirements'][req_name]['accuracy']:.2%}, Prec={test_metrics['requirements'][req_name]['precision']:.4f}, Rec={test_metrics['requirements'][req_name]['recall']:.4f}, F1={test_metrics['requirements'][req_name]['f1']:.4f}")

    # Secondary Test
    test_metrics['secondary'] = {}
    for sec_name in ['complexity', 'design_emphasis', 'task_intent']:
        entry = sec_models[sec_name]
        clf = entry['model']
        preds = clf.predict(X_test)
        y_true = test_df[sec_name]

        rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
        test_metrics['secondary'][sec_name] = {
            'accuracy': round(float(accuracy_score(y_true, preds)), 4),
            'macro_f1': round(float(f1_score(y_true, preds, average='macro', zero_division=0)), 4),
            'weighted_f1': round(float(f1_score(y_true, preds, average='weighted', zero_division=0)), 4),
            'classes': list(clf.classes_),
            'per_class': {}
        }
        for c in clf.classes_:
            if c in rep:
                test_metrics['secondary'][sec_name]['per_class'][c] = {
                    'precision': round(float(rep[c]['precision']), 4),
                    'recall': round(float(rep[c]['recall']), 4),
                    'f1': round(float(rep[c]['f1-score']), 4),
                    'support': int(rep[c]['support'])
                }
        print(f"  {sec_name:<30}: Acc={test_metrics['secondary'][sec_name]['accuracy']:.2%}, Macro F1={test_metrics['secondary'][sec_name]['macro_f1']:.4f}, Weighted F1={test_metrics['secondary'][sec_name]['weighted_f1']:.4f}")

    # 6. Save Model Artifacts
    print("\n[Step 4] Saving Retrained Model Artifacts to models/implementation_scope/v2/...")
    pipe_path = models_dir / "feature_pipeline.joblib"
    scope_path = models_dir / "primary_scope_model.joblib"
    req_path = models_dir / "requirement_models.joblib"
    sec_path = models_dir / "secondary_models.joblib"
    meta_path = models_dir / "model_metadata.json"

    joblib.dump(feature_pipeline, pipe_path)
    joblib.dump(best_scope_clf, scope_path)
    joblib.dump(req_models, req_path)
    joblib.dump(sec_models, sec_path)

    metadata = {
        'model_version': 'v2.0-guardvane',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'environment': {
            'scikit_learn_version': sklearn.__version__,
            'joblib_version': joblib.__version__,
            'python_version': sys.version
        },
        'training_data': {
            'corpus_a_rows': len(train_scope_df),
            'corpus_b_rows': len(train_auth_df),
            'validation_rows': len(val_df),
            'test_rows': len(test_df),
            'new_scope_records': 50,
            'new_auth_records': 1000
        },
        'feature_extraction': {
            'word_tfidf': {'ngram_range': [1, 2], 'min_df': 2, 'max_features': 5000, 'sublinear_tf': True},
            'char_tfidf': {'ngram_range': [3, 5], 'analyzer': 'char', 'min_df': 2, 'max_features': 10000, 'sublinear_tf': True},
            'total_features': X_train_scope.shape[1]
        },
        'primary_scope_model': {
            'algorithm': 'LogisticRegression',
            'classes': list(best_scope_clf.classes_),
            'config': best_scope_cfg
        },
        'requirement_models': {
            name: {
                'algorithm': 'LogisticRegression',
                'threshold': entry['threshold'],
                'config': entry['config']
            }
            for name, entry in req_models.items()
        },
        'secondary_models': {
            name: {
                'algorithm': 'LogisticRegression',
                'classes': list(entry['model'].classes_),
                'config': entry['config']
            }
            for name, entry in sec_models.items()
        },
        'file_hashes': {
            'feature_pipeline.joblib': sha256_file(str(pipe_path)),
            'primary_scope_model.joblib': sha256_file(str(scope_path)),
            'requirement_models.joblib': sha256_file(str(req_path)),
            'secondary_models.joblib': sha256_file(str(sec_path))
        }
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # 7. Save Experiment Reports
    with open(exp_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    with open(exp_dir / "validation_tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump(validation_tuning_results, f, indent=2)
    with open(exp_dir / "test_evaluation_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)

    print("\n[SUCCESS] Retraining v2 completed successfully.")
    print(f"Artifacts saved to {models_dir.resolve()}")
    print(f"Metrics saved to   {exp_dir.resolve()}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Implementation Scope Predictor - Baseline ML Experiment (v1)

This script trains and evaluates the baseline Logistic Regression model suite
for the Implementation Scope Predictor using the frozen v1 dataset splits:
- train.csv: 609 rows (70.1%)
- val.csv:   129 rows (14.8%)
- test.csv:  131 rows (15.1%)

Features:
- Word TF-IDF (1, 2 n-grams, min_df=2, max_features=5000, sublinear_tf=True)
- Char TF-IDF (3, 5 n-grams, min_df=2, max_features=10000, sublinear_tf=True)
- Fit on train.csv only.

Targets:
1. Primary Scope: multiclass ('backend', 'frontend', 'full_stack', 'design_only', 'security_only', 'unclear')
2. 6 Requirement Flags: binary ('frontend_required', 'backend_required', 'database_required',
                                'api_required', 'authentication_required', 'external_integration_required')
3. Secondary Targets: multiclass ('complexity', 'design_emphasis', 'task_intent')

Model selection and decision thresholds are tuned exclusively on validation data.
The test set is evaluated only once after the approach is finalized.
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("PromptShield Implementation Scope Predictor — Baseline ML Experiment v1")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print("=" * 70)

    # 1. Paths
    data_dir = "datasets/implementation_scope/experiments/v1"
    models_dir = "models/implementation_scope/v1"
    exp_dir = "experiments/implementation_scope/v1"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded Train: {train_df.shape} ({len(train_df)} rows)")
    print(f"Loaded Val:   {val_df.shape} ({len(val_df)} rows)")
    print(f"Loaded Test:  {test_df.shape} ({len(test_df)} rows)")

    # 2. Feature Extraction (TF-IDF FeatureUnion)
    print("\n[Step 1] Fitting Feature Extraction Pipeline on Train split only...")
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

    X_train = feature_pipeline.fit_transform(train_df['enhanced_prompt'])
    X_val = feature_pipeline.transform(val_df['enhanced_prompt'])
    X_test = feature_pipeline.transform(test_df['enhanced_prompt'])

    print(f"Feature Matrix Shape (Train): {X_train.shape}")
    print(f"Feature Matrix Shape (Val):   {X_val.shape}")
    print(f"Feature Matrix Shape (Test):  {X_test.shape}")

    # 3. Model Hyperparameter Tuning on Validation Data
    print("\n[Step 2] Hyperparameter Tuning on Validation Set...")
    validation_tuning_results = {}

    # Primary Scope Grid Search
    scope_search = []
    for solver in ['liblinear']:
        for cw in [None, 'balanced']:
            for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                clf.fit(X_train, train_df['primary_scope'])
                preds = clf.predict(X_val)
                macro_f1 = f1_score(val_df['primary_scope'], preds, average='macro', zero_division=0)
                weighted_f1 = f1_score(val_df['primary_scope'], preds, average='weighted', zero_division=0)
                acc = accuracy_score(val_df['primary_scope'], preds)
                scope_search.append({
                    'solver': solver,
                    'class_weight': cw,
                    'C': C,
                    'val_macro_f1': round(float(macro_f1), 4),
                    'val_weighted_f1': round(float(weighted_f1), 4),
                    'val_accuracy': round(float(acc), 4)
                })

    scope_search_df = pd.DataFrame(scope_search).sort_values(by='val_macro_f1', ascending=False)
    best_scope_cfg = scope_search_df.iloc[0].to_dict()
    validation_tuning_results['primary_scope'] = {
        'best_config': best_scope_cfg,
        'grid_results': scope_search
    }
    print(f"Primary Scope selected: C={best_scope_cfg['C']}, cw={best_scope_cfg['class_weight']} (Val Macro-F1: {best_scope_cfg['val_macro_f1']})")

    # Requirement Flags Grid Search (C, class_weight, threshold)
    req_targets = [
        'frontend_required',
        'backend_required',
        'database_required',
        'api_required',
        'authentication_required',
        'external_integration_required'
    ]
    validation_tuning_results['requirements'] = {}
    best_req_configs = {}

    for target in req_targets:
        y_tr = train_df[target].values
        y_va = val_df[target].values
        target_grid = []

        for solver in ['liblinear']:
            for cw in [None, 'balanced']:
                for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                    clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                    clf.fit(X_train, y_tr)
                    probs = clf.predict_proba(X_val)[:, 1]

                    for th in np.arange(0.20, 0.75, 0.05):
                        preds = (probs >= th).astype(int)
                        f1 = f1_score(y_va, preds, zero_division=0)
                        prec = precision_score(y_va, preds, zero_division=0)
                        rec = recall_score(y_va, preds, zero_division=0)
                        acc = accuracy_score(y_va, preds)
                        brier = brier_score_loss(y_va, probs)
                        target_grid.append({
                            'solver': solver,
                            'class_weight': cw,
                            'C': C,
                            'threshold': round(float(th), 2),
                            'val_f1': round(float(f1), 4),
                            'val_precision': round(float(prec), 4),
                            'val_recall': round(float(rec), 4),
                            'val_accuracy': round(float(acc), 4),
                            'val_brier': round(float(brier), 4)
                        })

        df_grid = pd.DataFrame(target_grid).sort_values(by=['val_f1', 'val_accuracy'], ascending=False)
        best_cfg = df_grid.iloc[0].to_dict()
        best_req_configs[target] = best_cfg
        validation_tuning_results['requirements'][target] = {
            'best_config': best_cfg,
            'top_5_configs': df_grid.head(5).to_dict(orient='records')
        }
        print(f"Req '{target}' selected: C={best_cfg['C']}, cw={best_cfg['class_weight']}, th={best_cfg['threshold']} (Val F1: {best_cfg['val_f1']}, Rec: {best_cfg['val_recall']}, Prec: {best_cfg['val_precision']})")

    # Secondary Targets Grid Search
    secondary_targets = ['complexity', 'design_emphasis', 'task_intent']
    validation_tuning_results['secondary'] = {}
    best_sec_configs = {}

    for target in secondary_targets:
        y_tr = train_df[target].values
        y_va = val_df[target].values
        sec_grid = []

        for solver in ['liblinear']:
            for cw in [None, 'balanced']:
                for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                    clf = LogisticRegression(solver=solver, C=C, class_weight=cw, max_iter=1000, random_state=42)
                    clf.fit(X_train, y_tr)
                    preds = clf.predict(X_val)
                    macro_f1 = f1_score(y_va, preds, average='macro', zero_division=0)
                    weighted_f1 = f1_score(y_va, preds, average='weighted', zero_division=0)
                    acc = accuracy_score(y_va, preds)
                    sec_grid.append({
                        'solver': solver,
                        'class_weight': cw,
                        'C': C,
                        'val_macro_f1': round(float(macro_f1), 4),
                        'val_weighted_f1': round(float(weighted_f1), 4),
                        'val_accuracy': round(float(acc), 4)
                    })

        df_sec = pd.DataFrame(sec_grid).sort_values(by='val_macro_f1', ascending=False)
        best_cfg = df_sec.iloc[0].to_dict()
        best_sec_configs[target] = best_cfg
        validation_tuning_results['secondary'][target] = {
            'best_config': best_cfg,
            'grid_results': sec_grid
        }
        print(f"Secondary '{target}' selected: C={best_cfg['C']}, cw={best_cfg['class_weight']} (Val Macro-F1: {best_cfg['val_macro_f1']}, Acc: {best_cfg['val_accuracy']})")

    # Save validation tuning results
    val_results_path = os.path.join(exp_dir, "validation_tuning_results.json")
    with open(val_results_path, 'w') as f:
        json.dump(validation_tuning_results, f, indent=2)
    print(f"\nSaved validation tuning results to {val_results_path}")

    # 4. Final Model Training & Serialization
    print("\n[Step 3] Training and Serializing Final Models...")

    # Primary Scope Model
    primary_scope_clf = LogisticRegression(
        solver=best_scope_cfg['solver'],
        C=best_scope_cfg['C'],
        class_weight=best_scope_cfg['class_weight'],
        max_iter=1000,
        random_state=42
    )
    primary_scope_clf.fit(X_train, train_df['primary_scope'])

    # Requirement Models
    requirement_models = {}
    for target in req_targets:
        cfg = best_req_configs[target]
        clf = LogisticRegression(
            solver=cfg['solver'],
            C=cfg['C'],
            class_weight=cfg['class_weight'],
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train, train_df[target])
        requirement_models[target] = {
            'model': clf,
            'threshold': cfg['threshold'],
            'config': cfg
        }

    # Secondary Models
    secondary_models = {}
    for target in secondary_targets:
        cfg = best_sec_configs[target]
        clf = LogisticRegression(
            solver=cfg['solver'],
            C=cfg['C'],
            class_weight=cfg['class_weight'],
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train, train_df[target])
        secondary_models[target] = {
            'model': clf,
            'config': cfg
        }

    # Save to models/implementation_scope/v1/
    pipeline_save_path = os.path.join(models_dir, "feature_pipeline.joblib")
    primary_scope_save_path = os.path.join(models_dir, "primary_scope_model.joblib")
    req_models_save_path = os.path.join(models_dir, "requirement_models.joblib")
    sec_models_save_path = os.path.join(models_dir, "secondary_models.joblib")
    metadata_save_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(feature_pipeline, pipeline_save_path)
    joblib.dump(primary_scope_clf, primary_scope_save_path)
    joblib.dump(requirement_models, req_models_save_path)
    joblib.dump(secondary_models, sec_models_save_path)

    model_metadata = {
        "model_version": "v1.0-baseline",
        "created_at": "2026-11-14T17:30:00+05:30",
        "environment": {
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "python_version": sys.version
        },
        "feature_extraction": {
            "word_tfidf": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_features": 5000,
                "sublinear_tf": True
            },
            "char_tfidf": {
                "ngram_range": [3, 5],
                "analyzer": "char",
                "min_df": 2,
                "max_features": 10000,
                "sublinear_tf": True
            },
            "total_features": int(X_train.shape[1])
        },
        "primary_scope_model": {
            "algorithm": "LogisticRegression",
            "classes": primary_scope_clf.classes_.tolist(),
            "config": best_scope_cfg
        },
        "requirement_models": {
            target: {
                "algorithm": "LogisticRegression",
                "threshold": best_req_configs[target]['threshold'],
                "config": best_req_configs[target]
            } for target in req_targets
        },
        "secondary_models": {
            target: {
                "algorithm": "LogisticRegression",
                "classes": secondary_models[target]['model'].classes_.tolist(),
                "config": best_sec_configs[target]
            } for target in secondary_targets
        },
        "file_hashes": {
            "feature_pipeline.joblib": sha256_file(pipeline_save_path),
            "primary_scope_model.joblib": sha256_file(primary_scope_save_path),
            "requirement_models.joblib": sha256_file(req_models_save_path),
            "secondary_models.joblib": sha256_file(sec_models_save_path)
        }
    }

    with open(metadata_save_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Saved model artifacts and metadata to {models_dir}")

    # 5. One-Time Test Evaluation
    print("\n[Step 4] Running One-Time Test Set Evaluation...")
    test_metrics = {}

    # 5.1 Primary Scope Test Evaluation
    test_scope_preds = primary_scope_clf.predict(X_test)
    test_scope_probs = primary_scope_clf.predict_proba(X_test)
    test_scope_classes = primary_scope_clf.classes_.tolist()

    scope_report = classification_report(
        test_df['primary_scope'],
        test_scope_preds,
        output_dict=True,
        zero_division=0
    )
    scope_cm = confusion_matrix(test_df['primary_scope'], test_scope_preds, labels=test_scope_classes).tolist()

    # Find false positives / negatives for primary_scope
    scope_errors = []
    for idx, (true_lbl, pred_lbl, prompt, row_id) in enumerate(zip(
        test_df['primary_scope'],
        test_scope_preds,
        test_df['enhanced_prompt'],
        test_df['row_id']
    )):
        if true_lbl != pred_lbl:
            scope_errors.append({
                'row_id': int(row_id),
                'true_label': str(true_lbl),
                'predicted_label': str(pred_lbl),
                'confidence': round(float(np.max(test_scope_probs[idx])), 4),
                'prompt': str(prompt)
            })

    test_metrics['primary_scope'] = {
        'accuracy': round(float(accuracy_score(test_df['primary_scope'], test_scope_preds)), 4),
        'macro_f1': round(float(f1_score(test_df['primary_scope'], test_scope_preds, average='macro', zero_division=0)), 4),
        'weighted_f1': round(float(f1_score(test_df['primary_scope'], test_scope_preds, average='weighted', zero_division=0)), 4),
        'classes': test_scope_classes,
        'per_class': {
            cls: {
                'precision': round(float(scope_report[cls]['precision']), 4),
                'recall': round(float(scope_report[cls]['recall']), 4),
                'f1': round(float(scope_report[cls]['f1-score']), 4),
                'support': int(scope_report[cls]['support'])
            } for cls in test_scope_classes if cls in scope_report
        },
        'confusion_matrix': scope_cm,
        'total_errors': len(scope_errors),
        'sample_errors': scope_errors[:5]
    }
    print(f"Test Primary Scope: Accuracy={test_metrics['primary_scope']['accuracy']}, Macro-F1={test_metrics['primary_scope']['macro_f1']}, Weighted-F1={test_metrics['primary_scope']['weighted_f1']}")

    # 5.2 Requirement Flags Test Evaluation
    test_metrics['requirements'] = {}
    for target in req_targets:
        clf = requirement_models[target]['model']
        th = requirement_models[target]['threshold']
        y_test = test_df[target].values

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= th).astype(int)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        brier = brier_score_loss(y_test, probs)
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()

        # Find False Positives and False Negatives
        fp_examples = []
        fn_examples = []
        for idx, (t, p, prob, prompt, row_id) in enumerate(zip(
            y_test, preds, probs, test_df['enhanced_prompt'], test_df['row_id']
        )):
            if t == 0 and p == 1:
                fp_examples.append({
                    'row_id': int(row_id),
                    'predicted_prob': round(float(prob), 4),
                    'prompt': str(prompt)
                })
            elif t == 1 and p == 0:
                fn_examples.append({
                    'row_id': int(row_id),
                    'predicted_prob': round(float(prob), 4),
                    'prompt': str(prompt)
                })

        test_metrics['requirements'][target] = {
            'threshold': th,
            'accuracy': round(float(acc), 4),
            'f1': round(float(f1), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'brier_score': round(float(brier), 4),
            'positive_support': int(np.sum(y_test == 1)),
            'negative_support': int(np.sum(y_test == 0)),
            'confusion_matrix': {
                'tn': cm[0][0],
                'fp': cm[0][1],
                'fn': cm[1][0],
                'tp': cm[1][1]
            },
            'sample_false_positives': fp_examples[:3],
            'sample_false_negatives': fn_examples[:3]
        }
        print(f"Test Req '{target}': F1={test_metrics['requirements'][target]['f1']}, Prec={test_metrics['requirements'][target]['precision']}, Rec={test_metrics['requirements'][target]['recall']}, Acc={test_metrics['requirements'][target]['accuracy']}, Brier={test_metrics['requirements'][target]['brier_score']}")

    # 5.3 Secondary Targets Test Evaluation
    test_metrics['secondary'] = {}
    for target in secondary_targets:
        clf = secondary_models[target]['model']
        y_test = test_df[target].values
        classes = clf.classes_.tolist()

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=classes).tolist()

        sec_errors = []
        for idx, (t, p, prompt, row_id) in enumerate(zip(
            y_test, preds, test_df['enhanced_prompt'], test_df['row_id']
        )):
            if t != p:
                sec_errors.append({
                    'row_id': int(row_id),
                    'true_label': str(t),
                    'predicted_label': str(p),
                    'confidence': round(float(np.max(probs[idx])), 4),
                    'prompt': str(prompt)
                })

        test_metrics['secondary'][target] = {
            'accuracy': round(float(acc), 4),
            'macro_f1': round(float(macro_f1), 4),
            'weighted_f1': round(float(weighted_f1), 4),
            'classes': classes,
            'per_class': {
                cls: {
                    'precision': round(float(report[cls]['precision']), 4),
                    'recall': round(float(report[cls]['recall']), 4),
                    'f1': round(float(report[cls]['f1-score']), 4),
                    'support': int(report[cls]['support'])
                } for cls in classes if cls in report
            },
            'confusion_matrix': cm,
            'total_errors': len(sec_errors),
            'sample_errors': sec_errors[:3]
        }
        print(f"Test Secondary '{target}': Accuracy={test_metrics['secondary'][target]['accuracy']}, Macro-F1={test_metrics['secondary'][target]['macro_f1']}, Weighted-F1={test_metrics['secondary'][target]['weighted_f1']}")

    # Save test metrics
    test_metrics_path = os.path.join(exp_dir, "test_evaluation_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved test evaluation metrics to {test_metrics_path}")

    # Save experiment config
    exp_config = {
        "experiment_name": "baseline_lr_tfidf_v1",
        "dataset_version": "v1.0-frozen",
        "dataset_paths": {
            "train": train_path,
            "val": val_path,
            "test": test_path
        },
        "feature_extraction": {
            "word_vectorizer": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_features": 5000,
                "sublinear_tf": True
            },
            "char_vectorizer": {
                "ngram_range": [3, 5],
                "analyzer": "char",
                "min_df": 2,
                "max_features": 10000,
                "sublinear_tf": True
            }
        },
        "models": {
            "primary_scope": best_scope_cfg,
            "requirements": best_req_configs,
            "secondary": best_sec_configs
        }
    }
    config_path = os.path.join(exp_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    print(f"Saved experiment config to {config_path}")

    print("\n[Done] Baseline experiment v1 completed successfully.")

if __name__ == '__main__':
    main()


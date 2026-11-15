# Repository Architecture & Directory Guide

This document describes the directory structure of the repository, detailing the specific role and purpose of each top-level folder and its subdirectories.

---

## Directory Overview

| Directory | Primary Role & Use |
| :--- | :--- |
| `analyzers/` | Core analysis engines: prompt security, vulnerability analysis, scope prediction, model & IDE comparison, and prompt optimization. |
| `data/` | Canonical storage for all raw, cleaned, split, review, and manifest datasets. |
| `models/` | Production deployment model artifacts, vectorizers, decision matrices, and schemas. |
| `artifacts/` | Training run checkpoints, metadata logs, and evaluation metrics generated during model training. |
| `ml/` | Training and standalone inference scripts for security risk and prompt injection models. |
| `training/` | Training scripts and unified system evaluation harness for all core ML classifiers. |
| `experiments/` | Machine learning experiment tracking, baseline documentation, and error analysis logs. |
| `optimizer/` | Modular rule definitions and prompt templates for deterministic prompt hardening. |
| `scripts/` | Data engineering, dataset cleaning, deduplication, and label audit utilities. |
| `tests/` | Pytest test suite covering unit, integration, benchmark, and end-to-end regression tests. |
| `docs/` | Technical design documentation, training methodologies, and research references. |
| `documentation/` | Supporting system documentation and specifications. |
| `reports/` | Generated evaluation reports, benchmark audits, and experiment findings. |
| `datasets/` | Backward-compatibility symlinks referencing canonical paths in `data/` with zero byte duplication. |

---

## Detailed Folder Guide

### 1. `analyzers/`
Houses the analytical components of PromptShield AI that inspect, classify, evaluate, and optimize developer prompts.
* `analyzers/prompt_analyzer.py`: Main entry point orchestrating prompt analysis across prompt security, application vulnerability detection, and sensitive data checks.
* `analyzers/prompt_security_analyzer.py`: Machine-learning classifier evaluating prompt injection, jailbreaks, and adversarial attacks with calibrated probabilities.
* `analyzers/implementation_scope_predictor.py`: Multi-target ML inference engine predicting primary development scope, 6 architectural requirement flags, complexity, design emphasis, and task intent.
* `analyzers/prompt_optimizer.py`: Deterministic prompt hardening engine that preserves explicitly declared security controls and injects missing defensive constraints.
* `analyzers/sensitive_data_detector.py`: Deterministic regex and Shannon entropy detectors that identify API keys, secrets, and credentials with automatic masking.
* `analyzers/ide_comparison/`: Dedicated comparison engine evaluating and scoring AI-assisted IDE coding environments (Google Antigravity, Cursor, GitHub Copilot, Windsurf).
* `analyzers/model_comparison/`: Offline multi-criteria decision-making engine (WSM and TOPSIS) that scores and ranks 21 foundation models against prompt requirements without external API calls; includes model registry, tokenizer adapters, and task classifier.
* `analyzers/prompt_understanding/`: Semantic analyzers extracting development scope, architectural requirements, and task intent.
* `analyzers/vulnerability_detection/`: Multi-label CWE surface detection, potential risk inference, and confirmed vulnerability analysis.
* `analyzers/security_requirements/`: Security rule evaluation and control detection modules.

---

### 2. `data/`
Houses all project datasets, partitioned by lifecycle phase to guarantee zero duplication and prevent data leakage:
* `data/clean/`: Verified, sanitized, and labeled datasets used for training and benchmarking:
  - `implementation_scope_cleaned.csv`: 848 deduplicated, label-audited developer prompt requirements.
  - `high_quality_security_dataset.csv`: Cleaned multi-label application security dataset across 15 CWE categories.
  - `mpdd_clean.csv`: Cleaned prompt injection and adversarial prompt dataset.
  - `web_security_payloads.csv`: Targeted web vulnerability test payloads.
* `data/raw/`: Original, immutable source datasets preserving original provenance:
  - `dev_prompt_requirements.csv`: Original 1,000-example export of developer prompt requirements.
  - `security_risk_dataset.csv`: Original security risk dataset.
  - `web_security_payloads_clean.json`: Raw web vulnerability payloads.
  - `external/owasp_llm_top10_samples.json`: OWASP Top 10 for LLM reference samples.
* `data/splits/`: Leakage-safe, stratified train/validation/test partitions:
  - `train.csv`, `val.csv`, `test.csv`: Stratified splits for Implementation Scope Predictor experiments.
  - `prompt_families.csv`: Cluster-grouped prompt families ensuring no prompt variants span across train and test splits.
  - `dataset_version_manifest.json`: Version manifest recording hashes, row counts, and split parameters.
* `data/review/`: Audit findings, flagged outliers, semantic anomalies, and proposed label corrections:
  - `outliers.csv`, `semantic_review.csv`, `invalid_completions.csv`, `label_inconsistencies.csv`, `mpdd_label_corrections.csv`, `path_traversal_examples.csv`.
* `data/reports/`: Data audit logs and quality assessments:
  - `dataset_quality_report.md`, `prompt_families_report.md`, `experiment_readiness_report.md`.
* `data/manifests/`: Dataset provenance schemas, hash manifests, and labeling policies:
  - `dataset_manifest.json`: Canonical dataset inventory with checksums and provenance.
  - `security_labeling_policy.md`: Official labeling criteria and evidence standards.

---

### 3. `models/`
Stores canonical trained model artifacts and configuration files loaded during runtime inference:
* `models/task_classification/`: Scikit-learn Pipeline classifying developer prompts into 6 primary task domains (`model.pkl`).
* `models/prompt_security/`: Calibrated TF-IDF + Logistic Regression model detecting prompt injections, jailbreaks, and adversarial attacks (`model.pkl`).
* `models/application_security/`: MultiOutputClassifier Pipeline detecting 15 CWE architectural attack surfaces (`model.pkl`).
* `models/complexity/`: Prompt complexity classifier ranking tasks into Simple, Moderate, Complex, or Extreme tiers (`model.pkl`).
* `models/implementation_scope/v1/`: 11 trained scikit-learn models (primary scope, 6 requirement flags, complexity, design emphasis, task intent) with vectorizers and threshold configs.
* `models/ranker/`: Calibrated multi-criteria decision matrices and weights for WSM / TOPSIS ranking (`ranker_weights.json`).
* `models/schemas.py`: Unified Pydantic and dataclass schemas defining model inputs, outputs, and confidence intervals.

---

### 4. `artifacts/`
Stores training run checkpoints, intermediate models, and evaluation outputs produced by training scripts:
* `artifacts/complexity_model/`: Checkpoint model artifact, metadata, and validation metrics from complexity model training.
* `artifacts/injection_detector/`: Checkpoint model artifact, metadata, and test metrics from injection detector training.
* `artifacts/task_classifier/`: Checkpoint model artifact, metadata, and cross-validation metrics from task classifier training.
* `artifacts/ranker/`: Calibrated weight distributions and preset matrices.

---

### 5. `ml/`
Contains machine learning training workflows and standalone inference utilities:
* `ml/train_security_model.py`: MultiOutputClassifier training pipeline for 15 application risk categories.
* `ml/train_prompt_security_model.py`: Injection detector training script with CalibratedClassifierCV.
* `ml/predict_security_risk.py`: Standalone CLI inference utility for application security risks.
* `ml/results/`: Evaluation metrics and unified evaluation summary JSON files (`prompt_security_metrics.json`, `application_risk_metrics.json`, `unified_evaluation_summary.json`).

---

### 6. `training/`
Contains system training scripts and the unified evaluation harness:
* `training/evaluate_models.py`: Unified evaluation harness benchmarking task classification, injection detection, complexity, and the 30-case Golden Benchmark Suite.
* `training/train_task_classifier.py`: End-to-end training script for the 6-domain task classifier.
* `training/train_injection_detector.py`: Injection detector training and artifact synchronization script.
* `training/train_complexity_model.py`: Complexity scoring model training script.
* `training/train_ranker.py`: Calibration script for multi-criteria ranking weights.

---

### 7. `experiments/`
Dedicated experiment tracking and reproducible baseline records:
* `experiments/implementation_scope/v1/`: Full reproducible experiment artifacts for baseline v1:
  - `train_baseline.py`: Reproducible training script for v1 baseline.
  - `baseline_experiment_report.md`: Complete experiment methodology, split statistics, and performance results.
  - `experiment_config.json`: Hyperparameters, vectorizer configurations, and probability decision thresholds.
  - `test_evaluation_metrics.json`: Unbiased test set evaluation metrics across all 10 prediction targets.
  - `validation_tuning_results.json`: Validation tuning metrics across decision thresholds and hyperparameter sweeps.
  - `error_analysis/`: Detailed error analysis reports, confusion matrices, and categorized failure breakdowns.

---

### 8. `optimizer/`
Contains deterministic prompt optimization rules and structured prompt templates:
* `optimizer/rules/`: Extensible security hardening rules (input validation, authentication, authorization, error handling, rate limiting).
* `optimizer/templates/`: Structured prompt templates for UI, Backend, Full Stack, and Systems tasks.

---

### 9. `scripts/`
Contains offline utilities and data engineering scripts:
* `scripts/data_preparation/`: Standalone scripts for cleaning, deduplicating, inspecting, and repairing datasets:
  - `add_targeted_security_examples.py`, `analyze_mpdd.py`, `audit_datasets.py`, `audit_security_dataset.py`, `clean_dataset.py`, `correct_obvious_labels.py`, `find_label_issues.py`, `generate_dataset_repairs.py`, `inspect_mpdd.py`, `inspect_mpdd_csv.py`, `prepare_mpdd.py`, `process_security_data.py`.

---

### 10. `tests/`
Automated test suite (193 tests) ensuring zero regressions across all system components:
* `tests/test_workflow_cli.py`: End-to-end tests for the unified workflow CLI (scope check, comparison selection, report formatting, regression scenarios).
* `tests/test_cli.py`: Test suite for the interactive workspace CLI menu, options 0-8, and non-interactive file analysis.
* `tests/test_end_to_end.py`: End-to-end integration tests verifying model loading, dataset schema integrity, and inference pipelines.
* `tests/test_evaluation_benchmark.py`: 30-case Golden Benchmark regression test suite verifying prompt injection detection, declared controls, and data splitting leakage prevention.
* `tests/test_model_comparison.py`: Unit and property tests for ranking algorithms (WSM, TOPSIS) and product registry metadata.
* `tests/test_analyzers_and_optimizer.py`: Unit tests for security analyzers, prompt optimizer rules, and token estimators.
* `tests/test_ui_prompt_understanding.py`: Semantic tests for UI and frontend prompt classification.

---

### 11. `docs/` & `documentation/`
Houses system documentation, technical design specifications, and research references:
* Methodology guides for model training, dataset labeling policies, and comparison benchmark whitepapers.

---

### 12. `reports/`
Stores generated evaluation reports, benchmark audits, and experiment findings.

---

### 13. `datasets/`
Provides backward-compatible symbolic links mapping legacy paths (`datasets/implementation_scope/*`) directly to `data/*` with zero byte duplication.


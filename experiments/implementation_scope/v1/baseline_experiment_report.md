# Implementation Scope Predictor — Baseline ML Experiment Report (v1)

**Experiment Version:** `v1.0-baseline`  
**Evaluation Date:** 2026-11-14T17:30:00+05:30  
**Dataset Artifact:** `datasets/implementation_scope/experiments/v1/`  
**Model Artifacts:** `models/implementation_scope/v1/`  
**Experiment Configuration:** `experiments/implementation_scope/v1/experiment_config.json`  
**Status:** Completed & Evaluated (One-Time Unseen Test Split Evaluation).

---

## 1. Executive Summary

This report documents the baseline Machine Learning experiment for the **PromptShield Implementation Scope Predictor**. Using the frozen, leakage-safe `v1` dataset splits, we established a deterministic, reproducible baseline using word and character subword TF-IDF representations paired with $L_2$-regularized Logistic Regression classifiers.

All hyperparameter selection ($C$, class weighting) and decision threshold calibrations were performed strictly on the **Validation set (129 rows)**. The **Test set (131 rows)** was evaluated exactly once after all modeling decisions were locked.

### Key Headline Results

| Target Dimension | Model Type | Validation Macro-F1 / F1 | Test Accuracy | Test Weighted-F1 | Test Macro-F1 / F1 | Status / Learnability |
|---|---|---|---|---|---|---|
| **`primary_scope`** | Multiclass (6 classes) | 0.3327 (Macro) | **85.50%** | **81.68%** | 0.3283 (Macro) | **Partially Learnable** (`backend` & `frontend` robust; minority classes collapse) |
| **`frontend_required`** | Binary ($\theta=0.35$) | 0.8155 | **95.42%** | — | **0.9167** | **Highly Learnable** |
| **`backend_required`** | Binary ($\theta=0.55$) | 0.9418 | **90.08%** | — | **0.9360** | **Highly Learnable** |
| **`database_required`** | Binary ($\theta=0.40$) | 0.8491 | **85.50%** | — | **0.8119** | **Strongly Learnable** |
| **`api_required`** | Binary ($\theta=0.30$) | 0.9000 | **91.60%** | — | **0.8451** | **Strongly Learnable** |
| **`authentication_required`**| Binary ($\theta=0.35$) | 0.6154 | **80.92%** | — | **0.3902** | **Unreliable / Data-Constrained** |
| **`external_integration_required`**| Binary ($\theta=0.35$)| 0.6667 | **73.28%** | — | **0.5070** | **Moderate / High False Positives** |
| **`complexity`** | Multiclass (3 tiers) | 0.6442 (Macro) | **64.12%** | **62.52%** | 0.5338 (Macro) | **Poor Linear Separation** |
| **`design_emphasis`** | Multiclass (3 tiers) | 0.5169 (Macro) | **88.55%** | **87.27%** | 0.6987 (Macro) | **Strongly Learnable for Low vs High** |
| **`task_intent`** | Multiclass (6 intents) | 0.7019 (Macro) | **80.15%** | **80.28%** | 0.6436 (Macro) | **Learnable for Frequent Actions** (`create`, `debug`, `optimize`) |

> [!WARNING]
> **Production Readiness Assessment**: This baseline ML model is **NOT ready for autonomous production deployment**. While core domain binary flags (`frontend_required`, `backend_required`, `api_required`, `database_required`) demonstrate solid generalization across unseen prompt families, minority primary scope categories (`design_only`, `security_only`, `unclear`, `full_stack`) and subtle requirement flags (`authentication_required`) suffer from acute data scarcity and high false-positive rates.

---

## 2. Dataset Partitioning & Leakage Audit

The experiment strictly utilized the frozen partitions created during dataset stabilization:

- **Source Provenance**: Adaption Labs synthetic developer requests export (`dev_prompt_requirements.csv`).
- **Cleaned Dataset**: 869 verified records (`fc0e0bb4f2125251256b897e39b2f6c62f7be88b0594fec2ad042334fad803ea`).
- **Partitioning Strategy**: `StratifiedGroupKFold` on normalized prompt families (`group_id`).
- **Partition Independence**:
  - $\text{Train Groups (569)} \cap \text{Val Groups (122)} = \emptyset$
  - $\text{Train Groups (569)} \cap \text{Test Groups (122)} = \emptyset$
  - $\text{Val Groups (122)} \cap \text{Test Groups (122)} = \emptyset$

| Partition | Row Count | Pct | Multi-Member Family Rows | SHA-256 Digest |
|---|---|---|---|---|
| **Train** | 609 | 70.1% | 51 rows | `a4f28ce751bca17970969cff90a8da5c6089fd231cfb5638987c71080f2663a5` |
| **Validation** | 129 | 14.8% | 11 rows | `70f24d96c91e7fcce4f63a31bcddbd26d31792ac29cad3de9fb65320793f496d` |
| **Test** | 131 | 15.1% | 10 rows | `3790b11aafc1542ceda333c07133f7747e1b7e3300286bff6c5830d7d817dfb5` |

---

## 3. Feature Extraction Architecture

Feature extraction was performed using a `scikit-learn` `FeatureUnion` fitted **exclusively on the training partition**:

1. **Word-Level TF-IDF**:
   - N-gram range: `(1, 2)` (unigrams and bigrams).
   - Frequency bounds: `min_df=2`, `max_features=5000`.
   - Weighting: Sublinear term-frequency scaling (`sublinear_tf=True`).
   - Purpose: Captures exact domain vocabulary (`fastapi`, `react`, `postgresql`, `jwt`, `docker`, `css grid`).
2. **Character-Level TF-IDF**:
   - N-gram range: `(3, 5)` (character chunks within word boundaries).
   - Analyzer: `char`.
   - Frequency bounds: `min_df=2`, `max_features=10000`.
   - Weighting: Sublinear term-frequency scaling (`sublinear_tf=True`).
   - Purpose: Captures morphological variations, compound tokens (`dangerouslySetInnerHTML`, `req.body`), framework prefixes/suffixes (`auth0`, `asyncpg`, `sql-`), and misspelled technical jargon.

**Vocabulary Dimension**: Exactly **14,628 dense features** extracted across the 609 training examples.

---

## 4. Hyperparameter Tuning & Model Selection

Tuning was conducted over a comprehensive grid evaluated exclusively on the validation set (`val.csv`). The `liblinear` solver was utilized to eliminate numerical overflow and matrix division anomalies observed in `lbfgs` under high-dimensional sparse representations.

### 4.1 Primary Scope Classifier Tuning

Evaluating unweighted versus balanced class weighting revealed critical insights into minority class representation:
- `class_weight=None`: Led to complete majority-class collapse. `full_stack` (16 val samples) received **0.00 recall**; `backend` absorbed 99% of mass. Macro-F1 stalled at $0.2639$.
- `class_weight='balanced'`: Enabled `full_stack` prediction (recall 0.25, precision 0.57).
- Optimal configuration: `solver='liblinear'`, `C=2.0`, `class_weight='balanced'`. Validation Macro-F1 reached **0.3327** with overall accuracy of **77.52%**.

### 4.2 Requirement Flags Threshold Calibration

Because binary requirement labels exhibit severe class imbalance (e.g., `authentication_required` positive rate is 7.2% in train, 9.3% in val), the standard $0.50$ decision threshold severely suppresses recall. We evaluated threshold sweeps $\theta \in [0.20, 0.70]$:

| Target Flag | Optimal $C$ | Class Weight | Calibrated Threshold $\theta$ | Val F1 | Val Precision | Val Recall | Val Brier |
|---|---|---|---|---|---|---|---|
| `frontend_required` | 5.0 | `None` | **0.35** | 0.8155 | 0.8936 | 0.7500 | 0.1313 |
| `backend_required` | 2.0 | `None` | **0.55** | 0.9418 | 0.8990 | 0.9889 | 0.0970 |
| `database_required` | 1.0 | `'balanced'` | **0.40** | 0.8491 | 0.8036 | 0.9000 | 0.1230 |
| `api_required` | 1.0 | `None` | **0.30** | 0.9000 | 0.8710 | 0.9310 | 0.0846 |
| `authentication_required` | 0.5 | `'balanced'` | **0.35** | 0.6154 | 0.4444 | 1.0000 | 0.1001 |
| `external_integration_required` | 0.5 | `'balanced'` | **0.35** | 0.6667 | 0.5532 | 0.8387 | 0.1496 |

---

## 5. Test Evaluation Results

The frozen models were evaluated on the 131 test examples.

### 5.1 Primary Scope Classification

- **Test Accuracy**: **85.50%** (112 / 131 correct)
- **Macro-F1**: **0.3283**
- **Weighted-F1**: **0.8168**

#### Per-Class Performance Table

| Class Label | Support (Test) | Precision | Recall | F1-Score | Analysis |
|---|---|---|---|---|---|
| `backend` | 89 | 0.8713 | **0.9888** | **0.9263** | Dominant majority class; near-perfect recall. |
| `frontend` | 28 | 0.8214 | **0.8214** | **0.8214** | Well-separated lexical boundary (`react`, `css`, `html`). |
| `full_stack` | 7 | 0.5000 | 0.1429 | 0.2222 | Heavy confusion with `backend` (3) and `frontend` (3). |
| `unclear` | 4 | 0.0000 | 0.0000 | 0.0000 | 100% misclassified into `backend`. |
| `security_only` | 2 | 0.0000 | 0.0000 | 0.0000 | 100% misclassified into `backend`. |
| `design_only` | 1 | 0.0000 | 0.0000 | 0.0000 | Misclassified into `frontend`. |

#### Confusion Matrix (True Rows vs Predicted Columns)

```
Classes: ['backend', 'design_only', 'frontend', 'full_stack', 'security_only', 'unclear']

                Pred: backend  design_only  frontend  full_stack  security_only  unclear
True: backend          88           0          1           0            0           0
True: design_only       0           0          1           0            0           0
True: frontend          4           0         23           1            0           0
True: full_stack        3           0          3           1            0           0
True: security_only     2           0          0           0            0           0
True: unclear           4           0          0           0            0           0
```

---

### 5.2 Binary Architectural Requirement Classifiers

| Requirement Dimension | Threshold $\theta$ | Test Precision | Test Recall | Test F1 | Test Accuracy | Brier Score | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| **`frontend_required`** | 0.35 | **0.9167** | **0.9167** | **0.9167** | **95.42%** | **0.0577** | 92 | 3 | 3 | 33 |
| **`backend_required`** | 0.55 | **0.9048** | **0.9694** | **0.9360** | **90.08%** | **0.0809** | 23 | 10 | 3 | 95 |
| **`database_required`** | 0.40 | **0.7593** | **0.8723** | **0.8119** | **85.50%** | **0.1185** | 71 | 13 | 6 | 41 |
| **`api_required`** | 0.30 | **0.8333** | **0.8571** | **0.8451** | **91.60%** | **0.0812** | 90 | 6 | 5 | 30 |
| **`authentication_required`** | 0.35 | **0.2857** | **0.6154** | **0.3902** | **80.92%** | **0.1189** | 98 | 20 | 5 | 8 |
| **`external_integration_required`**| 0.35 | **0.3600** | **0.8571** | **0.5070** | **73.28%** | **0.1315** | 78 | 32 | 3 | 18 |

---

### 5.3 Secondary Targets Evaluation

#### 1. `complexity` (Tiers: `high`, `medium`, `low`)
- **Overall Accuracy**: **64.12%** | **Macro-F1**: **0.5338** | **Weighted-F1**: **0.6252**
- `medium`: Support 79, Precision 0.6818, Recall 0.7595, **F1 0.7186**
- `high`: Support 33, Precision 0.5882, Recall 0.6061, **F1 0.5970**
- `low`: Support 19, Precision 0.4444, Recall 0.2105, **F1 0.2857**
- **Error Dynamic**: 15 out of 19 `low` complexity prompts were predicted as `medium`. TF-IDF linear weights cannot infer task difficulty or algorithmic depth from vocabulary alone.

#### 2. `design_emphasis` (Tiers: `low`, `medium`, `high`)
- **Overall Accuracy**: **88.55%** | **Macro-F1**: **0.6987** | **Weighted-F1**: **0.8727**
- `low`: Support 103, Precision 0.9189, Recall 0.9903, **F1 0.9533**
- `medium`: Support 15, Precision 0.6923, Recall 0.6000, **F1 0.6429**
- `high`: Support 13, Precision 0.7143, Recall 0.3846, **F1 0.5000**
- **Error Dynamic**: Prompts with styling cues (`masonry layout`, `custom scrollbar`) are cleanly separated from `low`, but distinguishing `medium` from `high` visual styling has significant ambiguity.

#### 3. `task_intent` (Intents: `create`, `debug`, `explain`, `modify`, `optimize`, `audit`)
- **Overall Accuracy**: **80.15%** | **Macro-F1**: **0.6436** | **Weighted-F1**: **0.8028**
- `create`: Support 67, Precision 0.8841, Recall 0.9104, **F1 0.8971**
- `debug`: Support 27, Precision 0.8571, Recall 0.8889, **F1 0.8727**
- `optimize`: Support 17, Precision 0.8571, Recall 0.7059, **F1 0.7742**
- `explain`: Support 17, Precision 0.5000, Recall 0.4118, **F1 0.4516**
- `modify`: Support 3, Precision 0.1667, Recall 0.3333, **F1 0.2222**
- `audit`: Support 0 in Test (0 in Val, 3 in Train). Unrepresented; cannot be evaluated empirically.

---

## 6. Detailed Error Analysis & Concrete Case Studies

### 6.1 Primary Scope Confusions
1. **`full_stack` Misclassified as `frontend` (Row 55)**:
   - *Prompt*: *"dynamics 365 plugin is throwing null ref on contact create, check the context. also the custom form fields look default and boring, can u inject some css to make the labels bold and the inputs have a focus ring? make it user friendly. and ensure data privacy compliance."*
   - *Issue*: Strong stylistic keywords (`css`, `labels bold`, `focus ring`, `user friendly`) overwhelmed the Dynamics 365 backend plugin context.
2. **`unclear` Misclassified as `backend` (Row 70 & 88)**:
   - *Prompt (Row 88)*: *"Clarifying the difference between null and undefined in JS but the examples are confusing. Need a precise explanation that covers engine behavior in V8 and SpiderMonkey. Also mention TC39 proposals if relevant. The audience is senior devs so skip the basics..."*
   - *Issue*: Technical depth tokens (`V8`, `SpiderMonkey`, `engine behavior`) triggered the backend prior because `unclear` has zero identifiable lexical markers.

### 6.2 Authentication False Positives & Negatives
1. **False Positive (Row 109)**:
   - *Prompt*: *"hey, need to whip up a quick login view for the new dashboard, just the UI stuff y'know? make it look slick with some dark mode vibes. oh and pls handle the form validation on the client side so we dont hit the server with bad data. thx!"*
   - *Predicted Prob*: **0.4649** (Exceeds calibrated threshold $\theta=0.35$).
   - *Root Cause*: The token `login` in a pure UI layout prompt triggers the authentication detector even when no backend session, hashing, or credential checking is required.
2. **False Negative (Row 44 & 655)**:
   - *Prompt (Row 44)*: *"Hey, I need to build a Python client for the Snowflake SQL API v2 to run some async queries in our data pipeline. We need to handle the requestId for idempotency, parse the bound parameters correctly, and deal with those 202/408 status codes when polling for results... Make sure to include the right headers and show how to retry if the connection drops..."*
   - *Predicted Prob*: **0.2851** (Below $\theta=0.35$).
   - *Root Cause*: Enterprise API prompts referring to *"right headers"* or bearer tokens without explicitly uttering *"auth"* or *"login"* fall below the detection threshold.

---

## 7. Critical Findings & Core Questions Addressed

### 1. Which targets are learnable?
- **Highly Learnable**:
  - `frontend_required` (Test F1: **0.9167**, Accuracy: **95.42%**)
  - `backend_required` (Test F1: **0.9360**, Accuracy: **90.08%**)
  - `api_required` (Test F1: **0.8451**, Accuracy: **91.60%**)
  - `database_required` (Test F1: **0.8119**, Accuracy: **85.50%**)
  - Major task intents: `create` (F1: **0.8971**), `debug` (F1: **0.8727**), `optimize` (F1: **0.7742**).
  - Primary scope major classes: `backend` (F1: **0.9263**), `frontend` (F1: **0.8214**).
- **Why they learn well**: These concepts possess distinct, unambiguous technical vocabularies (e.g., `sql`, `postgres`, `endpoint`, `rest`, `router`, `component`, `react`, `ui`, `css`, `debug`, `error`).

### 2. Which predictions are unreliable?
- **Severely Unreliable**:
  - **`primary_scope` minority classes**: `full_stack` (F1: **0.2222**), `design_only` (F1: **0.0000**), `security_only` (F1: **0.0000**), `unclear` (F1: **0.0000**).
  - **`authentication_required`**: Precision is only **28.57%**. 20 false positives out of 28 positive predictions on test set. The model triggers on superficial mentions of `login` or `user stats`.
  - **`external_integration_required`**: Precision is **36.00%** with 32 false positives. Any tool reference (cron, docker, express) leaks into integration predictions.
  - **`complexity`**: Low complexity classification fails (F1: **0.2857**, Recall: **21.05%**).
  - **`task_intent: audit`**: Total failure to evaluate due to zero validation and test examples.

### 3. Does the model generalize to unseen prompt families?
- **Yes, for core technical domains**: Because our splits strictly enforced zero prompt family leakage ($\text{Train} \cap \text{Test} = \emptyset$), the high test performance on `frontend_required` (95.42% acc) and `backend_required` (90.08% acc) proves that the n-gram representations generalize across distinct syntactic phrasing, informal greetings, and varying prompt structures.
- **No, for subtle semantic requirements**: Generalization breaks down when developer requests express multi-domain requirements without obvious keywords (e.g., configuring Snowflake API request headers without using the token "auth").

### 4. Is Logistic Regression a reasonable baseline?
- **Yes, as a benchmark and sanity check**:
  - Training takes < 5 seconds.
  - Linear models are fully interpretable and provide exact coefficient attribution.
  - Calibration (Brier scores between $0.0577$ and $0.1315$) is surprisingly respectable.
- **No, as a final production architecture**:
  - Linear models cannot capture contextual negation or modifier scope (e.g., *"build a login view, UI only, no backend auth"* triggers `authentication_required` because `login` has a high positive weight).
  - Linear models lack cross-attention between requirement tokens and scope constraints.
  - Dense semantic embeddings or instruction-tuned transformer encoders (e.g., ModernBERT or DeBERTa-v3) are required to disambiguate subtle intent.

### 5. What data should be collected next?
1. **Targeted Rare-Class Syntheses (+250 samples)**:
   - +50 `security_only` prompts (auditing JWT signing algorithms, verifying TLS configurations, scanning dependencies).
   - +50 `design_only` prompts (Figma token translations, pure CSS color paletting, typography scaling).
   - +50 `task_intent: audit` prompts (code security reviews, architectural compliance checks).
   - +50 pure `full_stack` prompts with explicit end-to-end multi-tier requirements.
   - +50 `unclear` prompts containing conversational ambiguities or under-specified developer questions.
2. **Negative Authentication Hard Negatives (+100 samples)**:
   - Prompts explicitly stating: *"login form UI only, mock data, no backend authentication"* to force the model to decouple the token `login` from the flag `authentication_required`.
3. **Hard External Integration Boundaries (+100 samples)**:
   - Prompts clarifying internal standard library usage (e.g., Node.js `fs`, Python `itertools`) versus genuine 3rd-party SaaS integrations (Stripe, Twilio, Salesforce).

---

## 8. Reproducibility Manifest & Artifact Directory

All serialized models and configuration files have been preserved:

```
models/implementation_scope/v1/
├── feature_pipeline.joblib          # Scikit-learn FeatureUnion (Word + Char TF-IDF)
├── primary_scope_model.joblib       # Logistic Regression (Balanced, C=2.0)
├── requirement_models.joblib        # 6 Calibrated Binary Logistic Regression Models
├── secondary_models.joblib          # Complexity, Design Emphasis, Task Intent Models
└── model_metadata.json              # Checksums, parameters, and environment specs

experiments/implementation_scope/v1/
├── train_baseline.py                # Reproducible execution script
├── experiment_config.json           # Grid parameters and vectorizer settings
├── validation_tuning_results.json   # Full validation hyperparameter search logs
├── test_evaluation_metrics.json     # Complete test metrics, CMs, and error cases
└── baseline_experiment_report.md    # This report
```

### Checksums

| File Artifact | SHA-256 Checksum |
|---|---|
| [`models/implementation_scope/v1/feature_pipeline.joblib`](file:///Users/francisanto/backend/backendprompt/models/implementation_scope/v1/feature_pipeline.joblib) | `9b3f3b9273c52e825a07248e351982b6be7520e55047b744439c3620f4f9db56` |
| [`models/implementation_scope/v1/primary_scope_model.joblib`](file:///Users/francisanto/backend/backendprompt/models/implementation_scope/v1/primary_scope_model.joblib) | `faec2b8108c454e6ff4cc6e9bca93b825eef0ffdbf6141c49dc745348bbdd214` |
| [`models/implementation_scope/v1/requirement_models.joblib`](file:///Users/francisanto/backend/backendprompt/models/implementation_scope/v1/requirement_models.joblib) | `f0fb5ee94665ce9a8aa292158933fe55dd092eafe21f37b98bf9b14b8db4d06a` |
| [`models/implementation_scope/v1/secondary_models.joblib`](file:///Users/francisanto/backend/backendprompt/models/implementation_scope/v1/secondary_models.joblib) | `451c8db1f39f37cff9d71c1103cba2fc1a2a472ca620b7bc2ae03ce96f1d2df8` |
| [`experiments/implementation_scope/v1/train_baseline.py`](file:///Users/francisanto/backend/backendprompt/experiments/implementation_scope/v1/train_baseline.py) | Verified runnable via `.venv/bin/python` |
| [`experiments/implementation_scope/v1/test_evaluation_metrics.json`](file:///Users/francisanto/backend/backendprompt/experiments/implementation_scope/v1/test_evaluation_metrics.json) | Complete test logs |


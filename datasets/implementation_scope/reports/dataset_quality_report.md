# Implementation Scope Predictor — Final Dataset Accounting & Training Readiness Audit Report

**Audit Date:** 2026-11-14  
**Dataset Artifact:** `PromptShield AI / Implementation Scope Predictor`  
**Source Provenance:** Adaption Labs Synthetic Developer Prompts Export (`dev_prompt_requirements.csv`)  
**Audit Scope:** Mathematical Row Accounting, Dataset Cleaning Verification, Target Suitability, and Data Leakage Analysis.  
**Status:** Audit Completed; Exact Accounting Reconciled; Model Training Deferred.

---

## 1. Mathematical Row Accounting & Set Reconciliations

### 1.1 Resolution of the Accounting Reconciliation Question

> [!IMPORTANT]
> **Reconciliation Clarification**:
> - Total source records: **1,000**
> - Total structurally excluded records across all source rows: **131** (21 invalid/schema + 110 flagged schema = 131)
> - Clean usable records: **869** (1,000 - 131 = 869)
> - Within the 979 schema-compliant records, exactly **110 rows** are flagged (979 - 110 = 869).
>
> The previously reported 131 unique flagged rows was the union across the **entire 1,000 source examples** (including the 21 non-JSON / schema-broken rows), not a subset of the 979 schema records.

### 1.2 Exact Set Definitions & Sizes

| Set Identifier | Category Description | Row Count | Specific Row IDs / Details |
|---|---|---|---|
| **$U$** | **Original Source Records** | **1,000** | Rows `0` through `999`. |
| **$A$** | **Invalid Completions (Non-JSON)** | **20** | Rows `13`, `30`, `94`, `119`, `221`, `268`, `328`, `497`, `539`, `584`, `592`, `645`, `679`, `734`, `740`, `807`, `876`, `919`, `938`, `946`. |
| **$B$** | **Schema Mismatch (Wrong Schema)** | **1** | Row `82` (contains valid JSON for a `contentTypes` NoSQL data model, missing all 16 target schema fields). |
| **$S$** | **Valid Schema-Compliant Records** ($U \setminus (A \cup B)$) | **979** | Rows possessing valid JSON and all 16 target scope predictor keys. |
| **$C$** | **Structural Label Inconsistencies** | **69** | Rows with explicit evidence paired with 0 flags (51 rows), full-stack missing flags (22 rows), or invalid primary scope (1 row). |
| **$D$** | **Semantic Review Flags** | **57** | Rows with programming/DB tasks mislabelled as design_only (14 rows), backend-only tasks mislabelled as full_stack (20 rows), etc. |
| **$E$** | **Prompt Outliers & Pipeline Leaks** | **7** | Rows `13`, `119`, `221`, `406`, `804`, `850`, `946` (length >1500 chars, system prompt leaks, internal eval harness refs). |

### 1.3 Intersections Between Categories

```
Non-JSON Completions (A: 20) ────────┐
                                     ├── [13, 119, 221, 946] (4 rows in A ∩ E)
Outliers & Pipeline Leaks (E: 7) ────┘
                                     └── [406, 804, 850] (3 rows in E ∩ S)

Structural Inconsistencies (C: 69) ──┐
                                     ├── 19 rows in C ∩ D (both structural & semantic contradiction)
Semantic Review Issues (D: 57) ──────┘
```

- **$A \cap B = \emptyset$** (0 rows): Non-JSON and schema mismatch sets are mutually disjoint.
- **$A \cap E = \{13, 119, 221, 946\}$** (4 rows): Non-JSON completions that also contained system prompt leaks or eval runner configs.
- **$B \cap E = \emptyset$** (0 rows).
- **$C \cap D = 19 \text{ rows}$**: Rows `61`, `84`, `91`, `115`, `132`, `167`, `191`, `204`, `334`, `339`, `436`, `537`, `555`, `594`, `595`, `604`, `638`, `757`, `983`. (These are prompts specifying pure backend REST APIs/scripts that were labelled `full_stack` with `frontend_required=0`, violating both structural integrity and semantic correctness).
- **$C \cap E = \emptyset$** (0 rows).
- **$D \cap E = \emptyset$** (0 rows).
- **$C \cap D \cap E = \emptyset$** (0 rows).

### 1.4 Formal Accounting Proof

1. **Flagged within the 979 Schema-Compliant Records ($F_S$)**:
   $$F_S = C \cup D \cup (E \cap S)$$
   $$|F_S| = |C| + |D| + |E \cap S| - |C \cap D|$$
   $$|F_S| = 69 + 57 + 3 - 19 = \mathbf{110} \text{ rows}$$

2. **Clean Usable Records ($R_{clean}$)**:
   $$|R_{clean}| = |S| - |F_S| = 979 - 110 = \mathbf{869} \text{ rows}$$

3. **Total Excluded Records across the Entire Dataset ($F_{total}$)**:
   $$|F_{total}| = |A| + |B| + |F_S| = 20 + 1 + 110 = \mathbf{131} \text{ rows}$$

4. **Global Conservation Check**:
   $$|U| = |R_{clean}| + |F_{total}| = 869 + 131 = \mathbf{1,000} \text{ rows}$$
   $$R_{clean} \cap F_{total} = \emptyset \quad \text{(Zero double-counting)}$$

---

## 2. Re-Verification of the Cleaned Dataset (N=869)

Every record in `implementation_scope_cleaned.csv` was programmatically audited:
- **`primary_scope` Validity**: 100% of rows contain valid scope labels (`backend`, `frontend`, `full_stack`, `unclear`, `design_only`, `security_only`). Zero invalid values.
- **Binary Requirement Flags**: All 6 flags (`frontend_required`, `backend_required`, `database_required`, `api_required`, `authentication_required`, `external_integration_required`) are strictly integers `0` or `1`. Zero nulls, floats, or booleans.
- **Evidence Consistency**: Zero contradictions exist between evidence fields and requirement flags. Every positive flag is backed by `explicit` or `inferred` evidence; every negative flag has `unknown` evidence.
- **`full_stack` Verification (51 rows)**: Every single full-stack example has both `frontend_required == 1` and `backend_required == 1`. Inspection confirms each prompt explicitly requires both client-side UI (React, Vue, Qt graphics) and server-side components (REST APIs, databases, microservices).
- **`design_only` Verification (9 rows)**: Every single remaining `design_only` example genuinely focuses exclusively on visual styling, CSS layouts, and UI components without backend dependencies (e.g. CSS Grid dashboard layouts, custom pill radio button animations, tooltip styling, flexbox alignment, responsive centering). Zero algorithmic or backend tasks remain.
- **`unclear` Verification (12 rows)**: Every single `unclear` prompt represents genuinely ambiguous conceptual, tooling, or educational queries (e.g. debugging JavaScript hoisting, PyCharm virtualenv configuration, C++ ctags navigation, core dump analysis).

---

## 3. Prediction Target Suitability Assessment

### Target 1: Primary Scope Classification (6 Classes)
- **Suitability**: **Partial / Severely Imbalanced**.
- **Distribution**:
  - `backend`: 551 (63.4%) — Highly viable.
  - `frontend`: 237 (27.3%) — Viable.
  - `full_stack`: 51 (5.9%) — Marginal.
  - `unclear`: 12 (1.4%) — Underrepresented.
  - `design_only`: 9 (1.0%) — **Too underrepresented for reliable training**.
  - `security_only`: 9 (1.0%) — **Too underrepresented for reliable training**.
- **Recommendation**: Minimum 50 examples per class required for reliable multi-class classification. Need +41 `design_only`, +41 `security_only`, and +38 `unclear` examples.

### Target 2: Core Requirement Flags (Binary)
- **Suitability**: **High**.
  - `backend_required`: 609 Positive (70.1%) vs 260 Negative (29.9%) — Ratio 1:0.4.
  - `frontend_required`: 306 Positive (35.2%) vs 563 Negative (64.8%) — Ratio 1:1.8.
  - `database_required`: 298 Positive (34.3%) vs 571 Negative (65.7%) — Ratio 1:1.9.
  - `api_required`: 209 Positive (24.1%) vs 660 Negative (75.9%) — Ratio 1:3.2.
- **Assessment**: Well-supported by dataset. Binary classifiers will achieve stable convergence with standard BCE loss.

### Target 3: Specialized Requirement Flags (Binary)
- **Suitability**: **Marginal to Poor**.
  - `external_integration_required`: 146 Positive (16.8%) vs 723 Negative (83.2%) — Ratio 1:5.0 (Moderate).
  - `authentication_required`: 69 Positive (7.9%) vs 800 Negative (92.1%) — Ratio 1:11.6 (**Severe imbalance**).
- **Assessment**: Positive instances for authentication (69) are sparse. Requires focal loss, class-weighted loss, or threshold tuning.

### Target 4: Design Emphasis (3 Classes)
- **Suitability**: **Skewed**.
  - `low`: 631 (72.6%)
  - `medium`: 141 (16.2%)
  - `high`: 97 (11.2%)
- **Assessment**: Usable with ordinal regression or class weighting.

### Target 5: Complexity (3 Classes)
- **Suitability**: **Good**.
  - `medium`: 504 (58.0%)
  - `high`: 203 (23.4%)
  - `low`: 162 (18.6%)
- **Assessment**: Well-distributed across all three tiers.

### Target 6: Task Intent (6 Classes)
- **Suitability**: **Heavily Skewed**.
  - `create`: 426 (49.0%) — Viable.
  - `debug`: 147 (16.9%) — Viable.
  - `explain`: 125 (14.4%) — Viable.
  - `optimize`: 119 (13.7%) — Viable.
  - `modify`: 49 (5.6%) — Marginal.
  - `audit`: 3 (0.3%) — **Completely inadequate for training (only 3 samples)**.
- **Assessment**: Model cannot learn the `audit` class reliably without substantial data collection (+47 examples).

---

## 4. Data Leakage & Synthetic Repetition Analysis

### 4.1 Synthetic Scenario Prompts
- In the original dataset, **14 prompts** followed the exact pattern:
  `"Hey, need to spin up a quick prototype for an AI project manager (scenario [0-9]{6})..."`
- 6 were excluded during quality screening due to label contradictions or invalid completions.
- **8 prompts remain in the clean dataset** (Rows 160, 326, 452, 530, 641, 733, 764, 870).
- **Leakage Risk**: If randomly split into train/test, identical prompt templates with different scenario numbers will appear in both splits, causing artificial test accuracy inflation.

### 4.2 Repetitive Template Families
Lexical clustering identified several synthetic template clusters:
- **REST API Boilerplate**: 19 prompts beginning with *"Hey, I need to spin up a quick REST API..."*
- **ServiceNow Sync Boilerplate**: 15 prompts beginning with *"I need a Python script to sync incidents from ServiceNow..."*
- **Prototype Spin-Up Boilerplate**: 10 prompts beginning with *"Hey, need to spin up a quick prototype for an AI project manager..."*

### 4.3 Grouping Recommendation
> [!IMPORTANT]
> **Cluster-Based GroupKFold Splitting**:
> When training models, **do not use random train/test splitting**. Implement semantic group hashing (e.g. normalizing scenario numbers, entity names, and template prefixes) and assign entire template families exclusively to either train, validation, or test.

---

## 5. Summary Table: Additional Samples Needed for Production Readiness

| Prediction Target | Deficient Class / Dimension | Clean Count | Recommended Minimum | Samples Needed |
|---|---|---|---|---|
| `primary_scope` | `design_only` | 9 | 50 | **+41** |
| `primary_scope` | `security_only` | 9 | 50 | **+41** |
| `primary_scope` | `unclear` | 12 | 50 | **+38** |
| `primary_scope` | `full_stack` | 51 | 100 | **+49** |
| `task_intent` | `audit` | 3 | 50 | **+47** |
| `task_intent` | `modify` | 49 | 100 | **+51** |
| `authentication_required`| Positive (1) | 69 | 150 | **+81** |

---

## 6. Audit Governance Conclusion

All 1,000 source examples are accounted for with zero double-counting. The clean dataset (`implementation_scope_cleaned.csv`) containing 869 examples is structurally intact, semantically verified, and free of contradictions.

In accordance with user requirements, **no models have been trained, no synthetic labels were fabricated, and existing PromptShield classifiers remain untouched.**

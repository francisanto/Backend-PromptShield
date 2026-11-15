# Implementation Scope Predictor — Experiment Readiness Report (v1)

**Experiment Version:** `v1`  
**Dataset Artifact:** `PromptShield AI / Implementation Scope Predictor`  
**Generated On:** 2026-11-14T15:25:00+05:30  
**Directory Path:** `datasets/implementation_scope/experiments/v1/`  
**Status:** Frozen & Split; Zero Data Leakage; Ready for Reproducible Baseline Training.

---

## 1. Executive Summary

The Implementation Scope Predictor dataset has been frozen, grouped into leakage-safe prompt families, and partitioned into reproducible **train (70.1%)**, **validation (14.8%)**, and **test (15.1%)** splits.

### Key Governance Verifications
- **Zero Label Tampering**: Cleaned labels and original source rows were preserved unmodified.
- **Zero Synthetic Fabrication**: No synthetic samples or augmentations were added.
- **Zero Model Retraining**: Existing PromptShield security classifiers remain frozen.
- **Zero Partition Leakage**: 100% of multi-member prompt families are assigned to exactly one partition.

---

## 2. Partition Summary & Reproducibility Hashes

| Partition | Row Count | Pct of Clean | Group Count | SHA-256 Checksum | File Path |
|---|---|---|---|---|---|
| **Train** | **609** | **70.1%** | 569 | `a4f28ce751bca17970969cff90a8da5c6089fd231cfb5638987c71080f2663a5` | [`train.csv`](file:///Users/francisanto/backend/backendprompt/datasets/implementation_scope/experiments/v1/train.csv) |
| **Validation** | **129** | **14.8%** | 122 | `70f24d96c91e7fcce4f63a31bcddbd26d31792ac29cad3de9fb65320793f496d` | [`val.csv`](file:///Users/francisanto/backend/backendprompt/datasets/implementation_scope/experiments/v1/val.csv) |
| **Test** | **131** | **15.1%** | 122 | `3790b11aafc1542ceda333c07133f7747e1b7e3300286bff6c5830d7d817dfb5` | [`test.csv`](file:///Users/francisanto/backend/backendprompt/datasets/implementation_scope/experiments/v1/test.csv) |
| **Total** | **869** | **100.0%** | **813** | — | — |

---

## 3. Data Leakage & Grouping Audit

### 3.1 Prompt Family Grouping
- **Total Prompt Groups:** `813`
- **Singleton Prompts (Size 1):** `797` groups (797 prompts)
- **Multi-Member Prompt Families (Size > 1):** `16` groups (72 prompts)

### 3.2 Partition Overlap Verification
- $\text{Train Groups} \cap \text{Validation Groups} = \emptyset$ (0 shared groups)
- $\text{Train Groups} \cap \text{Test Groups} = \emptyset$ (0 shared groups)
- $\text{Validation Groups} \cap \text{Test Groups} = \emptyset$ (0 shared groups)
- **Leakage Status**: **ZERO LEAKAGE DETECTED**. Every prompt in a synthetic scenario family, ServiceNow sync family, or Snowflake API family is strictly confined to a single partition.

---

## 4. Class Distributions Across Splits

### 4.1 Primary Scope Distribution

| Primary Scope | Total Clean | Train Count (%) | Val Count (%) | Test Count (%) | Representation Assessment |
|---|---|---|---|---|---|
| **`backend`** | 551 | 391 (71.0%) | 71 (12.9%) | 89 (16.2%) | Robust across all splits. |
| **`frontend`** | 237 | 172 (72.6%) | 37 (15.6%) | 28 (11.8%) | Robust across all splits. |
| **`full_stack`** | 51 | 28 (54.9%) | 16 (31.4%) | 7 (13.7%) | Adequate in Train/Val; small Test set. |
| **`unclear`** | 12 | 7 (58.3%) | 1 (8.3%) | 4 (33.3%) | Sparse in Val (1 sample). |
| **`design_only`** | 9 | 7 (77.8%) | 1 (11.1%) | 1 (11.1%) | **Critically sparse** (1 sample in Val, 1 in Test). |
| **`security_only`** | 9 | 4 (44.4%) | 3 (33.3%) | 2 (22.2%) | **Critically sparse** (3 in Val, 2 in Test). |

### 4.2 Binary Requirement Flags

| Requirement Flag | Total Pos (Pct) | Train Pos (%) | Val Pos (%) | Test Pos (%) |
|---|---|---|---|---|
| **`backend_required`** | 609 (70.1%) | 421 (69.1%) | 90 (14.8%) | 98 (16.1%) |
| **`frontend_required`** | 306 (35.2%) | 214 (69.9%) | 56 (18.3%) | 36 (11.8%) |
| **`database_required`** | 298 (34.3%) | 201 (67.4%) | 50 (16.8%) | 47 (15.8%) |
| **`api_required`** | 209 (24.1%) | 145 (69.4%) | 29 (13.9%) | 35 (16.7%) |
| **`external_integration_required`** | 146 (16.8%) | 94 (64.4%) | 31 (21.2%) | 21 (14.4%) |
| **`authentication_required`** | 69 (7.9%) | 44 (63.8%) | 12 (17.4%) | 13 (18.8%) |

### 4.3 Complexity & Design Emphasis

| Dimension | Tier | Total Clean | Train (%) | Val (%) | Test (%) |
|---|---|---|---|---|---|
| **`complexity`** | `medium` | 504 | 349 (69.2%) | 76 (15.1%) | 79 (15.7%) |
| | `high` | 203 | 136 (67.0%) | 34 (16.7%) | 33 (16.3%) |
| | `low` | 162 | 124 (76.5%) | 19 (11.7%) | 19 (11.7%) |
| **`design_emphasis`** | `low` | 631 | 437 (69.3%) | 91 (14.4%) | 103 (16.3%) |
| | `medium` | 141 | 105 (74.5%) | 21 (14.9%) | 15 (10.6%) |
| | `high` | 97 | 67 (69.1%) | 17 (17.5%) | 13 (13.4%) |

### 4.4 Task Intent

| Task Intent | Total Clean | Train (%) | Val (%) | Test (%) | Representation Assessment |
|---|---|---|---|---|---|
| **`create`** | 426 | 293 (68.8%) | 66 (15.5%) | 67 (15.7%) | Well-represented. |
| **`debug`** | 147 | 96 (65.3%) | 24 (16.3%) | 27 (18.4%) | Well-represented. |
| **`explain`** | 125 | 98 (78.4%) | 10 (8.0%) | 17 (13.6%) | Well-represented. |
| **`optimize`** | 119 | 79 (66.4%) | 23 (19.3%) | 17 (14.3%) | Well-represented. |
| **`modify`** | 49 | 40 (81.6%) | 6 (12.2%) | 3 (6.1%) | Sparse in Test (3 samples). |
| **`audit`** | 3 | 3 (100.0%) | 0 (0.0%) | 0 (0.0%) | **Unrepresented in Val & Test**. |

> [!CAUTION]
> **Deficient Evaluation Classes**:
> 1. **`task_intent: audit`**: Total sample size is 3. All 3 are allocated to Train. Neither validation nor test contains an example of `audit`. Performance on `audit` cannot be measured empirically in v1.
> 2. **`primary_scope: design_only`**: Only 1 sample in Val and 1 in Test. Metric estimates (F1, Precision, Recall) will be high-variance.
> 3. **`primary_scope: security_only`**: Only 3 samples in Val and 2 in Test.
> 4. **`authentication_required`**: Only 12 positives in Val and 13 in Test.

---

## 5. Artifact Directory Structure

All experiment files are organized in `datasets/implementation_scope/experiments/v1/`:

```
datasets/implementation_scope/experiments/v1/
├── dataset_version_manifest.json          # Complete provenance, schema, and hash manifest
├── prompt_families.csv                    # Clean records annotated with group_id and family_name
├── prompt_families_report.md              # Detailed documentation of all 16 multi-member families
├── train.csv                              # 609 rows (70.1%)
├── val.csv                                # 129 rows (14.8%)
├── test.csv                               # 131 rows (15.1%)
└── experiment_readiness_report.md         # This audit readiness report
```

---

## 6. Recommendations Prior to Baseline Training

1. **Evaluation Caveat**: Acknowledge that while backend and frontend scope predictions can be evaluated with high statistical confidence, `design_only`, `security_only`, and `audit` will produce noisy evaluation metrics until targeted data collection adds at least +40 samples per sparse class.
2. **Loss Function Design**: Use class-weighted multi-class cross-entropy for `primary_scope` and focal loss for `authentication_required` to prevent model collapse to majority classes.
3. **Cross-Validation**: Any k-fold validation must use `GroupKFold` using the provided `group_id` column in `prompt_families.csv`.

# Objective 2 Architecture: Multi-Model Comparison & Ranking Engine

## PromptShield AI — Architectural Specification

*Audited & Current: October 2026*  
*Component: Objective 2 Engine (`analyzers/model_comparison/` & `analyzers/ide_comparison/`)*

---

## 1. High-Level Architecture Overview

Objective 2 provides a **100% offline, local Multi-Model Comparison and Multi-Criteria Recommendation Engine** for prompts used with foundation models, IDE coding assistants, and AI application builders.

```text
                                    USER PROMPT
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     FastAPI Router & Schema Input     │
                     └───────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ Prompt Feature Extraction │                   │   Local Task Classifier   │
   │ - Code density            │                   │   - 6 domain categories   │
   │ - Structural complexity   │                   │   - Confidence scores     │
   │ - Security sensitivity    │                   │   - Uncertainty metrics   │
   │ - Expected output size    │                   │                           │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │       Model & Product Registry        │
                     │       - Category Isolation            │
                     │       - Active vs. Deprecated         │
                     │       - Verified October 2026 Pricing │
                     │       - Context Windows (128k - 2M)   │
                     └───────────────────┬───────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
           ▼                             ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│ Category A:         │       │ Category B:         │       │ Category C:         │
│ Foundation Models   │       │ IDE Assistants      │       │ AI App Builders     │
│ (21 Active Models)  │       │ (4 Verified Tools)  │       │ (4 Verified Tools)  │
└──────────┬──────────┘       └──────────┬──────────┘       └──────────┬──────────┘
           │                             │                             │
           ▼                             ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│ • Token Adapters:   │       │ • Workflow Criteria:│       │ • Product Criteria: │
│   - OpenAI Exact    │       │   - Multi-File Agent│       │   - Full-Stack      │
│   - Gemini Calib.   │       │   - Repo Indexing   │       │   - UI Quality      │
│   - Claude Calib.   │       │   - Terminal Exec   │       │   - Deployment      │
│ • Cost Breakdown:   │       │   - Privacy / Trust │       │   - Code Export     │
│   - Input / Output  │       │ • Subscription Model│       │ • Credit / Tier Plan│
│ • Suitability Fit   │       │   ($0, $10-$40/mo)  │       │   ($0, $20-$95/mo)  │
└──────────┬──────────┘       └──────────┬──────────┘       └──────────┬──────────┘
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Multi-Criteria Decision Making (MCDM) │
                     │ • Weighted Sum Model (WSM)            │
                     │ • TOPSIS (Ideal Euclidean Distance)   │
                     │ • Presets (Balanced, Cost, Quality)   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Transparent Report & Rationale       │
                     │  • Strictly Monotonic Rankings        │
                     │  • Evidenced Scoring Explanations     │
                     │  • Provenance Disclosures             │
                     └───────────────────────────────────────┘
```

---

## 2. Core Architectural Principles

1. **Category Isolation**:
   Foundation models (`foundation_model`), IDE coding assistants (`ide_assistant`), and AI app builders (`app_builder`) are fundamentally different product tiers. They are never ranked against each other in the same candidate pool.
2. **Transparent Provenance**:
   Every score explicitly declares its mathematical basis. What cannot be measured offline (such as live network latency or generation throughput) is marked `unavailable` rather than fabricated.
3. **Strict Lifecycle Enforcement**:
   Retired, shut down, and preview models are excluded from the default recommendation pool. Users must explicitly opt-in (`include_deprecated=True` or `include_preview=True`).
4. **Logarithmic Cost Normalization**:
   Linear scoring fails when comparing models priced at $0.10/M with models priced at $30.00/M. PromptShield AI uses logarithmic scaling to ensure fair multi-criteria trade-offs.

---

## 3. Component Breakdown

### 3.1 Feature Extractor (`analyzers/model_comparison/feature_extractor.py`)
Computes prompt characteristics:
- `char_count`, `word_count`, `line_count`
- `code_block_count`, `code_proportion`
- `has_sql`, `has_react`, `has_api`, `has_docker`, `has_regex`
- `security_sensitivity` (scans for credentials, tokens, or exploit signatures)
- `expected_output_tokens` (calibrated heuristic based on task directives)

### 3.2 Tokenizer Adapters (`analyzers/model_comparison/tokenizer_adapters.py`)
Provides transparent token estimation:
- **OpenAI Adapter**: Uses `tiktoken` with `o200k_base` and `cl100k_base` BPE encodings. Exact local token counting.
- **Google Adapter**: Calibrated SentencePiece heuristic (`char_ratio = 3.85`). Declared as calibrated estimate.
- **Anthropic Adapter**: Calibrated Claude BPE heuristic (`char_ratio = 3.90`). Declared as calibrated estimate.
- **DeepSeek & xAI Adapters**: Calibrated BPE estimates matching official tokenizer documentation.

### 3.3 Cost Calculator (`analyzers/model_comparison/cost_calculator.py`)
Calculates financial cost for each model candidate:
- `input_cost_usd` = $(\text{Input Tokens} / 1,000,000) \times P_{\text{input}}$
- `estimated_output_cost_usd` = $(\text{Estimated Output Tokens} / 1,000,000) \times P_{\text{output}}$
- `total_cost_usd` = $\text{Input Cost} + \text{Output Cost}$
- Missing or unknown prices are flagged with `has_unverified_pricing=True` and assigned maximum penalty.

### 3.4 Suitability Engine (`analyzers/model_comparison/suitability.py`)
Calculates domain alignment score $\in [0.0, 1.0]$:
- Combines model capability tiers, coding aptitude, reasoning aptitude, and context headroom.
- Penalizes small/lite models on complex architectures.
- Rewards frontier models on complex math, reasoning, and security audits.

### 3.5 Multi-Criteria Ranker (`analyzers/model_comparison/ranker.py`)
Executes ranking algorithms:
- **WSM (Weighted Sum Model)**: Linear combination of normalized benefit and cost criteria.
- **TOPSIS**: Relative closeness to positive-ideal and negative-ideal solutions.
- **Preset Weight Profiles**:
  - `balanced`: $w_{\text{suitability}} = 0.45$, $w_{\text{cost}} = 0.35$, $w_{\text{context}} = 0.20$
  - `cost_optimized`: $w_{\text{suitability}} = 0.20$, $w_{\text{cost}} = 0.70$, $w_{\text{context}} = 0.10$
  - `quality_optimized`: $w_{\text{suitability}} = 0.80$, $w_{\text{cost}} = 0.05$, $w_{\text{context}} = 0.15$
  - `long_context`: $w_{\text{suitability}} = 0.25$, $w_{\text{cost}} = 0.25$, $w_{\text{context}} = 0.50$

---

## 4. IDE & App Builder Comparison Architecture

### 4.1 IDE Coding Assistant Evaluation (`analyzers/ide_comparison/`)
Evaluated across 7 verified developer workflow dimensions:
1. `multi_file_agent`: Autonomous multi-file code editing and project refactoring.
2. `repo_indexing`: Depth and speed of local codebase semantic search and embedding.
3. `terminal_execution`: Autonomous terminal command synthesis and approval boundaries.
4. `model_flexibility`: Support for switching between frontier foundation models.
5. `ecosystem_integration`: Support across VS Code, JetBrains, Visual Studio, and CLI.
6. `privacy_controls`: Zero data retention agreements and local telemetry policies.
7. `plan_affordability`: Pricing tiers ($0 free, $10–$40/mo pro).

Verified Candidates:
- **`cursor`** (Anysphere)
- **`windsurf`** (Codeium)
- **`github-copilot`** (GitHub / Microsoft)
- **`google-antigravity`** (Google)

### 4.2 AI App Builder Evaluation (`analyzers/model_comparison/registry.py`)
Evaluated across 6 full-stack delivery dimensions:
1. `full_stack_backend`: Support for serverless APIs, auth, and database schemas.
2. `frontend_ui_quality`: Modern UI component generation (Tailwind, React, shadcn/ui).
3. `deployment_hosting`: Native 1-click cloud deployment and custom domain support.
4. `code_export_github`: Two-way git sync and zero-lock-in clean code export.
5. `autonomous_iteration`: Multi-step prompt iteration and visual preview debugging.
6. `pricing_accessibility`: Freemium credit allowance and affordable subscription tiers.

Verified Candidates:
- **`v0-dev`** (Vercel)
- **`bolt-new`** (StackBlitz)
- **`lovable-dev`** (Lovable)
- **`replit-agent`** (Replit)

---

## 5. Security & Verification Guarantees

1. **100% Deterministic Execution**: Repeated comparison of the same prompt yields bit-identical scores and rankings.
2. **Strictly Monotonic Ranks**: Ranked output is guaranteed to have unique sequential ranks $1, 2, \dots, N$ ordered by descending scores.
3. **No Zero-Cost Free Rides**: Unpriced models receive maximum cost penalties to prevent artificial ranking elevation.
4. **Offline Isolation**: Comparison engine has zero network dependencies; runs seamlessly in air-gapped environments.

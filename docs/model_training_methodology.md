# Model Training Methodology

## PromptShield AI — Machine Learning Architecture & Training Pipelines

*Last Updated: October 2026*  
*Project Repository: `backendprompt`*

---

## 1. Machine Learning Philosophy: Local, Deterministic & Offline-First

PromptShield AI adheres to a strict architectural philosophy: **all prompt analysis, security risk prediction, task classification, and foundation model recommendations must execute 100% locally and offline without sending prompt text to external cloud APIs**.

### Why Local Models Instead of Calling Remote LLMs?
1. **Zero Data Leakage**: Prompts often contain confidential business logic, proprietary architecture designs, or accidental credentials. Running local models guarantees zero third-party exposure.
2. **Deterministic & Auditable**: Probabilities and classification decisions can be traced to concrete mathematical formulas, vector spaces, and decision boundaries rather than nondeterministic generative black boxes.
3. **Sub-Millisecond Inference**: Local TF-IDF and calibrated linear classifiers evaluate prompts in 1–5 milliseconds, whereas remote API calls require 500–3,000 milliseconds of network round-trip latency.
4. **Minimal Resource Footprint**: The entire suite of model artifacts occupies less than 1.5 MB of RAM and requires zero GPU hardware, running effortlessly on local developer workstations and CI/CD pipelines.

---

## 2. Model 1: Local Task Classifier

### 2.1 Problem Formulation
Given an arbitrary input prompt $P$, predict its primary task domain $Y \in \{C_1, C_2, \dots, C_6\}$:
- `code_generation`: Software implementation, API design, algorithms, scripts, refactoring.
- `security_analysis`: Code audits, vulnerability assessment, threat modeling, exploit analysis.
- `data_analysis`: Data wrangling, statistics, visualization, pandas, SQL aggregations, EDA.
- `creative_and_writing`: Storytelling, poetry, marketing copy, fiction, character design.
- `reasoning_and_math`: Mathematical proofs, calculus, linear algebra, logic puzzles.
- `general_instruction`: Text formatting, translation, summaries, checklists, general Q&A.

### 2.2 Feature Representation & Pipeline
The model is implemented as a scikit-learn pipeline:

$$\mathbf{x} = \text{TF-IDF}(P) \in \mathbb{R}^{V}$$

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        solver="liblinear",
        C=1.5,
        class_weight="balanced",
        random_state=42
    ))
])
```

- **Sublinear TF**: Applies $1 + \log(\text{tf})$ to dampen the effect of repetitive keyword stuffing.
- **Bi-grams `(1, 2)`**: Captures critical domain phrases such as `"sql query"`, `"unit test"`, `"threat model"`, `"data frame"`.
- **Balanced Class Weights**: Automatically adjusts weights inversely proportional to class frequencies to prevent bias.

### 2.3 Uncertainty Modeling & Fallback Heuristics
Predictions are accompanied by a calibrated confidence score and uncertainty indicators:
- **`confident`**: $\max_k P(Y=k|P) \ge 0.50$
- **`marginal`**: $0.35 \le \max_k P(Y=k|P) < 0.50$
- **`low`**: $\max_k P(Y=k|P) < 0.35$

If the model is unavailable or confidence falls below the threshold, a deterministic keyword-based rule heuristic acts as a fallback to guarantee zero unhandled runtime exceptions.

---

## 3. Model 2: Prompt Injection & Adversarial Detector

### 3.1 Problem Formulation
Given an input prompt $P$, estimate the posterior probability that $P$ is an adversarial jailbreak or prompt injection:

$$\hat{y} = P(\text{Malicious} \mid P) \in [0.0, 1.0]$$

Decision Rule:
$$\hat{Y} = \begin{cases} 1 \ (\text{Malicious / Jailbreak}), & \text{if } \hat{y} \ge \tau \\ 0 \ (\text{Benign}), & \text{if } \hat{y} < \tau \end{cases}$$
where the decision threshold is calibrated to $\tau = 0.50$.

### 3.2 Leakage-Free Training Methodology
The training pipeline (`training/train_injection_detector.py`) strictly enforces pre-split dataset hygiene:

```text
               Raw MPDD Dataset
                      │
                      ▼
        Pre-Split Cleaning & Deduplication
                      │
                      ▼
         Cleaned Base Dataset (N=34,860)
                      │
         ┌────────────┴────────────┐
         │ (80% Stratified)        │ (20% Held-Out)
         ▼                         ▼
   Train Split Fold          Test Split Fold
         │                         │
         ▼ (Synthetic Augmentation)│ (Zero Augmentations,
  Augmented Train Set              │  Zero Synthetic Samples)
         │                         │
         ▼                         ▼
    Model Training         Evaluation & Metrics
```

1. **Strict Deduplication**: All prompt templates with identical text are unified prior to splitting.
2. **Train/Test Splitting**: Stratified splitting on label $y$ ensures identical class proportions.
3. **No Test Leakage**: Diverse unique benign augmentations (coding algorithms, data structures) are added exclusively to the training split.
4. **Independent Benchmark Suite**: The model is further evaluated against `tests/data/benchmark_evaluation_suite.json`, which has zero overlap with both training and test sets.

---

## 4. Model 3: Prompt Complexity Scoring Model

### 4.1 Problem Formulation
Tasks exhibit vastly different architectural complexity tiers:
- **`simple`**: 1-line syntax questions, standard definitions, basic lookups.
- **`moderate`**: Single-file scripts, basic CRUD endpoints, standard functions.
- **`complex`**: Multi-module architectures, concurrency, database schema design, security audits.
- **`extreme`**: Distributed consensus, formal verification, low-level lock-free memory, compiler design.

### 4.2 Feature Space & Discretization
The model extracts both dense structural indicators and sparse lexical features:
1. **Lexical Dimensions**: Character length, word count, line count.
2. **Syntax Density**: Code fences, function signatures (`def`, `class`, `fn`, `struct`), SQL keywords, regex literals.
3. **Directive Depth**: Multi-part condition clauses (`"ensure"`, `"must"`, `"without"`, `"transaction"`, `"rollback"`).
4. **Domain Complexity Indicators**: Concurrency primitives, distributed protocols, formal logic proofs.

Pipeline:
```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        solver="liblinear",
        C=2.0,
        class_weight="balanced",
        random_state=42
    ))
])
```

The output tier directly informs the foundation model suitability engine, preventing small models from being recommended for extreme distributed systems, and preventing expensive frontier models from being recommended for trivial syntax lookups.

---

## 5. Model 4: Multi-Criteria Ranker Optimization (WSM & TOPSIS)

### 5.1 Category Isolation
PromptShield AI strictly isolates product categories during ranking:
- **Category A**: Foundation Models (evaluated on token costs, suitability, and context headroom).
- **Category B**: IDE Assistants (evaluated on multi-file agents, repo indexing, terminal execution, and subscription pricing).
- **Category C**: AI App Builders (evaluated on full-stack backend, UI quality, deployment, and code export).

### 5.2 Weighted Sum Model (WSM)
For candidate model $i$ with normalized criteria scores $\bar{x}_{ij}$ and weights $w_j$:

$$S_i = \sum_{j=1}^{M} w_j \cdot \bar{x}_{ij}, \quad \text{where } \sum_{j=1}^M w_j = 1.0$$

#### Logarithmic Cost Normalization
Because foundation model pricing spans three orders of magnitude ($0.075 to $75.00 per 1M tokens), linear normalization disproportionately compresses budget models. PromptShield AI uses logarithmic cost normalization:

$$\bar{x}_{\text{cost}, i} = 1.0 - \frac{\ln(1 + \text{Cost}_i) - \ln(1 + \text{Cost}_{\min})}{\ln(1 + \text{Cost}_{\max}) - \ln(1 + \text{Cost}_{\min})}$$

### 5.3 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
Calculates Euclidean distance to the Positive Ideal Solution ($A^+$) and Negative Ideal Solution ($A^-$):

$$D_i^+ = \sqrt{\sum_{j=1}^{M} \left( v_{ij} - v_j^+ \right)^2}, \qquad D_i^- = \sqrt{\sum_{j=1}^{M} \left( v_{ij} - v_j^- \right)^2}$$

Relative Closeness Score:
$$C_i = \frac{D_i^-}{D_i^+ + D_i^-} \in [0.0, 1.0]$$

Where:
- $C_i \to 1.0$ indicates candidate is closest to the ideal solution.
- $C_i \to 0.0$ indicates candidate is closest to the worst possible solution.

### 5.4 Missing Price Penalty
Any model candidate with unverified, missing, or preview pricing is penalized with the maximum cost penalty ($\text{Score}_{\text{cost}} = 0.0$) and flagged with `has_unverified_pricing=True`. It is never treated as free or lowest cost.

---

## 6. Model Artifacts & File Structure

All trained artifacts are persisted using `joblib` with structured metadata:

```text
artifacts/
├── task_classifier/
│   ├── model.pkl               # Trained Pipeline & Class Mappings
│   ├── metadata.json           # Model metadata, version, and parameters
│   └── metrics.json            # Held-out evaluation metrics report
│
├── injection_detector/
│   ├── model.pkl               # Calibrated Injection Detector Pipeline
│   ├── metadata.json           # Model hyperparameters and decision threshold
│   └── metrics.json            # Confusion matrix, Precision, Recall, ROC-AUC
│
├── complexity_model/
│   ├── model.pkl               # 4-tier Complexity Classifier Pipeline
│   ├── metadata.json           # Tier definitions and feature config
│   └── metrics.json            # Evaluation metrics across complexity tiers
│
└── ranker/
    └── ranker_weights.json     # Calibrated weight matrices & preset distributions
```

# Dataset and Research Sources

## PromptShield AI — Data Ecosystem & Research Provenance

*Last Audited: October 2026*  
*Project Repository: `backendprompt`*

---

## 1. Executive Summary

PromptShield AI relies on a curated, auditable, and ethically sourced data ecosystem to train local machine learning models and validate offline rule-based heuristic analyzers. The platform's machine learning pipelines address three distinct problem spaces:

1. **Prompt Injection & Adversarial Jailbreak Detection** (Binary classification on user prompts)
2. **Software Architecture Attack Surface Prediction** (Multi-label classification of systems specified by prompts)
3. **Task Domain & Complexity Classification** (Multi-class classification for intelligent foundation model matching)

To eliminate data leakage, spurious feature correlations, and dataset poisoning, PromptShield AI enforces strict pre-split deduplication, explicit labeling policies, and independent benchmark suites.

---

## 2. Dataset Manifest & Catalog

All datasets are cataloged in `data/dataset_manifest.json` with cryptographic SHA-256 checksums and verified record counts.

| Dataset Identifier | Path | Records | Tier | Format | License | Primary Target |
|---|---|---|---|---|---|---|
| `mpdd_clean` | `data/processed/mpdd_clean.csv` | 34,860 | Processed | CSV | CC BY-NC 4.0 | Prompt Injection & Jailbreaks |
| `web_security_payloads_clean` | `data/raw/web_security_payloads_clean.json` | 505 | Raw | JSON | MIT | Web Exploits & Attack Signatures |
| `web_security_payloads` | `data/processed/web_security_payloads.csv` | 505 | Processed | CSV | MIT | Vulnerability Classification |
| `security_risk_dataset` | `data/security_risk_dataset.csv` | 296 | Processed | CSV | Proprietary | 15 Application Attack Surfaces |
| `owasp_llm_top10_samples` | `data/external/owasp_llm_top10_samples.json` | 10 | External | JSON | CC BY-SA 4.0 | OWASP LLM Top 10 Reference |
| `benchmark_evaluation_suite` | `tests/data/benchmark_evaluation_suite.json` | 30 | Evaluation | JSON | Test Suite | Golden Evaluation Benchmark |

---

## 3. Deep Dive: Malicious Prompt Dataset for Defenses (MPDD)

### 3.1 Dataset Overview
- **Source**: Malicious Prompt Dataset for Defenses (MPDD Academic Benchmark, IEEE / arXiv:2401.xxxxx).
- **Target Variable**: `isMalicious` (0 = Benign, 1 = Malicious / Jailbreak / Prompt Injection).
- **Raw Volume**: ~139,000 raw scraped rows.
- **Clean Volume**: 34,860 unique, deduplicated, verified rows.

### 3.2 Cleaning & Preprocessing Pipeline
Raw scraped prompt datasets frequently contain significant label noise, scraping artifacts, and synthetic duplicates. The data preprocessing pipeline (`data/clean_dataset.py` and `data/process_security_data.py`) applied the following transformations:

1. **Deduplication**: Exact string deduplication eliminated thousands of repetitive prompt injection templates.
2. **Unicode Normalization & Strip**: Cleaned invisible formatting characters, zero-width spaces, and control characters while preserving intended adversarial unicode mutations.
3. **Label Repair Protocol**:
   - Fixed mislabeled benign technical questions (e.g. AG News samples and StackOverflow JavaScript/HTML code snippets that were incorrectly flagged as malicious in the raw dataset).
   - Re-labeled 1,200+ benign technical instructions to class 0 to prevent the model from penalizing standard developer queries.
4. **Leakage-Free Train/Test Split**:
   - The cleaned base MPDD dataset is partitioned into an 80% train split and 20% held-out test split using stratified sampling before any synthetic data augmentations are introduced.
   - Augmentation samples (`BENIGN_AUGMENTATIONS`) are appended strictly to the training fold; the held-out test fold contains zero synthetic samples.

---

## 4. Deep Dive: Application Security Risk Dataset

### 4.1 Taxonomy of 15 Architectural Attack Surfaces
Unlike prompt injection (which attacks the LLM itself), the Application Risk Model predicts vulnerabilities that would arise in software created according to the prompt's specifications.

1. **`sqli`** — SQL Injection via unsanitized queries or missing ORM parameterization.
2. **`xss`** — Cross-Site Scripting via unsanitized browser rendering or missing CSP.
3. **`ssrf`** — Server-Side Request Forgery via unvalidated outbound HTTP calls.
4. **`idor`** — Insecure Direct Object References via unverified object ownership checks.
5. **`auth_weakness`** — Flawed password hashing, lack of MFA, or missing rate limiting.
6. **`authz_weakness`** — Missing or bypassable Role-Based Access Control (RBAC).
7. **`file_upload`** — Unrestricted file extensions, missing magic-byte verification.
8. **`path_traversal`** — Arbitrary file read/write via unsanitized file paths.
9. **`command_injection`** — Subshell command execution via `shell=True` or `exec`.
10. **`csrf`** — Cross-Site Request Forgery via missing synchronizer tokens or lax cookies.
11. **`sensitive_data`** — Plaintext secrets, exposed API keys, or unencrypted storage.
12. **`prompt_injection`** — Application passing untrusted inputs directly to downstream LLMs.
13. **`data_poisoning`** — Unvetted user training data ingested into ML pipelines.
14. **`model_extraction`** — Unrestricted inference queries enabling model stealing.
15. **`ai_tool_risk`** — Unrestricted agent tools executing destructive actions without approval.

### 4.2 Labeled Dataset Structure
- **Path**: `data/security_risk_dataset.csv`
- **Columns**: `prompt`, `sqli`, `xss`, `ssrf`, `idor`, `auth_weakness`, `authz_weakness`, `file_upload`, `path_traversal`, `command_injection`, `csrf`, `sensitive_data`, `prompt_injection`, `data_poisoning`, `model_extraction`, `ai_tool_risk`.
- **Validation**: Verified through `data/audit_security_dataset.py` ensuring non-zero positive and negative support for every label.

---

## 5. OWASP Top 10 for LLM Applications (2025/2026 Edition)

PromptShield AI maps its defense and evaluation capabilities directly to the OWASP Top 10 for Large Language Model Applications:

| OWASP LLM ID | Vulnerability Name | PromptShield Analyzer / Defense Component |
|---|---|---|
| **LLM01** | Prompt Injection | `analyzers/vulnerability_detection/injection_detector.py` (MPDD model) |
| **LLM02** | Sensitive Information Disclosure | `analyzers/vulnerability_detection/sensitive_data.py` (Automated secret redaction) |
| **LLM03** | Supply Chain Vulnerabilities | `analyzers/security_requirements/controls_extractor.py` (Dependency verification) |
| **LLM04** | Data and Model Poisoning | `training/train_injection_detector.py` (Cleaned datasets, no train-test leakage) |
| **LLM05** | Improper Output Handling | `optimizer/rules/security_rules.py` (Output sanitization and CSP constraints) |
| **LLM06** | Excessive Agency | `analyzers/ide_comparison/` & `models/schemas.py` (Tool execution approval hooks) |
| **LLM07** | System Prompt Leakage | `optimizer/templates/threat_modeling.py` (Delimiter sandboxing & hierarchy rules) |
| **LLM08** | Vector & Embedding Weaknesses | `analyzers/model_comparison/feature_extractor.py` (Structural prompt analysis) |
| **LLM09** | Misinformation / Hallucination | `analyzers/model_comparison/suitability.py` (Domain-specific reasoning aptitude) |
| **LLM10** | Unbounded Consumption | `analyzers/model_comparison/cost_calculator.py` (Token limits & cost modeling) |

---

## 6. External Research & Foundation Model Benchmarks

### 6.1 Provider API Documentation Sources (Audited October 2026)
- **OpenAI**: [OpenAI Platform Model Documentation](https://platform.openai.com/docs/models) (GPT-5.5, GPT-6 Astra/Sol/Luna context and pricing).
- **Google**: [Google AI Developer Documentation](https://ai.google.dev/gemini-api/docs/models) (Gemini 4 Argon, Gemini 3.8 Flash, Gemini 3.5 Flash context and pricing).
- **Anthropic**: [Anthropic Claude Model Overview](https://docs.anthropic.com/en/docs/about-claude/models) (Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5 pricing).
- **DeepSeek**: [DeepSeek API Documentation](https://api-docs.deepseek.com/quick_start/pricing) (DeepSeek-V4-Pro, DeepSeek-Flash pricing).
- **xAI**: [xAI Developer Documentation](https://docs.x.ai/docs/models) (Grok 4.7, Grok 4.3 pricing).

### 6.2 IDE & App Builder Industry Research
- **Cursor** (Anysphere): VS Code fork with Composer, background indexing, and model routing.
- **Windsurf** (Codeium): Continuous flow IDE with Cascade Fast Context and terminal execution.
- **GitHub Copilot** (Microsoft / GitHub): Ecosystem integration across VS Code, Visual Studio, JetBrains, and Xcode.
- **Google Antigravity**: Autonomous multi-surface agent architecture with verifiable artifacts.
- **Vercel v0**: React, Next.js, and shadcn/ui frontend component generator.
- **Bolt.new** (StackBlitz): WebContainer browser-based full-stack runtime.
- **Lovable.dev**: Supabase-integrated full-stack web application builder.
- **Replit Agent**: Autonomous Linux cloud container development platform.

---

## 7. Data Governance & Ethical Disclosures

1. **Non-Invasive Execution**: All datasets are stored locally in text, JSON, and CSV formats. No arbitrary code execution or deserialization of untrusted binaries is performed during dataset ingestion.
2. **Responsible Security Data Use**: The malicious prompt samples and exploit payloads in `data/raw/` and `data/processed/` are compiled strictly for defensive classifier training and threat modeling. They are never transmitted to external third-party APIs.
3. **Privacy Compliance**: All personal email addresses, API tokens, and credentials in the raw research data have been masked or removed in the processed tier.

# PromptShield AI — Dataset Card

**Dataset Collection:** PromptShield AI: Architectural Security Risks & Prompt Safeguards Benchmark  
**Publisher:** PromptShield AI Research Team  
**Platform:** Kaggle Datasets  
**Last Updated:** November 2026  
**Draft Version:** 1.0 (Stage 2 Review Draft)  

---

## 1. Dataset Overview & Motivation

### Title & Subtitle
* **Title:** PromptShield AI — Architectural Security Risks & Prompt Safeguards Benchmark
* **Subtitle:** Multi-label application security risks from natural-language specifications, curated MPDD label repairs, and synthetic sensitive-data evaluation suite.

### Motivation & Research Gap
Most existing cybersecurity machine learning datasets focus on **source-code vulnerability detection** (e.g. Juliet Suite, Devign, Big-Vul) or **binary prompt injection/jailbreak detection** (e.g. MPDD, In-The-Wild Jailbreaks). 

However, modern developers, AI code-generation agents, and software architects increasingly generate entire systems directly from **natural-language application requirements and prompts**. If an application requirement specifies an architecture with inherent security risks—such as unvalidated file uploads, dynamic SQL query concatenation, or unauthenticated API endpoints—generative coding models frequently generate insecure boilerplate code that reflects those flawed specifications.

**PromptShield AI bridges this critical gap** by providing:
1. **Component A (Prompt Security Quality):** A verified companion patch table repairing 3,283 mislabeled records in the popular MPDD prompt injection corpus, accompanied by a group-aware evaluation pipeline to eliminate template-based train/test leakage.
2. **Component B (Application Security Risk Prediction):** A curated multi-label dataset mapping natural-language software requirements to 15 distinct architectural attack surfaces (SQLi, XSS, SSRF, IDOR, Auth Weakness, Path Traversal, etc.).
3. **Component C (Sensitive Data Detection):** A standardized, 100% synthetic evaluation benchmark containing non-functional dummy credentials and RFC 2606 reserved domains to evaluate secret detection systems without exposing real data.

---

## 2. Dataset Structure & Contents

The downloadable package is organized into dedicated task-specific files:

```text
PromptShield-Kaggle/
│
├── dataset/
│   ├── application_security_risks.csv       # Primary multi-label dataset (321 rows, 15 risk labels)
│   ├── application_security_risks_base.csv  # Historical base dataset (296 rows, unpatched)
│   ├── path_traversal_curated_patch.csv     # Curated patch table (25 rows: 15 vuln, 10 secure)
│   ├── mpdd_curated_label_patches.csv       # Curated MPDD label repair table (3,283 records)
│   ├── sensitive_data_benchmark.csv         # Synthetic sensitive data evaluation suite (100 rows)
│   │
│   └── reference/
│       ├── owasp_llm_top10_samples.json     # OWASP GenAI Top 10 compliance reference (10 items)
│       └── benchmark_evaluation_suite.json  # Multi-domain golden benchmark suite (30 items)
```

---

## 3. Data Collection & Creation Methodology

### 3.1 Application Security Risks (`application_security_risks.csv`)
* **Creation Methodology:** **AI-assisted synthetic prompt engineering and templating.**
  * The prompts were generated using structured architectural templates covering common web application patterns (e.g. admin dashboards, document viewers, e-commerce checkouts, AI assistants).
* **Human Validation Notice:**
  * Prompts and label assignments were curated and reviewed internally by project developers during prototyping. **There has been no independent multi-annotator inter-rater reliability study (e.g. Fleiss' Kappa) conducted.**
  * **Critical Distinction:** Labels represent **potential architectural attack surfaces implied by requirements**, not confirmed vulnerabilities in running code. The absence of an explicit security control in a prompt does not constitute proof of vulnerability.
* **Schema Conformance & Patch Flag:**
  * To resolve the 0-positive unlearnable defect for `path_traversal`, 25 AI-proposed enterprise prompts (15 vulnerable, 10 secure; pending formal human review) were appended to create the 321-row flagship table.
  * An explicit column `is_patch_addition` (boolean) allows users to separate the original 296 base rows (`False`) from the 25 patch rows (`True`).

### 3.2 MPDD Curated Label Patches (`mpdd_curated_label_patches.csv`)
* **Context:** In the physical `mpdd_clean.csv` corpus (curated by Mohammed Amine Jebbar), 3,283 records in Class 1 (`isMalicious = 1`) represent non-malicious queries.
* **Safe Index-Based Architecture:** To strictly respect upstream copyrights, the patch table **omits raw prompt text completely**. It distributes only `row_index`, `prompt_sha256_prefix`, `original_isMalicious`, `proposed_isMalicious`, `category`, `reason`, and `confidence`.
* **Audit Methodology:** Algorithmic regex filtering followed by line-by-line categorization:
  - 2,896 standard AG News categorization tasks flipped to Benign (`0`).
  - 223 StackOverflow coding questions flipped to Benign (`0`).
  - 163 security-topic news articles reporting on historical events flipped to Benign (`0`).
  - **Row 10632 ("Neuromancer"):** True adversarial jailbreak containing weaponized XSS strictly preserved as Malicious (`1`).
* **Distribution Strategy:** To respect upstream platform terms, the raw 39k-row text corpus is **not redistributed directly**. Instead, our companion patch table is provided alongside a Kaggle Notebook that mounts the upstream dataset directly via platform inputs and applies patches in memory.

### 3.3 Synthetic Sensitive Data Benchmark (`sensitive_data_benchmark.csv`)
* **Creation Methodology:** **100% synthetic, programmatic generation.**
* **Ethical Compliance:**
  - Zero live credentials, real API tokens, or actual passwords.
  - Standardized fake prefixes: `sk-test-PROMPTSHIELD-DEMO-XXXX`, `AKIAEXAMPLE...`.
  - All email addresses use RFC 2606 reserved domains (`@example.com`, `@example.org`, `.test`).
  - All phone numbers use the North American Numbering Plan reserved fictional range (`555-0100` to `555-0199`).
  - Every row carries an explicit boolean flag `is_synthetic = True`.

---

## 4. Multi-Label Taxonomy: 15 Architectural Attack Surfaces

| Risk Label | Full Name | Label Definition (Active = 1) | Positive Count ($N=321$) | Positive % |
| :--- | :--- | :--- | :---: | :---: |
| `sqli` | SQL Injection | Requirement describes user-controlled input incorporated into database queries without specifying ORM parameterization. | 32 | 9.97% |
| `xss` | Cross-Site Scripting | Requirement involves displaying user-generated content in a browser without output encoding or CSP headers. | 36 | 11.21% |
| `ssrf` | Server-Side Request Forgery | Server fetches external resources or URLs supplied directly by user input. | 16 | 4.98% |
| `idor` | Insecure Direct Object Reference | Requirement describes object/resource access using user-supplied IDs without verifying object ownership. | 36 | 11.21% |
| `auth_weakness` | Authentication Weakness | Application handles login, registration, or credentials without specifying hashing, rate limiting, or MFA. | 32 | 9.97% |
| `authz_weakness` | Authorization Weakness | Multi-tenant or multi-role system where users manage resources without explicit Role-Based Access Control (RBAC). | 38 | 11.84% |
| `file_upload` | Unrestricted File Upload | Application accepts uploaded files without specifying extension whitelisting, magic-byte checks, or isolated storage. | 34 | 10.59% |
| `path_traversal` | Directory / Path Traversal | Application accesses local disk files using user-controlled filename or path parameters without boundary checks. | **15** | **4.67%** |
| `command_injection` | OS Command Injection | Application executes system shell commands based on user inputs or parameters. | 26 | 8.10% |
| `csrf` | Cross-Site Request Forgery | Application performs authenticated state-changing actions via HTTP requests without CSRF protection tokens. | 27 | 8.41% |
| `sensitive_data` | Sensitive Data Handling | System processes or stores confidential data (credentials, financial records, PII) without specifying encryption. | 49 | 15.26% |
| `prompt_injection` | Downstream Prompt Injection | Application passes untrusted user input directly into downstream LLM context windows without isolation. | 27 | 8.41% |
| `data_poisoning` | ML Data Poisoning | ML pipeline ingests unvetted user-submitted data for continuous model training. | 27 | 8.41% |
| `model_extraction` | Model Extraction / Stealing | System provides unrestricted model query access enabling distillation or model reconstruction. | 26 | 8.10% |
| `ai_tool_risk` | Unrestricted Agent Execution | Autonomous AI agent possesses destructive tool execution privileges without human approval gates. | 29 | 9.03% |

---

## 5. Known Limitations

1. **Compact Sample Size ($N=321$):** While addressing a unique problem domain, 321 records is a compact dataset for 15-class multi-label classification.
2. **Class Imbalance:** Highly specific attack surfaces (`ssrf` with 16 positives and `path_traversal` with 15 positives) have low positive support (~4.7–5.0%). Standard single train/test splits exhibit high evaluation variance; repeated stratified K-fold cross-validation is strongly recommended.
3. **Synthetic Templating Bias:** Prompts are synthetically structured enterprise specifications. Generalization to unstructured user chat or multi-paragraph enterprise RFPs may require additional fine-tuning.
4. **Metadata Missingness:** In `application_security_risks.csv`, metadata columns (`technologies`, `security_controls`, `data_types`, `user_roles`, `external_inputs`) contain ~41% uncollected historical fields, designated as empty strings `""` (`[UNKNOWN_METADATA]`). Baseline classifiers should train strictly on the natural-language `prompt` feature.

---

## 6. Intended Use & Ethical Disclosures

### Intended Use Cases
- Benchmarking NLP and multi-label classifiers that detect security attack surfaces in natural-language specifications.
- Training defensive guardrails for AI coding assistants (e.g. Cursor, Copilot, Antigravity) to warn developers before generating insecure boilerplate.
- Academic research on prompt safety, label noise repair, and template leakage mitigation.

### Out-of-Scope & Misuse
- **Not for Production Vulnerability Auditing:** This dataset must not be used as an automated substitute for professional static code analysis (SAST) or penetration testing.
- **Not for Malicious Exploitation:** Prompts and dummy test vectors are curated strictly for defensive classification.

---

## 7. Suggested Kaggle Tags & Metadata

* **Primary Tags:** `cybersecurity`, `nlp`, `text-classification`, `llm`, `security`, `data-quality`, `software-engineering`
* **License:** Multi-Licensed (CC BY 4.0, CC0 1.0, CC BY-SA 4.0). See `DATA_SOURCES_AND_LICENSES.md`.


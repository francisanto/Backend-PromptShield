# Data Sources and Licensing

**Dataset Collection:** PromptShield AI — Security and Sensitive Data Benchmark  
**Publisher:** PromptShield AI Research Team  
**Platform:** Kaggle Datasets  
**Last Updated:** November 2026  
**Draft Version:** 1.0 (Stage 2 Review Draft)  

---

## 1. Overview & Multi-License Policy

The PromptShield AI dataset collection adheres to a strict multi-licensing framework. Rather than forcing diverse assets under an assumed monolithic license, each dataset in this repository is distributed under its own explicit, legally verified terms:

| Dataset / Asset | File Path | Copyright Holder | Creation Methodology | Published License |
| :--- | :--- | :--- | :--- | :--- |
| **Application Security Risks (v1.1)** | `dataset/application_security_risks.csv` | PromptShield AI (Author) | AI-assisted synthetic application requirements | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Application Security Risks (Base)** | `dataset/application_security_risks_base.csv` | PromptShield AI (Author) | AI-assisted synthetic application requirements | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Path Traversal Curated Patch** | `dataset/path_traversal_curated_patch.csv` | PromptShield AI (Author) | AI-assisted synthetic enterprise prompt design | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **MPDD Curated Label Patches** | `dataset/mpdd_curated_label_patches.csv` | PromptShield AI (Author) | Human & algorithmic dataset quality audit annotations | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Synthetic Sensitive-Data Benchmark** | `dataset/sensitive_data_benchmark.csv` | PromptShield AI (Author) | 100% synthetic dummy tokens & RFC 2606 reserved domains | **Creative Commons Zero 1.0 Universal (CC0 1.0) / Public Domain** |
| **PromptShield Golden Benchmark Suite** | `dataset/reference/benchmark_evaluation_suite.json` | PromptShield AI (Author) | Hand-crafted multi-domain evaluation prompts | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **OWASP Top 10 for LLMs Reference** | `dataset/reference/owasp_llm_top10_samples.json` | OWASP Foundation | Cybersecurity expert consensus threat prompts | **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** |

---

## 2. Dataset-by-Dataset Provenance & Licensing Disclosures

### 2.1 PromptShield Original Datasets (CC BY 4.0)

#### A. Application Security Risks (`application_security_risks.csv` & `application_security_risks_base.csv`)
* **Authorship:** Created by the PromptShield AI team.
* **Creation Methodology:** AI-assisted prompt templating derived from standard software architectural specifications and OWASP/CWE vulnerability categories.
* **Human Validation Notice:** Prompts were curated and reviewed by project engineers. However, the multi-label assignments represent potential architectural attack surfaces inferred from requirements, not ground-truth vulnerabilities verified in running production software.
* **License Grant:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share, copy, redistribute, adapt, remix, transform, and build upon the material for any purpose, even commercially, provided you give appropriate credit to PromptShield AI.

#### B. MPDD Curated Label Patches (`mpdd_curated_label_patches.csv`)
* **Authorship:** Created by the PromptShield AI quality auditing team.
* **Creation Methodology:** Algorithmic regex filtering and individual human review of 3,283 candidate mislabeled records in the third-party MPDD corpus.
* **Safe Index-Based Architecture:** To strictly respect third-party copyrights, the published patch file **omits the raw prompt text completely**. It contains exclusively: `row_index` (matching upstream Kaggle dataset), `prompt_sha256_prefix`, `original_isMalicious`, `proposed_isMalicious`, `category`, `reason`, and `confidence`.
* **Zero Third-Party Text Redistribution:** Users download the upstream MPDD corpus directly from its original Kaggle repository and join the patch table locally in memory.
* **License Grant:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (PromptShield AI holds 100% authorship of these audit annotations).

#### C. Path Traversal Curated Patch (`path_traversal_curated_patch.csv`)
* **Authorship:** Authored by PromptShield AI engineers.
* **Creation Methodology:** AI-assisted formulation of 25 enterprise prompts (15 vulnerable patterns such as dynamic query parameters and Zip Slip, and 10 secure patterns including S3 UUID lookups, whitelist dictionaries, and in-memory streams) designed as a candidate patch to resolve the 0-positive unlearnable class.
* **Review Status:** AI-proposed candidate patch pending formal multi-reviewer human sign-off.
* **License Grant:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

#### D. Golden Benchmark Evaluation Suite (`benchmark_evaluation_suite.json`)
* **Authorship:** Authored by PromptShield AI.
* **Creation Methodology:** 30 standardized test prompts spanning benign coding, static systems, informational questions, explicit jailbreaks, and directive overrides.
* **License Grant:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

### 2.2 Synthetic Benchmark Dataset (CC0 1.0 Universal)

#### Synthetic Sensitive Data Benchmark (`sensitive_data_benchmark.csv`)
* **Authorship:** Generated programmatically by PromptShield AI.
* **Ethical Safeguards & Composition:**
  * Contains **zero real credentials, personal information, or proprietary secrets**.
  * All tokens use standardized, non-functional dummy prefixes (e.g. `sk-test-PROMPTSHIELD-DEMO-`, `AKIAEXAMPLE...`).
  * All email addresses utilize RFC 2606 reserved domains (`@example.com`, `@example.org`, `.test`).
  * All phone numbers utilize the North American Numbering Plan reserved fictional range (`555-0100` to `555-0199`).
  * Every record carries an explicit boolean flag `is_synthetic = True`.
* **License Grant:** Dedicated to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

---

### 2.3 Third-Party Reference Datasets (CC BY-SA 4.0)

#### OWASP Top 10 for LLM Applications Reference Suite (`owasp_llm_top10_samples.json`)
* **Original Creator:** OWASP GenAI Security Project (OWASP Foundation).
* **Source:** [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) (2025/2026 Edition).
* **Taxonomy vs Prompt Authorship:** The taxonomic categories (`LLM01` through `LLM10`) and threat definitions are established by the OWASP Foundation under CC BY-SA 4.0. The individual illustrative prompt strings were formulated by PromptShield AI as synthetic test vectors to model each category (they are not verbatim OWASP documentation extracts).
* **Terms & Attribution:** Distributed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). Attribution is prominently given to the OWASP Foundation.

---

## 3. Excluded Datasets & Upstream Platform Terms

To maintain strict copyright compliance, the following third-party datasets are **not distributed as direct downloadable files** in this collection:

### 3.1 Malicious Prompt Detection Dataset (MPDD)
* **Curator:** Mohammed Amine Jebbar.
* **Kaggle Source:** [`mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd`](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd).
* **License Status:** Designated as **"Other / Unknown"** on Kaggle. Governed by Kaggle's Platform Terms of Service for educational and research evaluation.
* **Academic Reference:** Benchmarked in Hassan et al. (KAUST), *"Efficient and Adaptable Detection of Malicious LLM Prompts via Bootstrap Aggregation"* (arXiv:2602.08062).
* **Distribution Policy:** We do not re-upload Mohammed Amine Jebbar's raw 39k-row dataset. Instead, our Kaggle Notebooks provide reproducible pipelines that import the original Kaggle dataset directly into a user's session and apply our companion label patches in memory.

### 3.2 Web Security Payloads Corpus
* **Curators:** Curated from OWASP testing guides and Exploit-DB (Offensive Security).
* **Exclusion Reason:** Exploit-DB Terms of Service prohibit unauthorized bulk redistribution or database republishing without written permission. Furthermore, the dataset lacks row-level provenance tags separating OWASP entries from Exploit-DB entries.
* **Distribution Policy:** Excluded from the downloadable Kaggle package to guarantee zero copyright ambiguity. Retained internally for offline testing only.

---

## 4. Citation Instructions

If you use the PromptShield AI datasets or curated label patches in academic research, please cite:

```bibtex
@misc{promptshield2026,
  author = {PromptShield AI Team},
  title = {PromptShield AI: Architectural Security Risks and Evaluated Prompt Safeguards Benchmark},
  year = {2026},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/promptshield/promptshield-security-benchmark}},
  note = {Curated multi-label architectural security risk corpus and MPDD label repair annotations}
}
```

For the underlying MPDD dataset, cite the upstream curator and benchmark literature:
```bibtex
@misc{jebbar2024mpdd,
  author = {Jebbar, Mohammed Amine},
  title = {Malicious Prompt Detection Dataset (MPDD)},
  year = {2024},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd}}
}

@article{hassan2025bagel,
  title = {Efficient and Adaptable Detection of Malicious LLM Prompts via Bootstrap Aggregation},
  author = {Hassan, Shayan Ali and Ni, Tao and Qazi, Zafar Ayyub and Canini, Marco},
  journal = {arXiv preprint arXiv:2602.08062},
  year = {2025}
}
```


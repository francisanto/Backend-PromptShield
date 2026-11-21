#!/usr/bin/env python3
"""
PromptShield / Guardvane — Safe Incoming Dataset Validation & Audit Pipeline.
=============================================================================
Validates new CSV datasets placed in data/incoming/ before integration:
1. Schema & Required Columns Verification
2. Missing & Empty Prompt Detection
3. Label Domain & Binary Range Enforcement
4. In-file Exact Duplicate & Contradictory Label Detection
5. Strict Test-Set Leakage Prevention (Exact match & TF-IDF similarity > 0.70)
6. Training-Set Redundancy Auditing (Exact duplicate & high similarity > 0.85)
7. Provenance Tracking & Non-destructive Output Reporting

Outputs:
- Accepted records: data/clean/validated_incoming/<filename_stem>_accepted.csv
- Rejected records: data/review/<filename_stem>_rejected.csv (with rejection reasons)
- Audit report:     data/reports/validation_<filename_stem>.json & .md
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Model configurations
MODEL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "task_classifier": {
        "name": "Task Classifier",
        "prompt_cols": ["prompt"],
        "required_label_cols": ["task_category"],
        "allowed_labels": {
            "task_category": [
                "code_generation",
                "security_analysis",
                "data_analysis",
                "creative_and_writing",
                "reasoning_and_math",
                "general_instruction",
            ]
        },
        "binary_cols": [],
    },
    "application_risk": {
        "name": "Application Risk Model",
        "prompt_cols": ["prompt"],
        "required_label_cols": [
            "sqli", "xss", "ssrf", "idor", "auth_weakness", "authz_weakness",
            "file_upload", "path_traversal", "command_injection", "csrf",
            "sensitive_data", "prompt_injection", "data_poisoning",
            "model_extraction", "ai_tool_risk"
        ],
        "allowed_labels": {},
        "binary_cols": [
            "sqli", "xss", "ssrf", "idor", "auth_weakness", "authz_weakness",
            "file_upload", "path_traversal", "command_injection", "csrf",
            "sensitive_data", "prompt_injection", "data_poisoning",
            "model_extraction", "ai_tool_risk"
        ],
    },
    "complexity": {
        "name": "Complexity Model",
        "prompt_cols": ["prompt"],
        "required_label_cols": ["complexity_tier"],
        "allowed_labels": {
            "complexity_tier": ["simple", "moderate", "complex", "extreme"]
        },
        "binary_cols": [],
    },
    "implementation_scope": {
        "name": "Implementation Scope Predictor",
        "prompt_cols": ["enhanced_prompt", "prompt"],
        "required_label_cols": [
            "primary_scope",
            "frontend_required",
            "backend_required",
            "database_required",
            "api_required",
            "authentication_required",
            "external_integration_required",
            "design_emphasis",
            "complexity",
            "task_intent",
        ],
        "allowed_labels": {
            "primary_scope": [
                "backend", "frontend", "full_stack", "design_only", "security_only", "unclear"
            ],
            "design_emphasis": ["low", "medium", "high"],
            "complexity": ["low", "medium", "high"],
            "task_intent": [
                "create", "modify", "debug", "optimize", "audit", "explain", "unclear", "test", "refactor"
            ],
        },
        "binary_cols": [
            "frontend_required",
            "backend_required",
            "database_required",
            "api_required",
            "authentication_required",
            "external_integration_required",
        ],
    },
}


def sha256_file(filepath: Path) -> str:
    """Calculates SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_reference_datasets(model_key: str) -> Tuple[pd.Series, pd.Series]:
    """
    Loads the exact reference training and test prompt sets for leakage detection.
    Returns (train_prompts, test_prompts).
    """
    if model_key == "task_classifier":
        from ml.train_task_classifier import TASK_DATASET
        df = pd.DataFrame(TASK_DATASET, columns=["prompt", "task_category"])
        X_tr, X_te, _, _ = train_test_split(
            df["prompt"], df["task_category"], test_size=0.20, stratify=df["task_category"], random_state=42
        )
        return X_tr.reset_index(drop=True), X_te.reset_index(drop=True)

    elif model_key == "application_risk":
        from ml.train_security_model import RISK_LABELS
        p = REPO_ROOT / "data" / "security_risk_dataset.csv"
        df = pd.read_csv(p)
        X_tr, X_te, _, _ = train_test_split(
            df["prompt"], df[RISK_LABELS], test_size=0.20, random_state=42
        )
        return X_tr.reset_index(drop=True), X_te.reset_index(drop=True)

    elif model_key == "complexity":
        from training.train_complexity_model import COMPLEXITY_DATASET
        df = pd.DataFrame(COMPLEXITY_DATASET, columns=["prompt", "complexity_tier"])
        X_tr, X_te, _, _ = train_test_split(
            df["prompt"], df["complexity_tier"], test_size=0.20, stratify=df["complexity_tier"], random_state=42
        )
        return X_tr.reset_index(drop=True), X_te.reset_index(drop=True)

    elif model_key == "implementation_scope":
        tr_p = REPO_ROOT / "data" / "splits" / "train.csv"
        te_p = REPO_ROOT / "data" / "splits" / "test.csv"
        b1_p = REPO_ROOT / "data" / "clean" / "implementation_scope_batch_01_clean.csv"
        
        tr_df = pd.read_csv(tr_p)
        te_df = pd.read_csv(te_p)
        b1_df = pd.read_csv(b1_p) if b1_p.exists() else pd.DataFrame()
        
        train_prompts = pd.concat([tr_df["enhanced_prompt"], b1_df.get("enhanced_prompt", pd.Series())], ignore_index=True)
        test_prompts = te_df["enhanced_prompt"]
        return train_prompts.reset_index(drop=True), test_prompts.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def auto_detect_model_type(df: pd.DataFrame) -> Optional[str]:
    """Infers target model from DataFrame columns."""
    cols = set(df.columns)
    if "task_category" in cols:
        return "task_classifier"
    if {"sqli", "xss", "ssrf"}.issubset(cols):
        return "application_risk"
    if "complexity_tier" in cols:
        return "complexity"
    if "primary_scope" in cols or "frontend_required" in cols:
        return "implementation_scope"
    return None


def validate_incoming_dataset(
    csv_path: Path,
    target_model: Optional[str] = None,
    test_leakage_threshold: float = 0.70,
    train_similarity_threshold: float = 0.85,
    min_prompt_length: int = 10,
) -> Dict[str, Any]:
    """
    Executes a comprehensive validation audit on an incoming dataset.
    """
    csv_path = Path(csv_path).resolve()
    print("=" * 80)
    print(f"PROMPTSHIELD / GUARDVANE — INCOMING DATA VALIDATION HARNESS")
    print(f"File: {csv_path.name}")
    print("=" * 80)

    if not csv_path.exists():
        raise FileNotFoundError(f"Incoming CSV file does not exist: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")

    total_records = len(df)
    file_sha256 = sha256_file(csv_path)
    print(f"Loaded {total_records:,} rows | File Hash: {file_sha256[:16]}...")

    # Detect model type
    if not target_model:
        target_model = auto_detect_model_type(df)
        if not target_model:
            raise ValueError(
                f"Could not auto-detect target model from columns: {list(df.columns)}. "
                f"Please specify --model with one of {list(MODEL_SCHEMAS.keys())}"
            )

    schema = MODEL_SCHEMAS[target_model]
    model_name = schema["name"]
    print(f"Target Model: {model_name} (`{target_model}`)")

    # Identify prompt column
    prompt_col = None
    for cand in schema["prompt_cols"]:
        if cand in df.columns:
            prompt_col = cand
            break

    if not prompt_col:
        raise ValueError(
            f"Missing required prompt column for {model_name}. Looked for {schema['prompt_cols']} in {list(df.columns)}"
        )
    print(f"Identified prompt column: '{prompt_col}'")

    # Load frozen reference splits for leakage detection
    ref_train_prompts, ref_test_prompts = load_reference_datasets(target_model)
    print(f"Loaded Reference Datasets: Train = {len(ref_train_prompts):,} prompts | Test = {len(ref_test_prompts):,} prompts (Frozen)")

    # Setup Vectorizers for Leakage and Similarity Audits
    leakage_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=15000)
    all_known_prompts = pd.concat([ref_train_prompts, ref_test_prompts], ignore_index=True)
    leakage_vectorizer.fit(all_known_prompts)

    ref_test_vecs = leakage_vectorizer.transform(ref_test_prompts)
    ref_train_vecs = leakage_vectorizer.transform(ref_train_prompts)

    ref_test_exact_set = set(ref_test_prompts.str.strip().str.lower())
    ref_train_exact_set = set(ref_train_prompts.str.strip().str.lower())

    accepted_records = []
    rejected_records = []
    flagged_records = []

    seen_prompts_in_batch: Dict[str, Any] = {}
    rejection_reasons_counter: Dict[str, int] = {}

    def log_rejection(idx: int, row_data: Dict, reason: str, details: str = ""):
        rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
        record = dict(row_data)
        record["_row_index"] = idx
        record["_rejection_reason"] = reason
        record["_rejection_details"] = details
        rejected_records.append(record)

    # Validate each row
    print("\nAuditing individual records for schema, nulls, duplicates, and test leakage...")
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        prompt_val = row.get(prompt_col)

        # 1. Null / Empty prompt check
        if pd.isna(prompt_val) or not str(prompt_val).strip():
            log_rejection(idx, row_dict, "EMPTY_PROMPT", "Prompt is null, empty, or whitespace-only")
            continue

        prompt_str = str(prompt_val).strip()
        prompt_norm = prompt_str.lower()

        # 2. Minimum length check
        if len(prompt_str) < min_prompt_length:
            log_rejection(idx, row_dict, "TOO_SHORT", f"Length {len(prompt_str)} < {min_prompt_length} chars")
            continue

        # 3. Label column presence & type validation
        schema_fail = False
        for req_col in schema["required_label_cols"]:
            if req_col not in df.columns:
                log_rejection(idx, row_dict, "MISSING_REQUIRED_COLUMN", f"Required column '{req_col}' missing")
                schema_fail = True
                break
            val = row.get(req_col)
            if pd.isna(val):
                log_rejection(idx, row_dict, "NULL_LABEL", f"Label column '{req_col}' is null")
                schema_fail = True
                break

            # Categorical validation
            if req_col in schema["allowed_labels"]:
                allowed = schema["allowed_labels"][req_col]
                if str(val).strip() not in allowed:
                    log_rejection(idx, row_dict, "INVALID_LABEL_VALUE", f"Column '{req_col}'='{val}' not in {allowed}")
                    schema_fail = True
                    break

            # Binary validation
            if req_col in schema["binary_cols"]:
                try:
                    int_val = int(val)
                    if int_val not in (0, 1):
                        log_rejection(idx, row_dict, "INVALID_BINARY_VALUE", f"Column '{req_col}'={val} not in {{0, 1}}")
                        schema_fail = True
                        break
                except (ValueError, TypeError):
                    log_rejection(idx, row_dict, "INVALID_BINARY_TYPE", f"Column '{req_col}'={val} not integer 0/1")
                    schema_fail = True
                    break

        if schema_fail:
            continue

        # 4. In-batch Exact Duplicate & Contradictory Label Check
        if prompt_norm in seen_prompts_in_batch:
            prev_row = seen_prompts_in_batch[prompt_norm]
            labels_match = all(str(prev_row.get(c)) == str(row.get(c)) for c in schema["required_label_cols"])
            if not labels_match:
                log_rejection(idx, row_dict, "CONTRADICTORY_IN_BATCH", f"Same prompt exists with conflicting labels at row {prev_row['_row_index']}")
            else:
                log_rejection(idx, row_dict, "DUPLICATE_IN_BATCH", f"Exact duplicate of row {prev_row['_row_index']}")
            continue

        # 5. Exact Overlap with Frozen Test Set (STRICT LEAKAGE GATE)
        if prompt_norm in ref_test_exact_set:
            log_rejection(idx, row_dict, "TEST_SET_EXACT_LEAKAGE", "Matches held-out frozen test prompt verbatim")
            continue

        # 6. Exact Overlap with Existing Training Set (Redundant)
        if prompt_norm in ref_train_exact_set:
            log_rejection(idx, row_dict, "TRAIN_SET_EXACT_DUPLICATE", "Already exists in existing training split")
            continue

        # 7. Semantic Near-Duplicate / Similarity Leakage Audit via TF-IDF
        p_vec = leakage_vectorizer.transform([prompt_str])
        
        # Test leakage similarity
        test_sims = cosine_similarity(p_vec, ref_test_vecs)[0]
        max_test_sim = float(np.max(test_sims))
        if max_test_sim >= test_leakage_threshold:
            top_test_idx = int(np.argmax(test_sims))
            matched_test_prompt = str(ref_test_prompts.iloc[top_test_idx])[:100]
            log_rejection(
                idx, row_dict, "TEST_SET_SIMILARITY_LEAKAGE",
                f"Cosine similarity {max_test_sim:.3f} >= {test_leakage_threshold:.2f} with test prompt: '{matched_test_prompt}...'"
            )
            continue

        # Train redundancy similarity (flag or reject if near-verbatim)
        train_sims = cosine_similarity(p_vec, ref_train_vecs)[0]
        max_train_sim = float(np.max(train_sims))
        is_near_train_dup = max_train_sim >= train_similarity_threshold

        # Record accepted row
        clean_row = dict(row_dict)
        clean_row["_row_index"] = idx
        clean_row["_provenance_source"] = f"incoming_{csv_path.name}"
        clean_row["_provenance_validated_at"] = datetime.now(timezone.utc).isoformat()
        clean_row["_provenance_is_ai_generated"] = True
        clean_row["_max_test_similarity"] = round(max_test_sim, 4)
        clean_row["_max_train_similarity"] = round(max_train_sim, 4)
        
        if is_near_train_dup:
            clean_row["_audit_warning"] = f"HIGH_TRAIN_SIMILARITY_{max_train_sim:.3f}"
            flagged_records.append(clean_row)

        accepted_records.append(clean_row)
        seen_prompts_in_batch[prompt_norm] = {"_row_index": idx, **row_dict}

    accepted_df = pd.DataFrame(accepted_records)
    rejected_df = pd.DataFrame(rejected_records)

    accepted_count = len(accepted_records)
    rejected_count = len(rejected_records)
    flagged_count = len(flagged_records)

    print("\n" + "=" * 80)
    print("VALIDATION AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Incoming Rows:  {total_records:,}")
    print(f"Accepted Records:     {accepted_count:,} ({accepted_count/total_records*100:.1f}%)" if total_records else "0")
    print(f"Rejected Records:     {rejected_count:,} ({rejected_count/total_records*100:.1f}%)" if total_records else "0")
    print(f"Flagged Review:       {flagged_count:,}")
    print("\nRejection Breakdown by Reason:")
    for r_reason, count in sorted(rejection_reasons_counter.items(), key=lambda x: -x[1]):
        print(f"  • {r_reason:<32}: {count:,} ({count/total_records*100:.1f}%)")

    # Output file paths
    clean_out_dir = REPO_ROOT / "data" / "clean" / "validated_incoming"
    review_out_dir = REPO_ROOT / "data" / "review"
    reports_dir = REPO_ROOT / "data" / "reports"
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    review_out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    accepted_csv = clean_out_dir / f"{csv_path.stem}_accepted.csv"
    rejected_csv = review_out_dir / f"{csv_path.stem}_rejected.csv"
    report_json = reports_dir / f"validation_{csv_path.stem}.json"
    report_md = reports_dir / f"validation_{csv_path.stem}.md"

    # Save outputs
    if not accepted_df.empty:
        accepted_df.to_csv(accepted_csv, index=False)
        print(f"\nSaved Accepted Records: {accepted_csv.resolve()}")
    if not rejected_df.empty:
        rejected_df.to_csv(rejected_csv, index=False)
        print(f"Saved Rejected Records: {rejected_csv.resolve()}")

    # Generate JSON summary
    audit_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(csv_path),
        "input_sha256": file_sha256,
        "target_model": target_model,
        "target_model_name": model_name,
        "total_incoming_records": total_records,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "flagged_count": flagged_count,
        "rejection_breakdown": rejection_reasons_counter,
        "leakage_audit": {
            "test_set_exact_leaks": rejection_reasons_counter.get("TEST_SET_EXACT_LEAKAGE", 0),
            "test_set_similarity_leaks": rejection_reasons_counter.get("TEST_SET_SIMILARITY_LEAKAGE", 0),
            "test_leakage_threshold": test_leakage_threshold,
            "status": "ZERO_TEST_LEAKAGE_ENFORCED"
        },
        "artifacts_created": {
            "accepted_csv": str(accepted_csv) if not accepted_df.empty else None,
            "rejected_csv": str(rejected_csv) if not rejected_df.empty else None
        }
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved Validation JSON:  {report_json.resolve()}")

    # Generate Markdown Summary
    md_lines = [
        f"# Incoming Dataset Validation Report: `{csv_path.name}`",
        f"- **Audit Date**: {audit_summary['timestamp']}",
        f"- **Target Model**: {model_name} (`{target_model}`)",
        f"- **Source File Hash**: `{file_sha256[:16]}...`",
        "",
        "## Summary Metrics",
        f"- **Total Incoming Rows**: {total_records:,}",
        f"- **Accepted Clean Rows**: **{accepted_count:,}**",
        f"- **Rejected Rows**: **{rejected_count:,}**",
        f"- **Flagged Warnings**: **{flagged_count:,}**",
        f"- **Test Leakage Violations Prevented**: **{audit_summary['leakage_audit']['test_set_exact_leaks'] + audit_summary['leakage_audit']['test_set_similarity_leaks']}**",
        "",
        "## Rejection Reasons Table",
        "| Rejection Reason | Count | Explanation |",
        "| :--- | :---: | :--- |"
    ]
    for reason, count in rejection_reasons_counter.items():
        md_lines.append(f"| `{reason}` | {count:,} | Automated filter |")

    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Validation MD:    {report_md.resolve()}")
    print("=" * 80)

    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Validate incoming CSV datasets for Guardvane / PromptShield AI.")
    parser.add_argument("csv_path", nargs="?", help="Path to incoming CSV file.")
    parser.add_argument("--incoming-dir", default="data/incoming", help="Directory containing incoming CSV files.")
    parser.add_argument("--model", choices=list(MODEL_SCHEMAS.keys()), help="Target model key (auto-detected if omitted).")
    parser.add_argument("--test-leakage-threshold", type=float, default=0.70, help="TF-IDF cosine similarity threshold for test leakage rejection.")
    parser.add_argument("--train-similarity-threshold", type=float, default=0.85, help="Similarity threshold for train redundancy flagging.")
    args = parser.parse_args()

    incoming_dir = REPO_ROOT / args.incoming_dir

    if args.csv_path:
        csv_files = [Path(args.csv_path)]
    else:
        csv_files = list(incoming_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found to validate in '{incoming_dir.resolve()}'. Place new CSVs in data/incoming/.")
        sys.exit(0)

    for f in csv_files:
        validate_incoming_dataset(
            csv_path=f,
            target_model=args.model,
            test_leakage_threshold=args.test_leakage_threshold,
            train_similarity_threshold=args.train_similarity_threshold
        )


if __name__ == "__main__":
    main()


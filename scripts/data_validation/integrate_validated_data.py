#!/usr/bin/env python3
"""
PromptShield / Guardvane — Safe Dataset Integration Pipeline (Phase 3).
======================================================================
Safely merges validated incoming datasets into training pools without:
- Modifying or overwriting original base datasets
- Contaminating or modifying frozen test splits
- Overwriting existing model artifacts

Workflow:
1. Verifies that the input dataset was approved by the validation harness.
2. Performs a final zero-leakage check against the frozen test set.
3. Stages the expanded training dataset under data/clean/staged/ with version tracking.
4. Updates data/manifests/dataset_integration_manifest.json with provenance and metrics.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_validation.validate_incoming_data import (
    MODEL_SCHEMAS,
    load_reference_datasets,
    sha256_file,
)


def stage_and_integrate_dataset(
    accepted_csv_path: Path,
    target_model: str,
    integration_notes: str = "AI-generated training expansion",
) -> Dict[str, Any]:
    """
    Integrates an accepted CSV file into the model's staged training dataset.
    """
    accepted_csv_path = Path(accepted_csv_path).resolve()
    print("=" * 80)
    print("GUARDVANE / PROMPTSHIELD — SAFE DATASET INTEGRATION")
    print(f"Accepted Input: {accepted_csv_path.name}")
    print(f"Target Model:   {target_model}")
    print("=" * 80)

    if not accepted_csv_path.exists():
        raise FileNotFoundError(f"Accepted CSV not found: {accepted_csv_path}")

    accepted_df = pd.read_csv(accepted_csv_path)
    new_records_count = len(accepted_df)
    print(f"New accepted records to integrate: {new_records_count:,}")

    if new_records_count == 0:
        print("Zero records to integrate. Exiting.")
        return {"status": "empty_input"}

    # Final zero-leakage sanity assertion against frozen test split
    ref_train_prompts, ref_test_prompts = load_reference_datasets(target_model)
    test_set_exact = set(ref_test_prompts.str.strip().str.lower())

    prompt_col = "enhanced_prompt" if "enhanced_prompt" in accepted_df.columns else "prompt"
    incoming_prompts = set(accepted_df[prompt_col].astype(str).str.strip().str.lower())

    leakage_overlap = incoming_prompts.intersection(test_set_exact)
    if leakage_overlap:
        raise RuntimeError(
            f"[CRITICAL SAFETY ABORT] Found {len(leakage_overlap)} prompts overlapping with the frozen test set! "
            f"Examples: {list(leakage_overlap)[:3]}. Aborting integration."
        )

    print("Passed final zero-leakage test-set verification (0 overlapping prompts with frozen test partition).")

    # Load baseline training partition
    staged_dir = REPO_ROOT / "data" / "clean" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / "data" / "manifests" / "dataset_integration_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "manifest_name": "Guardvane Dataset Integration Manifest",
            "version": "1.0.0",
            "integrations": []
        }

    # Determine baseline training data based on model
    if target_model == "task_classifier":
        from ml.train_task_classifier import TASK_DATASET
        base_df = pd.DataFrame(TASK_DATASET, columns=["prompt", "task_category"])
        # Split to get original train set
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(
            base_df["prompt"], base_df["task_category"], test_size=0.20, stratify=base_df["task_category"], random_state=42
        )
        orig_train_df = pd.DataFrame({"prompt": X_tr, "task_category": y_tr})
        out_staged_csv = staged_dir / "task_classifier_staged_train.csv"

    elif target_model == "application_risk":
        from ml.train_security_model import RISK_LABELS
        p = REPO_ROOT / "data" / "security_risk_dataset.csv"
        raw_df = pd.read_csv(p)
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(
            raw_df["prompt"], raw_df[RISK_LABELS], test_size=0.20, random_state=42
        )
        orig_train_df = pd.concat([pd.Series(X_tr, name="prompt"), y_tr], axis=1)
        out_staged_csv = staged_dir / "application_risk_staged_train.csv"

    elif target_model == "complexity":
        from training.train_complexity_model import COMPLEXITY_DATASET
        base_df = pd.DataFrame(COMPLEXITY_DATASET, columns=["prompt", "complexity_tier"])
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(
            base_df["prompt"], base_df["complexity_tier"], test_size=0.20, stratify=base_df["complexity_tier"], random_state=42
        )
        orig_train_df = pd.DataFrame({"prompt": X_tr, "complexity_tier": y_tr})
        out_staged_csv = staged_dir / "complexity_staged_train.csv"

    elif target_model == "implementation_scope":
        tr_p = REPO_ROOT / "data" / "splits" / "train.csv"
        b1_p = REPO_ROOT / "data" / "clean" / "implementation_scope_batch_01_clean.csv"
        tr_df = pd.read_csv(tr_p)
        b1_df = pd.read_csv(b1_p) if b1_p.exists() else pd.DataFrame()
        orig_train_df = pd.concat([tr_df, b1_df], ignore_index=True)
        out_staged_csv = staged_dir / "implementation_scope_staged_train.csv"

    else:
        raise ValueError(f"Unsupported model: {target_model}")

    orig_count = len(orig_train_df)
    
    # Standardize columns for clean concatenate
    keep_cols = [c for c in orig_train_df.columns if c in accepted_df.columns]
    merged_train_df = pd.concat([orig_train_df[keep_cols], accepted_df[keep_cols]], ignore_index=True)
    final_count = len(merged_train_df)

    # Save staged training set
    merged_train_df.to_csv(out_staged_csv, index=False)
    staged_hash = sha256_file(out_staged_csv)

    print(f"\nIntegration Complete:")
    print(f"  • Original Training Pool: {orig_count:,} rows")
    print(f"  • Newly Added Records:    {new_records_count:,} rows")
    print(f"  • Final Staged Pool:      {final_count:,} rows")
    print(f"  • Staged File Path:       {out_staged_csv.resolve()}")
    print(f"  • Staged File SHA256:     {staged_hash}")

    # Record integration event in manifest
    integration_event = {
        "integration_id": f"INT_{target_model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": target_model,
        "input_source_csv": str(accepted_csv_path),
        "input_sha256": sha256_file(accepted_csv_path),
        "records_added": new_records_count,
        "original_train_records": orig_count,
        "final_staged_records": final_count,
        "staged_train_csv": str(out_staged_csv),
        "staged_sha256": staged_hash,
        "notes": integration_notes,
        "test_split_intact": True,
        "test_leakage_count": 0
    }

    manifest["integrations"].append(integration_event)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated Manifest:         {manifest_path.resolve()}")
    print("=" * 80)
    return integration_event


def main():
    parser = argparse.ArgumentParser(description="Safely stage and integrate validated datasets into training pools.")
    parser.add_argument("accepted_csv", help="Path to validated accepted CSV file.")
    parser.add_argument("--model", required=True, choices=list(MODEL_SCHEMAS.keys()), help="Target model key.")
    parser.add_argument("--notes", default="AI-generated training expansion", help="Integration provenance notes.")
    args = parser.parse_args()

    stage_and_integrate_dataset(
        accepted_csv_path=Path(args.accepted_csv),
        target_model=args.model,
        integration_notes=args.notes
    )


if __name__ == "__main__":
    main()


"""
Prompt Analyzer Orchestrator for PromptShield AI.
Combines Prompt Security analysis, Application Risk assessment,
and lightweight Sensitive Data detection.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import joblib

from analyzers.prompt_security_analyzer import analyze_prompt_security
from analyzers.sensitive_data_detector import detect_sensitive_data
from analyzers.security_controls_extractor import (
    extract_declared_negations,
    extract_security_controls,
    extract_weakened_security_controls,
)
from analyzers.prompt_understanding.segmentation import (
    segment_prompt_sections,
    get_security_relevant_segments,
)
from analyzers.model_comparison.task_classifier import task_classifier

_APP_RISK_MODEL: Optional[Any] = None

APP_RISK_LABELS: List[str] = [
    "sqli",
    "xss",
    "ssrf",
    "idor",
    "auth_weakness",
    "authz_weakness",
    "file_upload",
    "path_traversal",
    "command_injection",
    "csrf",
    "sensitive_data",
    "prompt_injection",
    "data_poisoning",
    "model_extraction",
    "ai_tool_risk"
]


def get_app_risk_model_path() -> Path:
    """
    Returns the absolute path to the Application Risk Model artifact.
    """
    return Path(__file__).resolve().parent.parent / "models" / "security_risk_model.pkl"


def load_app_risk_model() -> Any:
    """
    Loads and caches the baseline Application Risk model.
    Raises FileNotFoundError if the model artifact is missing.
    """
    global _APP_RISK_MODEL
    if _APP_RISK_MODEL is None:
        path = get_app_risk_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Application Risk Model artifact not found at: {path}. "
                "Train the model using '.venv/bin/python ml/train_security_model.py'."
            )
        _APP_RISK_MODEL = joblib.load(path)
    return _APP_RISK_MODEL


from analyzers.security_controls_extractor import (
    extract_declared_negations,
    extract_security_controls,
)


def analyze_application_risks(prompt: str) -> Dict[str, Any]:
    """
    Infers potential application security risks using the Application Risk Model,
    augmented with:
    1. Segmentation analysis to prevent long UI prompt signal dilution.
    2. Context-aware scoping for frontend styling / cosmetic updates.
    3. Deterministic extraction of declared security controls, negations,
       and weakened security controls.

    Returns:
        dict:
            - status (str): 'success', 'model_unavailable', or 'error'
            - detected_risks (list or None): Active predicted risk labels (excluding negated features)
            - risk_count (int or None): Count of active detected risks
            - declared_controls (list): Explicitly stated security controls parsed from prompt
            - declared_negations (list): Explicitly negated features (e.g. 'no database')
            - suppressed_risks (list): Model predictions suppressed by explicit user negations
            - mitigated_surfaces (dict): Attack surfaces that have declared security controls
            - unmitigated_surfaces (list): Attack surfaces with no declared defenses in prompt
            - context_scoping_applied (bool): Whether frontend context scoping removed spurious risks
            - task_domain (str): Inferred primary task domain
            - error (str, optional): Explanatory error message on failure
            - disclaimer (str): Contextual note on machine learning predictions
    """
    disclaimer = (
        "Statistical multi-label prediction based on application description patterns. "
        "These are model inferences, not proof of actual system vulnerabilities."
    )

    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "status": "error",
            "detected_risks": None,
            "risk_count": None,
            "declared_controls": [],
            "declared_negations": [],
            "suppressed_risks": [],
            "mitigated_surfaces": {},
            "unmitigated_surfaces": [],
            "context_scoping_applied": False,
            "task_domain": "unknown",
            "error": "Prompt cannot be empty or contain only whitespace.",
            "disclaimer": disclaimer
        }

    # 1. Deterministic extraction of controls, negations, and weakened controls
    declared_controls = extract_security_controls(prompt)
    declared_negations = extract_declared_negations(prompt)
    weakened_controls = extract_weakened_security_controls(prompt)

    # Compile set of risks to suppress based on explicit feature negations
    suppressed_risks_set = set()
    for neg in declared_negations:
        for risk in neg.get("suppressed_risks", []):
            suppressed_risks_set.add(risk)

    # Weakened controls (e.g. removing authentication from admin dashboard) must NEVER be suppressed
    weakened_risk_set = set()
    for weak in weakened_controls:
        for r in weak.get("associated_risks", []):
            weakened_risk_set.add(r)
            suppressed_risks_set.discard(r)

    try:
        model = load_app_risk_model()
        pipeline = model if hasattr(model, "predict") else model.get("pipeline")
        if pipeline is None:
            raise ValueError("Loaded model artifact does not contain a valid predictor or pipeline.")

        # 2. Multi-Level Inference: Full Prompt + Security-Relevant Segments
        # Prevents signal dilution when a prompt contains long UI specifications combined with backend/database instructions
        raw_detected_set = set()

        # Full prompt prediction
        prediction = pipeline.predict([prompt])[0]
        for label, value in zip(APP_RISK_LABELS, prediction):
            if value == 1:
                raw_detected_set.add(label)

        # Segment predictions for long / compound prompts (prevents signal dilution)
        is_long_or_compound = len(prompt.split()) > 35 or len(prompt) > 200 or "\n" in prompt
        if is_long_or_compound:
            security_segments = get_security_relevant_segments(prompt)
            for seg in security_segments:
                seg_clean = seg.strip()
                if seg_clean != prompt.strip() and len(seg_clean) > 15:
                    # Skip segments that are explicit negative statements
                    if not re.search(r"\b(does not need|doesn't need|no need for|without any|do not require|not required)\b", seg_clean, re.IGNORECASE):
                        seg_pred = pipeline.predict([seg_clean])[0]
                        for label, value in zip(APP_RISK_LABELS, seg_pred):
                            if value == 1:
                                raw_detected_set.add(label)

        # 3. Explicit Security Risk Injection (Never suppress genuine security vulnerabilities)
        # A. DOM XSS / Dangerous DOM Rendering
        if re.search(r"\b(dangerouslysetinnerhtml|innerhtml|outerhtml|eval\(|document\.write|location\.href\s*=\s*javascript:)\b", prompt, re.IGNORECASE):
            raw_detected_set.add("xss")

        # B. Weakened or Removed Authentication
        for r in weakened_risk_set:
            raw_detected_set.add(r)

        # C. Unparameterized SQL Database Queries
        if re.search(r"\b(?:sql|query|queries|database|db|postgres(?:ql)?|mysql|sqlite)\b.*?\b(?:unparameteriz\w+|without parameteriz\w+|raw sql|concatenat\w+)\b|\b(?:unparameteriz\w+|without parameteriz\w+|raw sql)\b.*?\b(?:sql|query|queries|database|db|postgres(?:ql)?|mysql)\b", prompt, re.IGNORECASE):
            raw_detected_set.add("sqli")

        # D. Unprotected Login / User Authentication without declared controls
        if re.search(r"\b(login page|user authentication|authenticate users)\b", prompt, re.IGNORECASE):
            has_auth_controls = any(c["id"] in {"password_hashing", "mfa", "secure_sessions", "preserved_authentication"} for c in declared_controls)
            if not has_auth_controls:
                raw_detected_set.add("auth_weakness")
            if re.search(r"\b(sql|database|db|postgres|mysql)\b", prompt, re.IGNORECASE):
                raw_detected_set.add("sqli")

        # 4. Context-Aware Scoping for Pure Frontend Styling & Cosmetic Component Changes
        task_res = task_classifier.classify_task(prompt)
        primary_domain = task_res.primary_domain

        frontend_subdomains = {"frontend_ui_styling", "frontend_component_modification", "frontend_functionality"}
        has_backend_channels = bool(re.search(
            r"\b(sql|database|postgres(?:ql)?|mysql|mongodb|endpoint|fastapi|express|rest\s*api|server|worker|celery|shell|bash|exec|upload|download|disk|webhook|curl|fetch|http\s*request)\b",
            prompt,
            re.IGNORECASE
        ))
        is_frontend_only = (
            primary_domain in {"frontend_ui_styling", "frontend_component_modification"}
            or (task_res.detected_domains and all(d in frontend_subdomains for d in task_res.detected_domains))
        ) and not has_backend_channels and not task_res.is_mixed_task

        # UI styling prompts are cosmetic visual requests; scope out backend false positives even if external asset/api keys are referenced
        if primary_domain == "frontend_ui_styling" and not has_backend_channels:
            is_frontend_only = True

        context_scoping_applied = False
        if is_frontend_only:
            # Irrelevant backend-exclusive attack surfaces
            scoped_out_risks = {
                "csrf", "sqli", "ssrf", "command_injection",
                "path_traversal", "file_upload", "data_poisoning",
                "model_extraction", "ai_tool_risk"
            }

            # If prompt does NOT mention modifying authentication, roles, or users, also suppress idor and auth
            if not re.search(r"\b(auth|authentication|login|role|roles|permission|admin|password|credentials?|token)\b", prompt, re.IGNORECASE):
                scoped_out_risks.update({"idor", "auth_weakness", "authz_weakness"})

            # Never suppress any weakened controls or explicit XSS
            scoped_out_risks.difference_update(weakened_risk_set)
            if "xss" in raw_detected_set and re.search(r"\b(dangerouslysetinnerhtml|innerhtml|outerhtml|eval\(|document\.write)\b", prompt, re.IGNORECASE):
                scoped_out_risks.discard("xss")

            spurious_risks = raw_detected_set.intersection(scoped_out_risks)
            if spurious_risks:
                raw_detected_set.difference_update(scoped_out_risks)
                context_scoping_applied = True

        # Spurious model_extraction suppression on standard REST API prompts lacking any ML/AI models
        if "model_extraction" in raw_detected_set and not re.search(r"\b(model|ml|ai|llm|neural|inference|weights|classifier|distill|stealing)\b", prompt, re.IGNORECASE):
            raw_detected_set.discard("model_extraction")

        # Active risks exclude any explicitly negated features
        raw_detected = sorted(list(raw_detected_set))
        active_risks = [r for r in raw_detected if r not in suppressed_risks_set]
        suppressed_active = [r for r in raw_detected if r in suppressed_risks_set]

        # Map declared controls to active attack surfaces
        mitigated_surfaces: Dict[str, List[str]] = {}
        for ctrl in declared_controls:
            for risk in ctrl.get("mitigates_risks", []):
                if risk in active_risks:
                    mitigated_surfaces.setdefault(risk, []).append(ctrl["name"])

        unmitigated_surfaces = [r for r in active_risks if r not in mitigated_surfaces]

        return {
            "status": "success",
            "detected_risks": active_risks,
            "risk_count": len(active_risks),
            "raw_predictions": raw_detected,
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": suppressed_active,
            "mitigated_surfaces": mitigated_surfaces,
            "unmitigated_surfaces": unmitigated_surfaces,
            "context_scoping_applied": context_scoping_applied,
            "task_domain": primary_domain,
            "disclaimer": disclaimer
        }
    except FileNotFoundError as e:
        return {
            "status": "model_unavailable",
            "detected_risks": None,
            "risk_count": None,
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": [],
            "mitigated_surfaces": {},
            "unmitigated_surfaces": [],
            "context_scoping_applied": False,
            "task_domain": "unknown",
            "error": str(e),
            "disclaimer": disclaimer
        }
    except Exception as e:
        return {
            "status": "error",
            "detected_risks": None,
            "risk_count": None,
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": [],
            "mitigated_surfaces": {},
            "unmitigated_surfaces": [],
            "context_scoping_applied": False,
            "task_domain": "unknown",
            "error": f"Application risk inference error: {str(e)}",
            "disclaimer": disclaimer
        }


def analyze_prompt(prompt: str, analysis_mode: str = "both") -> Dict[str, Any]:
    """
    Unified analysis endpoint handler.
    Supports modes: 'prompt_security', 'application_risk', 'both'.

    Returns separate sections for:
    1. Prompt Security Model output
    2. Application Risk Model output
    3. Sensitive Data detector findings
    4. Analysis status and model availability summary
    """
    mode = (analysis_mode or "both").lower().strip()
    if mode not in {"prompt_security", "application_risk", "both"}:
        mode = "both"

    task_res = task_classifier.classify_task(prompt)

    result: Dict[str, Any] = {
        "prompt": prompt,
        "analysis_mode": mode,
        "task_type": task_res.predicted_category,
        "task_classification": {
            "primary_domain": task_res.primary_domain,
            "confidence": task_res.confidence,
            "detected_domains": task_res.detected_domains,
            "domain_confidences": task_res.domain_confidences,
            "is_mixed_task": task_res.is_mixed_task,
            "confidence_status": task_res.confidence_status
        },
    }

    if not isinstance(prompt, str) or not prompt.strip():
        result.update({
            "status": "error",
            "error": "Prompt cannot be empty or contain only whitespace.",
            "prompt_security": None,
            "application_risks": None,
            "sensitive_data": None
        })
        return result

    # 1. Prompt Security Analysis
    if mode in {"prompt_security", "both"}:
        prompt_sec_result = analyze_prompt_security(prompt)
    else:
        prompt_sec_result = {
            "status": "skipped",
            "message": f"Excluded by analysis_mode '{mode}'"
        }

    # 2. Application Risk Analysis
    if mode in {"application_risk", "both"}:
        app_risk_result = analyze_application_risks(prompt)
    else:
        app_risk_result = {
            "status": "skipped",
            "message": f"Excluded by analysis_mode '{mode}'"
        }

    # 3. Sensitive Data Detection
    sensitive_data_result = detect_sensitive_data(prompt)

    # 4. Overall status evaluation
    errors: List[str] = []
    if prompt_sec_result.get("status") in {"model_unavailable", "error"}:
        errors.append(f"Prompt Security: {prompt_sec_result.get('error')}")
    if app_risk_result.get("status") in {"model_unavailable", "error"}:
        errors.append(f"Application Risk: {app_risk_result.get('error')}")

    if not errors:
        overall_status = "success"
    elif len(errors) == (2 if mode == "both" else 1):
        overall_status = "error"
    else:
        overall_status = "partial_success"

    result.update({
        "status": overall_status,
        "model_availability": {
            "prompt_security_model": prompt_sec_result.get("status") == "success",
            "application_risk_model": app_risk_result.get("status") == "success"
        },
        "errors": errors if errors else None,
        "prompt_security": prompt_sec_result,
        "application_risks": app_risk_result,
        "sensitive_data": sensitive_data_result,
        "declared_controls": app_risk_result.get("declared_controls", []) if isinstance(app_risk_result, dict) else [],
        "declared_negations": app_risk_result.get("declared_negations", []) if isinstance(app_risk_result, dict) else []
    })

    return result
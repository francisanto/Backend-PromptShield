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
    extract_attack_surfaces,
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
    attack_surfaces = extract_attack_surfaces(prompt, declared_negations, declared_controls)

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
        if re.search(r"\b(?:sql|query|queries|database|db|postgres(?:ql)?|mysql|sqlite|select\s+.*?\s+from|insert\s+into|update\s+.*?\s+set|delete\s+from)\b.*?\b(?:unparameteriz\w+|without parameteriz\w+|raw sql|concatenat\w+|string format\w*|f['\"])\b|\b(?:unparameteriz\w+|without parameteriz\w+|raw sql|concatenat\w+|string format\w*|f['\"])\b.*?\b(?:sql|query|queries|database|db|postgres(?:ql)?|mysql|select\s+.*?\s+from|insert\s+into)\b", prompt, re.IGNORECASE):
            raw_detected_set.add("sqli")

        # D. Unprotected Login / User Authentication without declared controls
        if re.search(r"\b(login(?:\s*(?:page|form|screen|view|component|endpoint))?|user authentication|authenticate users)\b", prompt, re.IGNORECASE):
            is_login_negated = any(n.get("feature") == "login" for n in declared_negations)
            is_db_negated = any(n.get("feature") == "database" for n in declared_negations)
            if not is_login_negated:
                has_auth_controls = any(c["id"] in {"password_hashing", "mfa", "secure_sessions", "preserved_authentication"} for c in declared_controls)
                if not has_auth_controls:
                    raw_detected_set.add("auth_weakness")
                if not is_db_negated and re.search(r"\b(sql|database|db|postgres|mysql)\b", prompt, re.IGNORECASE):
                    raw_detected_set.add("sqli")

        # E. Plaintext / Unhashed Password Storage
        if re.search(
            r"\b(?:plaintext|plain text|cleartext|unhashed|unencrypted)\s+(?:passwords?|credentials?|secrets?)\b|"
            r"\b(?:passwords?|credentials?|secrets?)\s+(?:stored\s+|saved\s+|kept\s+)?in\s+(?:plaintext|plain text|cleartext|unhashed|unencrypted)\b|"
            r"\b(?:store\w*|sav\w*|keep\w*)\s+(?:passwords?|credentials?|secrets?)\s+in\s+(?:plaintext|plain text|cleartext|unhashed|unencrypted)\b",
            prompt, re.IGNORECASE
        ):
            raw_detected_set.add("auth_weakness")
            raw_detected_set.add("sensitive_data")

        # F. Server-Side Request Forgery on User-Supplied URLs
        if re.search(
            r"\b(?:accepts?|takes?|user[- ]supplied|arbitrary|user[- ]provided|any)\b.*?\b(?:target\s+)?urls?\b.*?\b(?:fetch\w*|curl|proxy|request\w*|download\w*)\b|"
            r"\b(?:fetch\w*|curl|proxy|request\w*|download\w*)\s+.*?\b(?:arbitrary|user[- ]supplied|user[- ]provided|any)\b.*?\b(?:target\s+)?urls?\b|"
            r"\b(?:http\s+(?:post|get)\s+requests?\s+to\s+arbitrary\s+urls?)\b|"
            r"\b(?:without\s+(?:url\s+|domain\s+|destination\s+)?(?:validation|allowlist\w*|whitelist\w*|restriction\w*))\b",
            prompt, re.IGNORECASE
        ):
            raw_detected_set.add("ssrf")

        # G. Path Traversal on User-Supplied Filenames / Paths
        if re.search(
            r"\b(?:filename|filepath|path)\b.*?\bwithout\s+(?:path\s+|input\s+)?(?:validation|sanitiz\w+|check)\b|"
            r"\b(?:filename|filepath|path)\b.*?\b(?:unvalidated|unsanitized)\b|"
            r"\b(?:unvalidated|unsanitized)\s+.*?\b(?:filename|filepath|path)\b|"
            r"\b(?:directory traversal|\.\.\/|\.\.\\)\b",
            prompt, re.IGNORECASE
        ):
            raw_detected_set.add("path_traversal")

        # H. Command Execution / Subprocess with Shell
        if re.search(r"\b(?:subprocess|popen|system|exec|spawn)\b.*?\bshell\s*=\s*true\b|\b(?:run|execute)\s+user[- ](?:supplied|provided)\s+.*?\b(?:shell|bash|command)\b", prompt, re.IGNORECASE):
            raw_detected_set.add("command_injection")

        # I. Unsafe Token / Credential Exposure (Logging, Public Response, Hardcoding, Improper Disclosure)
        UNSAFE_TOKEN_EXPOSURE_REGEX = re.compile(
            r"\b(?:log|logging|print|echo)\s+.*?\b(?:tokens?|passwords?|credentials?|secrets?|api[_-]?keys?)\b.*?\b(?:logs?|console|stdout|stderr|file|debug)\b|"
            r"\b(?:return|expose|include|send|leak)\s+.*?\b(?:tokens?|passwords?|credentials?|secrets?|api[_-]?keys?)\b.*?\b(?:public|response|json response|api response|body|url|get request)\b|"
            r"\b(?:hardcode|hardcoded|default)\s+.*?\b(?:tokens?|passwords?|credentials?|secrets?|api[_-]?keys?)\b|"
            r"\b(?:view|display|show|copy|expose|disclose)\s+.*?\b(?:private|user(?:\x27s)?)\s+.*?\b(?:tokens?|passwords?|credentials?|secrets?)\b.*?\b(?:admin|other|dashboard|panel|console)\b",
            re.IGNORECASE
        )
        if UNSAFE_TOKEN_EXPOSURE_REGEX.search(prompt):
            raw_detected_set.add("sensitive_data")
            if re.search(r"\b(?:view|display|show|copy|disclose)\s+.*?\b(?:private|user(?:\x27s)?)\s+.*?\b(?:tokens?|passwords?|credentials?|secrets?)\b", prompt, re.IGNORECASE):
                raw_detected_set.add("authz_weakness")

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

        # Spurious SSRF suppression when prompt contains no outbound URL, webhook, or remote network request cues
        if "ssrf" in raw_detected_set:
            has_ssrf_cues = bool(re.search(
                r"\b(url|urls|http|https|webhook|webhooks|curl|proxy|remote\s+(?:server|url|api|host)|outbound|egress|download\s+from\s+url)\b",
                prompt,
                re.IGNORECASE
            ))
            if not has_ssrf_cues:
                raw_detected_set.discard("ssrf")

        # Spurious XSS suppression when prompt contains no HTML rendering, DOM manipulation, or user content display
        if "xss" in raw_detected_set:
            has_xss_cues = bool(re.search(
                r"\b(xss|cross[- ]site scripting|dangerouslysetinnerhtml|innerhtml|outerhtml|eval\(|document\.write|"
                r"render\w*\s+.*?(?:html|markdown|comment|message|post|content|user input|untrusted)|"
                r"(?:unescaped|sanitiz\w+|dompurify|escap\w+)\s+(?:html|input|content))\b",
                prompt,
                re.IGNORECASE
            ))
            if not has_xss_cues:
                raw_detected_set.discard("xss")

        # Spurious command_injection suppression when prompt contains no OS command, terminal, or shell execution cues
        if "command_injection" in raw_detected_set:
            has_cmd_cues = bool(re.search(
                r"\b(shell\s*=\s*true|bash|terminal|command\s+line|cli|exec(?:ve|v)?|system\s+command|subprocess|popen|spawn|os\.system|shell\s+command|execute\s+(?:a\s+)?shell|run\s+(?:a\s+)?shell)\b",
                prompt,
                re.IGNORECASE
            ))
            if not has_cmd_cues:
                raw_detected_set.discard("command_injection")

        # Scoping for pure data analysis tasks (e.g. pandas/dataframes/CSV processing)
        if primary_domain == "data_analysis":
            if not re.search(r"\b(sql|database|db|postgres|mysql|sqlite|query|table)\b", prompt, re.IGNORECASE):
                raw_detected_set.discard("sqli")
            if not re.search(r"\b(html|dom|render|endpoint|api|webhook)\b", prompt, re.IGNORECASE):
                raw_detected_set.difference_update({"xss", "csrf", "ssrf", "idor"})
            if not re.search(r"\b(train|model|weights|poison|adversarial)\b", prompt, re.IGNORECASE):
                raw_detected_set.discard("data_poisoning")

        # Prune spurious sensitive_data prediction on secure token handling lacking any unsafe exposure
        SECURE_TOKEN_HANDLING_REGEX = re.compile(
            r"\b(cryptographically secure tokens?|time[- ]limited(?:\s+cryptographically secure)?\s+tokens?|short[- ]lived tokens?|token expiration|hashed (?:reset )?tokens?|tokens? hashed using|secure (?:reset )?tokens?|stored securely)\b",
            re.IGNORECASE
        )
        if "sensitive_data" in raw_detected_set:
            is_secure_token = bool(SECURE_TOKEN_HANDLING_REGEX.search(prompt))
            has_unsafe_exposure = bool(UNSAFE_TOKEN_EXPOSURE_REGEX.search(prompt))
            if is_secure_token and not has_unsafe_exposure and not bool(re.search(r"\b(api[_-]?key|sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|bearer\s+[a-zA-Z0-9_.]{20,}|AIza[0-9A-Za-z-_]{30,})\b", prompt, re.IGNORECASE)):
                raw_detected_set.discard("sensitive_data")

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

        # Categorize risk evidence level: confirmed_vulnerability, missing_security_control, potential_attack_surface
        risk_evidence: Dict[str, Dict[str, str]] = {}
        for risk in active_risks:
            if risk == "xss" and re.search(r"\b(dangerouslysetinnerhtml|innerhtml|outerhtml|eval\(|document\.write)\b", prompt, re.IGNORECASE):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to render unescaped or dangerous DOM content."
                }
            elif risk == "sqli":
                has_untrusted_input = bool(re.search(
                    r"\b(?:user[- ](?:supplied|controlled|provided|input)|user\s+(?:comment|review|input|name|id)|http\s+request|request\s+(?:body|param(?:eter)?s?)|request_param|query[- ]string|query\s+param(?:eter)?s?)\b|"
                    r"\b(?:endpoint|route|api)\b.*?\b(?:f[\x27\"].*?\{|string format)\b|"
                    r"\b(?:unparameteriz\w+|without parameteriz\w+|raw sql|concatenat\w+)\b.*?\b(?:user|input|request|parameter)\b",
                    prompt, re.IGNORECASE
                ))
                has_dynamic_format = bool(re.search(
                    r"\b(?:unparameteriz\w+|without parameteriz\w+|raw sql|concatenat\w+|string format\w*|f['\"])\b",
                    prompt, re.IGNORECASE
                ))
                if has_untrusted_input and has_dynamic_format:
                    risk_evidence[risk] = {
                        "evidence_tier": "confirmed_vulnerability",
                        "description": "Explicit instruction for unparameterized or formatted SQL query directly incorporating untrusted user input."
                    }
                elif has_dynamic_format:
                    risk_evidence[risk] = {
                        "evidence_tier": "potential_attack_surface",
                        "description": "Dynamic SQL query formatting detected without confirmed untrusted input source; potential attack surface subject to input provenance."
                    }
                else:
                    risk_evidence[risk] = {
                        "evidence_tier": "potential_attack_surface",
                        "description": "Architectural SQL attack surface inferred from specification. Prompt provides insufficient information to confirm a vulnerability."
                    }
            elif risk in ("auth_weakness", "authz_weakness") and bool(weakened_controls):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to remove or weaken administrative authentication controls."
                }
            elif risk in ("auth_weakness", "sensitive_data") and (
                re.search(
                    r"\b(?:plaintext|plain text|cleartext|unhashed|unencrypted)\s+(?:passwords?|credentials?|secrets?)\b|"
                    r"\b(?:passwords?|credentials?|secrets?)\s+(?:stored\s+|saved\s+|kept\s+)?in\s+(?:plaintext|plain text|cleartext|unhashed|unencrypted)\b|"
                    r"\b(?:store\w*|sav\w*|keep\w*)\s+(?:passwords?|credentials?|secrets?)\s+in\s+(?:plaintext|plain text|cleartext|unhashed|unencrypted)\b",
                    prompt, re.IGNORECASE
                )
            ):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to store passwords in plaintext without hashing (CWE-256 / CWE-312)."
                }
            elif risk == "command_injection" and re.search(r"\b(shell\s*=\s*true|user[- ]supplied.*(?:shell|command)|execute.*command.*server)\b", prompt, re.IGNORECASE):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Server-side command execution using user-controlled parameters."
                }
            elif risk == "ssrf" and re.search(
                r"\b(?:accepts?|takes?|user[- ]supplied|arbitrary|user[- ]provided|any)\b.*?\b(?:target\s+)?urls?\b.*?\b(?:fetch\w*|curl|proxy|request\w*|download\w*)\b|"
                r"\b(?:fetch\w*|curl|proxy|request\w*|download\w*)\s+.*?\b(?:arbitrary|user[- ]supplied|user[- ]provided|any)\b.*?\b(?:target\s+)?urls?\b|"
                r"\b(?:http\s+(?:post|get)\s+requests?\s+to\s+arbitrary\s+urls?)\b|"
                r"\b(?:without\s+(?:url\s+|domain\s+|destination\s+)?(?:validation|allowlist\w*|whitelist\w*|restriction\w*))\b",
                prompt, re.IGNORECASE
            ):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to fetch arbitrary or user-supplied URLs without validation (SSRF)."
                }
            elif risk == "path_traversal" and re.search(
                r"\b(?:filename|filepath|path)\b.*?\bwithout\s+(?:path\s+|input\s+)?(?:validation|sanitiz\w+|check)\b|"
                r"\b(?:filename|filepath|path)\b.*?\b(?:unvalidated|unsanitized)\b|"
                r"\b(?:unvalidated|unsanitized)\s+.*?\b(?:filename|filepath|path)\b|"
                r"\b(?:directory traversal|\.\.\/|\.\.\\)\b",
                prompt, re.IGNORECASE
            ):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to access local filesystem paths using user input without path validation (Path Traversal)."
                }
            elif risk == "sensitive_data" and UNSAFE_TOKEN_EXPOSURE_REGEX.search(prompt):
                risk_evidence[risk] = {
                    "evidence_tier": "confirmed_vulnerability",
                    "description": "Explicit instruction to expose, log, hardcode, or disclose sensitive credentials/tokens."
                }
            elif risk == "auth_weakness" and re.search(r"\b(login(?:\s*(?:page|form|screen|view|component|endpoint|api))?|user authentication)\b", prompt, re.IGNORECASE) and not declared_controls:
                risk_evidence[risk] = {
                    "evidence_tier": "potential_attack_surface",
                    "description": "Authentication endpoint requested without declared security controls. Insufficient information to confirm a vulnerability; MFA and specific hashing algorithms are not universally mandatory."
                }
            else:
                risk_evidence[risk] = {
                    "evidence_tier": "potential_attack_surface",
                    "description": "Architectural attack surface inferred from functional specification. Prompt provides insufficient information to confirm a vulnerability."
                }

        # Deterministic evidence partitions (Priority 4)
        detected_risk_candidates = list(active_risks)
        confirmed_vulnerabilities = sorted([r for r, ev in risk_evidence.items() if ev.get("evidence_tier") == "confirmed_vulnerability"])
        potential_attack_surfaces = sorted([r for r, ev in risk_evidence.items() if ev.get("evidence_tier") == "potential_attack_surface"])
        missing_security_controls = sorted([r for r, ev in risk_evidence.items() if ev.get("evidence_tier") == "missing_security_control"])
        mitigated_risks = sorted(list(mitigated_surfaces.keys()))
        unmitigated_risks = [r for r in active_risks if r not in mitigated_surfaces]
        insufficient_information = sorted([
            r for r, ev in risk_evidence.items()
            if ev.get("evidence_tier") in ("insufficient_information", "potential_attack_surface")
            or "insufficient information" in ev.get("description", "").lower()
        ])
        not_identified_from_info = sorted([
            r for r in APP_RISK_LABELS
            if r not in confirmed_vulnerabilities and r not in potential_attack_surfaces and r not in missing_security_controls
        ])

        disclaimer = (
            "Statistical multi-label prediction and rule-based heuristic extraction based on application description patterns. "
            "Natural-language requirements analysis cannot guarantee runtime security or absence of flaws; implementation audits are required. "
            "Potential risks inferred from requirements indicate architectural attack surfaces, not confirmed code vulnerabilities."
        )

        return {
            "status": "success",
            "detected_risks": active_risks,
            "risk_count": len(active_risks),
            "raw_predictions": raw_detected,
            "detected_risk_candidates": detected_risk_candidates,
            "confirmed_vulnerabilities": confirmed_vulnerabilities,
            "potential_attack_surfaces": potential_attack_surfaces,
            "missing_security_controls": missing_security_controls,
            "mitigated_risks": mitigated_risks,
            "unmitigated_risks": unmitigated_risks,
            "insufficient_information": insufficient_information,
            "potential_risks_inferred": potential_attack_surfaces,
            "not_identified_from_info": not_identified_from_info,
            "attack_surfaces": attack_surfaces,
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": suppressed_active,
            "mitigated_surfaces": mitigated_surfaces,
            "unmitigated_surfaces": unmitigated_surfaces,
            "risk_evidence": risk_evidence,
            "context_scoping_applied": context_scoping_applied,
            "task_domain": primary_domain,
            "disclaimer": disclaimer
        }
    except FileNotFoundError as e:
        return {
            "status": "model_unavailable",
            "detected_risks": None,
            "risk_count": None,
            "detected_risk_candidates": [],
            "confirmed_vulnerabilities": [],
            "potential_attack_surfaces": [],
            "missing_security_controls": [],
            "mitigated_risks": [],
            "unmitigated_risks": [],
            "insufficient_information": [],
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": [],
            "mitigated_surfaces": {},
            "unmitigated_surfaces": [],
            "risk_evidence": {},
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
            "detected_risk_candidates": [],
            "confirmed_vulnerabilities": [],
            "potential_attack_surfaces": [],
            "missing_security_controls": [],
            "mitigated_risks": [],
            "unmitigated_risks": [],
            "insufficient_information": [],
            "declared_controls": declared_controls,
            "declared_negations": declared_negations,
            "suppressed_risks": [],
            "mitigated_surfaces": {},
            "unmitigated_surfaces": [],
            "risk_evidence": {},
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
        "attack_surfaces": app_risk_result.get("attack_surfaces", []) if isinstance(app_risk_result, dict) else [],
        "declared_controls": app_risk_result.get("declared_controls", []) if isinstance(app_risk_result, dict) else [],
        "declared_negations": app_risk_result.get("declared_negations", []) if isinstance(app_risk_result, dict) else []
    })

    return result
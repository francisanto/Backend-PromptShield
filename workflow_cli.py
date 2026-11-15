#!/usr/bin/env python3
"""
PromptShield AI — Unified Application Workflow CLI.
===================================================
A complete interactive terminal interface integrating:
1. Prompt Input (Multiline with clean termination sequence)
2. Development Scope Selection (Frontend, Backend, Full Stack with ML alignment reporting)
3. Complete PromptShield AI Analysis:
   - A. Task Domain Classification
   - B. Implementation Scope Analysis (Scope, 6 Architectural Requirements, Complexity, Design, Intent)
   - C. Prompt Security & Adversarial Injection Detection
   - D. Application Vulnerability Analysis (15 CWE surfaces, Evidence Tiers, Declared Controls)
   - E. Sensitive Data & Secrets Detection (with automatic masking)
   - F. Actionable Security Recommendations
   - G. Deterministic Prompt Optimization
   - H. Multi-Provider Token Estimation (Exact vs. Calibrated Estimates)
4. Optional Offline Foundation Model Comparison & Ranking
5. Consolidated Terminal Report with readable tables, severity indicators, and ANSI styling
"""

import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project Component Imports
from analyzers.prompt_analyzer import analyze_prompt, analyze_application_risks
from analyzers.prompt_security_analyzer import analyze_prompt_security
from analyzers.sensitive_data_detector import detect_sensitive_data, PATTERNS, redact_secret
from analyzers.model_comparison.task_classifier import task_classifier
from analyzers.model_comparison.engine import comparison_engine, ModelComparisonReport
from analyzers.model_comparison.registry import ProductCategory, registry
from analyzers.ide_comparison.engine import ide_comparison_engine
from analyzers.model_comparison.ranker import RankingPreset
from analyzers.model_comparison.tokenizer_adapters import get_tokenizer_adapter
from analyzers.prompt_optimizer import prompt_optimizer, OptimizationResult
from analyzers.implementation_scope_predictor import (
    scope_predictor,
    ScopePredictionResult,
    RequirementPrediction,
)
from cli import mask_sensitive_text, generate_recommendations, RISK_METADATA


class Colors:
    """ANSI color formatting with automatic TTY and NO_COLOR detection."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @property
    def RESET(self) -> str:
        return "\033[0m" if self.enabled else ""

    @property
    def BOLD(self) -> str:
        return "\033[1m" if self.enabled else ""

    @property
    def DIM(self) -> str:
        return "\033[2m" if self.enabled else ""

    @property
    def RED(self) -> str:
        return "\033[91m" if self.enabled else ""

    @property
    def GREEN(self) -> str:
        return "\033[92m" if self.enabled else ""

    @property
    def YELLOW(self) -> str:
        return "\033[93m" if self.enabled else ""

    @property
    def BLUE(self) -> str:
        return "\033[94m" if self.enabled else ""

    @property
    def CYAN(self) -> str:
        return "\033[96m" if self.enabled else ""

    @property
    def MAGENTA(self) -> str:
        return "\033[95m" if self.enabled else ""


def is_color_supported() -> bool:
    """Checks whether ANSI color output is appropriate for the current terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def get_cli_banner(colors: Colors) -> str:
    """Returns the main CLI startup banner."""
    lines = [
        f"{colors.BOLD}{colors.CYAN}================================================================================{colors.RESET}",
        f"{colors.BOLD}{colors.CYAN}        PROMPTSHIELD AI — UNIFIED WORKFLOW & ANALYSIS WORKSPACE        {colors.RESET}",
        f"{colors.DIM}   Local Security, Scope Prediction, Architecture Sizing & Model Evaluation     {colors.RESET}",
        f"{colors.BOLD}{colors.CYAN}================================================================================{colors.RESET}",
    ]
    return "\n".join(lines)


# ==============================================================================
# STEP 1: PROMPT INPUT
# ==============================================================================

def read_multiline_prompt(
    colors: Colors,
    prompt_label: str = "Enter the prompt you want PromptShield AI to analyze:",
    sentinel: str = "END"
) -> Optional[str]:
    """
    Accepts multiline prompt input from the user until the termination sequence is entered.
    Termination sequence: 'END' on a line by itself (case-insensitive) or Ctrl+D / EOF.
    Typing 'exit', 'quit', or 'cancel' on the first line returns None to cancel.
    """
    print(f"\n{colors.BOLD}{prompt_label}{colors.RESET}")
    print(f"• Paste or enter multiline text below.")
    print(f"• Type {colors.BOLD}'{sentinel}'{colors.RESET} on a new line when finished to proceed.")
    print(f"• Type {colors.BOLD}'exit'{colors.RESET} or {colors.BOLD}'quit'{colors.RESET} on the first line to return/exit.")
    print("-" * 80)

    lines: List[str] = []
    while True:
        try:
            line = input(f"{colors.CYAN}>{colors.RESET} ")
        except (KeyboardInterrupt, EOFError):
            print()
            return None

        # Check for immediate exit
        if not lines and line.strip().lower() in {"exit", "quit", "q", "cancel", ":q"}:
            return None

        # Check for termination sequence
        if line.strip().upper() == sentinel.upper():
            break

        lines.append(line)

    prompt_text = "\n".join(lines).strip()
    return prompt_text


# ==============================================================================
# STEP 2: SCOPE SELECTION & ALIGNMENT
# ==============================================================================

VALID_SCOPES = {
    "1": ("frontend", "Frontend"),
    "frontend": ("frontend", "Frontend"),
    "2": ("backend", "Backend"),
    "backend": ("backend", "Backend"),
    "3": ("full_stack", "Full Stack"),
    "full stack": ("full_stack", "Full Stack"),
    "fullstack": ("full_stack", "Full Stack"),
    "full-stack": ("full_stack", "Full Stack"),
}


def display_scope_check_results(
    colors: Colors,
    scope_pred: ScopePredictionResult,
    task_category: str
) -> None:
    """
    Displays the ML scope and architectural requirements check results immediately
    after prompt entry.
    """
    print(f"\n{colors.BOLD}================================================================================{colors.RESET}")
    print(f"{colors.BOLD}STEP 2 — MODEL SCOPE CHECK & ARCHITECTURAL CLASSIFICATION{colors.RESET}")
    print(f"{colors.BOLD}================================================================================{colors.RESET}")
    if scope_pred.status == "success":
        pred_disp = scope_pred.primary_scope.replace("_", " ").title()
        conf_pct = f"{scope_pred.primary_scope_confidence * 100:.1f}%"
        print(f"Model Scope Detection Results:")
        print(f"  • {colors.BOLD}Primary Scope:{colors.RESET}          {colors.CYAN}{pred_disp}{colors.RESET} ({conf_pct} confidence)")

        reqs = scope_pred.requirements
        print(f"  • {colors.BOLD}Architectural Flags:{colors.RESET}")
        for r_key, r_label in [
            ("frontend_required", "Frontend Required"),
            ("backend_required", "Backend Required"),
            ("database_required", "Database Required"),
            ("api_required", "API Required"),
            ("authentication_required", "Authentication Required"),
            ("external_integration_required", "External Integration"),
        ]:
            if r_key in reqs:
                req = reqs[r_key]
                flag_str = f"{colors.GREEN}Yes{colors.RESET}" if req.predicted_flag == 1 else f"{colors.DIM}No{colors.RESET}"
                prob_str = f"({req.probability * 100:.1f}%)"
                print(f"      - {r_label:<28} {flag_str} {prob_str}")

        comp_disp = scope_pred.complexity.capitalize()
        comp_conf = f"{scope_pred.complexity_confidence * 100:.1f}%"
        design_disp = scope_pred.design_emphasis.capitalize()
        design_conf = f"{scope_pred.design_emphasis_confidence * 100:.1f}%"
        intent_disp = scope_pred.task_intent.capitalize()
        intent_conf = f"{scope_pred.task_intent_confidence * 100:.1f}%"

        print(f"  • {colors.BOLD}Complexity:{colors.RESET}             {comp_disp} ({comp_conf})")
        print(f"  • {colors.BOLD}Design Emphasis:{colors.RESET}        {design_disp} ({design_conf})")
        print(f"  • {colors.BOLD}Task Intent:{colors.RESET}            {intent_disp} ({intent_conf})")
    else:
        print(f"  {colors.YELLOW}Scope Predictor model unavailable ({scope_pred.error or 'offline'}).{colors.RESET}")

    cat_str = getattr(task_category, "predicted_category", str(task_category))
    cat_disp = cat_str.replace("_", " ").title()
    print(f"  • {colors.BOLD}Inferred Domain:{colors.RESET}        {cat_disp}")
    print("-" * 80)


def select_development_scope(
    colors: Colors,
    detected_scope: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """
    Prompts the user to select one of exactly three development scopes:
    1. Frontend
    2. Backend
    3. Full Stack

    If detected_scope is provided, pressing Enter defaults to the detected scope.
    Returns a tuple of (scope_id, display_name), or None if canceled.
    """
    print("Confirm or select the intended development scope for this prompt:")
    print(f"  {colors.BOLD}1.{colors.RESET} Frontend")
    print(f"  {colors.BOLD}2.{colors.RESET} Backend")
    print(f"  {colors.BOLD}3.{colors.RESET} Full Stack")
    print("-" * 80)

    # Determine default choice from detected scope
    default_id = "full_stack"
    default_disp = "Full Stack"
    if detected_scope:
        norm = detected_scope.lower()
        if norm in {"frontend", "design_only"}:
            default_id = "frontend"
            default_disp = "Frontend"
        elif norm in {"backend", "security_only"}:
            default_id = "backend"
            default_disp = "Backend"
        elif norm in {"full_stack", "unclear"}:
            default_id = "full_stack"
            default_disp = "Full Stack"

    prompt_label = (
        f"{colors.CYAN}Enter choice [1-3, or press Enter to confirm '{default_disp}']:{colors.RESET} "
        if detected_scope else
        f"{colors.CYAN}Enter choice [1-3]:{colors.RESET} "
    )

    while True:
        try:
            choice = input(prompt_label).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return None

        if choice in {"exit", "quit", "q", "cancel"}:
            return None

        if not choice and detected_scope:
            print(f"{colors.GREEN}✓ Confirmed Scope: {colors.BOLD}{default_disp}{colors.RESET}")
            return default_id, default_disp

        if choice in VALID_SCOPES:
            scope_id, display_name = VALID_SCOPES[choice]
            print(f"{colors.GREEN}✓ Selected Scope: {colors.BOLD}{display_name}{colors.RESET}")
            return scope_id, display_name

        print(f"{colors.RED}Invalid selection '{choice}'. Please enter 1, 2, or 3.{colors.RESET}")


def evaluate_scope_alignment(
    selected_scope_id: str,
    selected_scope_display: str,
    scope_pred: ScopePredictionResult
) -> Dict[str, Any]:
    """
    Compares the user's selected scope with the ML-predicted scope.
    Detects and reports disagreements without overriding user selection.
    """
    if scope_pred.status != "success":
        return {
            "agrees": None,
            "selected_scope": selected_scope_display,
            "predicted_scope": "Unavailable",
            "confidence": 0.0,
            "status_text": f"Scope Predictor unavailable ({scope_pred.error or 'offline'}).",
            "is_nuance": False
        }

    pred_scope = scope_pred.primary_scope
    pred_conf = scope_pred.primary_scope_confidence

    # Map user selection to compare with ML primary_scope classes
    # classes: ['backend', 'design_only', 'frontend', 'full_stack', 'security_only', 'unclear']
    normalized_selected = selected_scope_id

    if normalized_selected == pred_scope:
        agrees = True
        status_text = (
            f"Scope Agreement: User selection '{selected_scope_display}' aligns with "
            f"ML primary scope prediction ('{pred_scope}', {pred_conf*100:.1f}% confidence)."
        )
        is_nuance = False
    elif pred_scope in {"design_only", "security_only", "unclear"}:
        agrees = False
        status_text = (
            f"Scope Nuance: User selected '{selected_scope_display}', but ML predicted specialized "
            f"scope '{pred_scope}' ({pred_conf*100:.1f}% confidence). "
            f"User selection '{selected_scope_display}' is preserved as contextual intent."
        )
        is_nuance = True
    else:
        agrees = False
        pred_display = pred_scope.replace("_", " ").title()
        status_text = (
            f"Scope Disagreement: User selected '{selected_scope_display}', but ML predicted "
            f"'{pred_display}' ({pred_conf*100:.1f}% confidence). "
            f"User selection will be preserved as guiding context; predictor findings are reported alongside."
        )
        is_nuance = False

    return {
        "agrees": agrees,
        "selected_scope": selected_scope_display,
        "predicted_scope": pred_scope,
        "confidence": pred_conf,
        "status_text": status_text,
        "is_nuance": is_nuance
    }


# ==============================================================================
# STEP 3: RUN COMPLETE PROMPTSHIELD ANALYSIS
# ==============================================================================

def estimate_tokens_comparison(orig_text: str, opt_text: str) -> Dict[str, Any]:
    """
    Calculates token counts for original and optimized prompts across
    major provider tokenizer adapters.
    Strictly distinguishes exact measurements (tiktoken) from estimates (BPE/SentencePiece).
    """
    targets = [
        ("OpenAI GPT-4o (o200k_base)", "gpt-4o"),
        ("Anthropic Claude 3.5 (BPE)", "claude-3-5-sonnet-20241022"),
        ("Google Gemini 1.5 (SentencePiece)", "gemini-1.5-pro"),
    ]

    results = []
    for label, mid in targets:
        try:
            adapter = get_tokenizer_adapter(mid)
            orig_res = adapter.count_tokens(orig_text)
            opt_res = adapter.count_tokens(opt_text)
            delta = opt_res.token_count - orig_res.token_count
            pct = (delta / orig_res.token_count * 100.0) if orig_res.token_count > 0 else 0.0

            method_label = "Exact (tiktoken)" if orig_res.is_exact else "Calibrated Estimate"
            results.append({
                "label": label,
                "model_id": mid,
                "orig_tokens": orig_res.token_count,
                "opt_tokens": opt_res.token_count,
                "delta": delta,
                "pct_change": round(pct, 1),
                "method": method_label,
                "is_exact": orig_res.is_exact,
                "disclaimer": orig_res.token_count_disclaimer
            })
        except Exception as e:
            results.append({
                "label": label,
                "model_id": mid,
                "orig_tokens": max(0, len(orig_text) // 4),
                "opt_tokens": max(0, len(opt_text) // 4),
                "delta": 0,
                "pct_change": 0.0,
                "method": "Fallback Heuristic",
                "is_exact": False,
                "error": str(e),
                "disclaimer": "Character count heuristic; tokenizer adapter unavailable."
            })

    return {
        "status": "success",
        "comparisons": results
    }


def run_complete_promptshield_analysis(
    prompt: str,
    selected_scope_id: str,
    selected_scope_display: str,
    precomputed_scope_pred: Optional[ScopePredictionResult] = None
) -> Dict[str, Any]:
    """
    Executes the complete PromptShield AI analysis pipeline:
    A. Task Classification
    B. Implementation Scope Prediction (v1)
    C. Prompt Security / Injection Detection
    D. Application Vulnerability Analysis
    E. Sensitive Data Detection
    F. Actionable Security Recommendations
    G. Deterministic Prompt Optimization
    H. Multi-Provider Token Estimation
    """
    # 1. Implementation Scope Predictor
    if precomputed_scope_pred is not None:
        scope_pred = precomputed_scope_pred
    else:
        scope_pred = scope_predictor.predict(prompt)

    # 2. Scope Alignment Check
    alignment = evaluate_scope_alignment(selected_scope_id, selected_scope_display, scope_pred)

    # 3. Prompt Security & Application Vulnerability Analysis
    core_analysis = analyze_prompt(prompt, analysis_mode="both")

    # 4. Sensitive Data Detection
    sensitive_data = core_analysis.get("sensitive_data") or detect_sensitive_data(prompt)

    # 5. Security Recommendations
    recommendations = generate_recommendations(core_analysis)

    # 6. Prompt Optimization (using selected scope context)
    scope_category_map = {
        "frontend": "frontend_ui",
        "backend": "backend_api",
        "full_stack": "full_stack",
    }
    opt_task_cat = scope_category_map.get(selected_scope_id, "general_instruction")

    try:
        opt_res = prompt_optimizer.optimize_prompt(
            prompt=prompt,
            security_analysis=core_analysis,
            task_category=opt_task_cat
        )
        opt_status = "success"
    except Exception as e:
        opt_res = OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=prompt,
            task_category=opt_task_cat,
            clarifications_or_assumptions=[f"Optimizer error: {str(e)}"],
            applied_hardening_rules=[],
            changes_summary=["Optimization unavailable due to processing exception."],
            is_substantially_changed=False,
            detected_risks_addressed=[]
        )
        opt_status = "error"

    # 7. Token Estimation
    token_est = estimate_tokens_comparison(prompt, opt_res.optimized_prompt)

    return {
        "prompt": prompt,
        "masked_prompt": mask_sensitive_text(prompt),
        "selected_scope_id": selected_scope_id,
        "selected_scope_display": selected_scope_display,
        "scope_prediction": scope_pred,
        "scope_alignment": alignment,
        "core_analysis": core_analysis,
        "sensitive_data": sensitive_data,
        "recommendations": recommendations,
        "optimization": opt_res,
        "optimization_status": opt_status,
        "token_estimation": token_est
    }


# ==============================================================================
# STEP 3: OPTIONAL COMPARISON (IDE / FOUNDATION MODELS / BOTH / CONTINUE)
# ==============================================================================

def prompt_comparison_options(colors: Colors) -> str:
    """
    Displays optional comparison modes:
    1. Continue (Standard Security Scanning & Prompt Optimization)
    2. IDE Comparison (Compare AI Coding Environments: Antigravity, Cursor, etc.)
    3. Model Comparison (Compare Foundation LLMs: Claude, GPT, Gemini, etc.)
    4. Both (IDE & Model Comparison)
    """
    print(f"\n{colors.BOLD}================================================================================{colors.RESET}")
    print(f"{colors.BOLD}STEP 3 — OPTIONAL COMPARISON{colors.RESET}")
    print(f"{colors.BOLD}================================================================================{colors.RESET}")
    print("Choose an optional comparison mode or continue to security scanning & optimization:")
    print(f"  {colors.BOLD}1.{colors.RESET} Continue (Standard Security Scanning & Prompt Optimization)")
    print(f"  {colors.BOLD}2.{colors.RESET} IDE Comparison (Compare AI Coding Environments: Antigravity, Cursor, etc.)")
    print(f"  {colors.BOLD}3.{colors.RESET} Model Comparison (Compare Foundation LLMs: Claude, GPT, Gemini, etc.)")
    print(f"  {colors.BOLD}4.{colors.RESET} Both (IDE & Model Comparison)")
    print("-" * 80)

    while True:
        try:
            choice = input(f"{colors.CYAN}Enter choice [1-4, or press Enter to Continue]:{colors.RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return "cancel"

        if choice in {"exit", "quit", "q", "cancel"}:
            return "cancel"

        if choice in {"", "1", "continue", "c", "skip", "no", "n"}:
            return "continue"
        elif choice in {"2", "ide", "ides"}:
            return "ide"
        elif choice in {"3", "model", "models"}:
            return "model"
        elif choice in {"4", "both", "all", "b"}:
            return "both"

        print(f"{colors.RED}Invalid choice '{choice}'. Please enter 1, 2, 3, or 4.{colors.RESET}")


def list_available_ide_assistants(colors: Colors) -> List[str]:
    """Lists eligible IDE coding assistants from the registry."""
    ides = registry.list_ides()
    print(f"\n{colors.BOLD}Eligible IDE Coding Assistants ({len(ides)}):{colors.RESET}")
    for idx, ide in enumerate(ides, 1):
        pricing = getattr(ide, "pricing_tier_summary", getattr(ide, "pricing_summary", ""))
        pricing_str = f" — {pricing}" if pricing else ""
        print(f"  {colors.CYAN}{idx}.{colors.RESET} {colors.BOLD}{ide.display_name}{colors.RESET} ({ide.vendor}){pricing_str}")
    return [ide.display_name for ide in ides]


def list_available_foundation_models(colors: Colors) -> List[str]:
    """Lists eligible Foundation Models from the registry."""
    models = registry.list_models()
    print(f"\n{colors.BOLD}Eligible Foundation Models ({len(models)}):{colors.RESET}")
    for idx, m in enumerate(models, 1):
        print(f"  {colors.CYAN}{idx:2d}.{colors.RESET} {colors.BOLD}{m.display_name}{colors.RESET} ({m.provider})")
    return [m.display_name for m in models]


def prompt_model_comparison(colors: Colors) -> bool:
    """
    Backward-compatible helper function for legacy tests.
    Asks the user whether they would like to compare AI models for this prompt.
    Options:
    1. Yes
    2. No
    """
    print(f"\n{colors.BOLD}================================================================================{colors.RESET}")
    print(f"{colors.BOLD}STEP 4 — OPTIONAL MODEL COMPARISON{colors.RESET}")
    print(f"{colors.BOLD}================================================================================{colors.RESET}")
    print("Would you like to compare AI models for this prompt?")
    print(f"  {colors.BOLD}1.{colors.RESET} Yes")
    print(f"  {colors.BOLD}2.{colors.RESET} No")
    print("-" * 80)

    while True:
        try:
            choice = input(f"{colors.CYAN}Enter choice [1-2]:{colors.RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return False

        if choice in {"1", "yes", "y"}:
            return True
        elif choice in {"2", "no", "n"}:
            return False

        print(f"{colors.RED}Invalid choice '{choice}'. Enter 1 for Yes or 2 for No.{colors.RESET}")


def run_model_comparison(prompt: str) -> Optional[ModelComparisonReport]:
    """
    Executes local offline model suitability scoring, costing, and ranking.
    """
    try:
        report = comparison_engine.compare_prompt(
            prompt=prompt,
            category=ProductCategory.FOUNDATION_MODEL,
            preset=RankingPreset.BALANCED
        )
        return report
    except Exception as e:
        print(f"Model comparison unavailable: {str(e)}")
        return None


# ==============================================================================
# STEP 5: CONSOLIDATED REPORT FORMATTING
# ==============================================================================

def _format_ide_comparison_section(
    lines: List[str],
    ide_report: ModelComparisonReport,
    colors: Colors,
    subdiv: str,
    prefix: str = "13"
) -> None:
    lines.append(f"\n{colors.BOLD}[{prefix}] IDE CODING ASSISTANT COMPARISON & RANKING{colors.RESET}")
    lines.append(subdiv)
    ranked = getattr(ide_report, "ranked_products", [])
    if ranked:
        lines.append(f"  {colors.BOLD}Ranked IDE Assistant Recommendations:{colors.RESET}\n")
        lines.append(f"  {'Rank':<6} {'IDE Assistant':<24} {'Vendor':<22} {'Score':<10} {'Pricing Summary'}")
        lines.append("  " + "-" * 76)
        for p in ranked:
            p_score = f"{p.score*100:.1f}%"
            lines.append(f"  #{p.rank:<5} {p.display_name:<24} {p.vendor:<22} {p_score:<10} {p.pricing_summary}")

        lines.append(f"\n  {colors.BOLD}IDE Assistant Evaluation Rationale:{colors.RESET}")
        for p in ranked:
            strengths = ", ".join(p.key_strengths[:3]) if hasattr(p, "key_strengths") and p.key_strengths else ""
            lines.append(f"  • {colors.BOLD}#{p.rank} {p.display_name}{colors.RESET} ({p.vendor}):")
            lines.append(f"    - Overall Score: {p.score*100:.1f}%")
            if strengths:
                lines.append(f"    - Key Strengths: {strengths}")
            lines.append(f"    - Rationale: {p.reasoning}")
    else:
        lines.append(f"  {colors.YELLOW}No eligible IDE coding assistants evaluated.{colors.RESET}")


def _format_model_comparison_section(
    lines: List[str],
    comparison_report: ModelComparisonReport,
    colors: Colors,
    subdiv: str,
    prefix: str = "13"
) -> None:
    lines.append(f"\n{colors.BOLD}[{prefix}] OFFLINE FOUNDATION MODEL COMPARISON & RANKING{colors.RESET}")
    lines.append(subdiv)
    ranked = getattr(comparison_report, "ranked_models", [])
    if ranked:
        lines.append(f"  {colors.BOLD}Ranked Model Recommendations (Balanced TOPSIS / WSM Preset):{colors.RESET}\n")
        lines.append(f"  {'Rank':<6} {'Model':<24} {'Provider':<14} {'Suitability':<14} {'Est. Cost':<14} {'Token Count':<12}")
        lines.append("  " + "-" * 76)

        top_models = ranked[:5]
        for rm in top_models:
            suit_pct = f"{rm.suitability_score*100:.1f}%"
            cost_str = f"${rm.estimated_cost_usd:.5f}" if rm.estimated_cost_usd is not None else "N/A"
            tok_str = f"{rm.token_count}"
            lines.append(f"  #{rm.rank:<5} {rm.display_name:<24} {rm.provider:<14} {suit_pct:<14} {cost_str:<14} {tok_str:<12}")

        lines.append(f"\n  {colors.BOLD}Top Model Evaluation Rationale:{colors.RESET}")
        for rm in ranked[:3]:
            details = comparison_report.model_details.get(rm.model_id) if hasattr(comparison_report, "model_details") else None
            caps = ", ".join(details.capabilities[:3]) if details and hasattr(details, "capabilities") else ""
            lines.append(f"  • {colors.BOLD}#{rm.rank} {rm.display_name}{colors.RESET} ({rm.provider}):")
            lines.append(f"    - Suitability Score: {rm.suitability_score*100:.1f}%")
            lines.append(f"    - Rationale: {rm.reasoning}")
            if caps:
                lines.append(f"    - Key Capabilities: {caps}")
    else:
        lines.append(f"  {colors.YELLOW}No eligible foundation models evaluated.{colors.RESET}")


def format_consolidated_report(
    analysis_data: Dict[str, Any],
    comparison_report: Optional[ModelComparisonReport] = None,
    colors: Optional[Colors] = None,
    ide_comparison_report: Optional[ModelComparisonReport] = None
) -> str:
    """
    Constructs the organized, human-readable terminal report containing:
    1. Original prompt (masked)
    2. User-selected scope
    3. Predicted implementation scope & alignment
    4. Task classification
    5. Architectural requirements (6 binary flags)
    6. Complexity, design emphasis, and task intent
    7. Prompt injection / security analysis
    8. Application vulnerability findings (15 categories & evidence tiers)
    9. Sensitive data findings
    10. Security recommendations
    11. Original and optimized prompts
    12. Token estimates
    13. IDE and/or Model comparison (if requested)
    """
    if colors is None:
        colors = Colors(enabled=False)
    lines: List[str] = []
    w = 80
    div = "=" * w
    subdiv = "-" * w

    lines.append("\n" + div)
    lines.append(f"{colors.BOLD}{colors.CYAN}              PROMPTSHIELD AI — CONSOLIDATED WORKFLOW REPORT{colors.RESET}")
    lines.append(f"{colors.DIM}  Note: All classifications, risk inferences, and scope predictions are statistical{colors.RESET}")
    lines.append(f"{colors.DIM}  model estimates and heuristic indicators, not verified facts or security guarantees.{colors.RESET}")
    lines.append(div)

    # 1. Original Prompt
    lines.append(f"\n{colors.BOLD}[1] ORIGINAL PROMPT{colors.RESET}")
    lines.append(subdiv)
    masked_prompt = analysis_data["masked_prompt"]
    # Indent prompt text
    for p_line in masked_prompt.splitlines():
        lines.append(f"  {colors.DIM}{p_line}{colors.RESET}")

    # 2 & 3. Development Scope & Predicted Implementation Scope
    lines.append(f"\n{colors.BOLD}[2 & 3] DEVELOPMENT SCOPE & PREDICTED IMPLEMENTATION SCOPE{colors.RESET}")
    lines.append(subdiv)
    selected_disp = analysis_data["selected_scope_display"]
    align = analysis_data["scope_alignment"]
    scope_pred: ScopePredictionResult = analysis_data["scope_prediction"]

    lines.append(f"  • User-Selected Scope:    {colors.BOLD}{colors.GREEN}{selected_disp}{colors.RESET}")

    if scope_pred.status == "success":
        pred_disp = scope_pred.primary_scope.replace("_", " ").title()
        lines.append(
            f"  • ML-Predicted Scope:     {colors.BOLD}{colors.CYAN}{pred_disp}{colors.RESET} "
            f"(Confidence: {scope_pred.primary_scope_confidence*100:.1f}%)"
        )
        if align.get("agrees"):
            lines.append(f"  • Status:                 {colors.GREEN}✓ {align['status_text']}{colors.RESET}")
        else:
            lines.append(f"  • Status:                 {colors.YELLOW}⚠️  {align['status_text']}{colors.RESET}")

        # Primary Scope Probabilities
        proba_parts = [
            f"{k.replace('_', ' ')}: {v*100:.1f}%"
            for k, v in sorted(scope_pred.primary_scope_probabilities.items(), key=lambda x: -x[1])
        ]
        lines.append(f"  • Scope Distribution:     {colors.DIM}{', '.join(proba_parts)}{colors.RESET}")
    else:
        lines.append(f"  • ML-Predicted Scope:     {colors.RED}{align['status_text']}{colors.RESET}")

    # 4. Task Classification
    lines.append(f"\n{colors.BOLD}[4] TASK CLASSIFICATION{colors.RESET}")
    lines.append(subdiv)
    core = analysis_data["core_analysis"]
    task_class = core.get("task_classification", {})
    primary_domain = task_class.get("primary_domain", "unknown")
    task_conf = task_class.get("confidence", 0.0)
    conf_status = task_class.get("confidence_status", "UNKNOWN")

    marginal_flag = f" {colors.YELLOW}[MARGINAL CONFIDENCE - MODEL ESTIMATE]{colors.RESET}" if task_conf < 0.60 else ""
    lines.append(
        f"  • Primary Domain:         {colors.BOLD}{primary_domain.replace('_', ' ').title()}{colors.RESET} "
        f"(Confidence: {task_conf*100:.1f}% [{conf_status}]){marginal_flag}"
    )
    detected_domains = task_class.get("detected_domains", [])
    if detected_domains:
        lines.append(f"  • Detected Domains:       {', '.join([d.replace('_', ' ') for d in detected_domains])}")
    lines.append(f"  {colors.DIM}* Note: Task classification is a statistical prediction to guide model selection.{colors.RESET}")

    # 5. Architectural Requirements (6 Binary Flags)
    lines.append(f"\n{colors.BOLD}[5] ARCHITECTURAL REQUIREMENTS (IMPLEMENTATION SCOPE){colors.RESET}")
    lines.append(subdiv)
    if scope_pred.status == "success" and scope_pred.requirements:
        lines.append(f"  {'Requirement Flag':<32} {'Prediction':<16} {'Probability':<14} {'Threshold':<12} {'Margin':<14}")
        lines.append("  " + "-" * 88)
        for req_key, req in scope_pred.requirements.items():
            flag_text = f"{colors.GREEN}REQUIRED [1]{colors.RESET}" if req.is_required else f"{colors.DIM}NO [0]{colors.RESET}"
            p_text = f"{req.probability*100:.1f}%"
            th_text = f"{req.threshold*100:.1f}%"
            margin_sign = "+" if req.margin >= 0 else ""
            marginal_note = f" {colors.YELLOW}⚠️ Marginal{colors.RESET}" if abs(req.margin) < 0.10 else ""
            m_text = f"{margin_sign}{req.margin*100:.1f}%{marginal_note}"
            disp_key = req_key.replace("_", " ").title()
            lines.append(f"  {disp_key:<32} {flag_text:<25} {p_text:<14} {th_text:<12} {m_text:<14}")
        lines.append(f"\n  {colors.DIM}* Note: Architectural requirements are statistical predictions calibrated to balance precision and recall.{colors.RESET}")
    else:
        lines.append(f"  {colors.YELLOW}Architectural requirements predictions unavailable.{colors.RESET}")

    # 6. Complexity, Design Emphasis & Task Intent
    lines.append(f"\n{colors.BOLD}[6] COMPLEXITY, DESIGN EMPHASIS & TASK INTENT{colors.RESET}")
    lines.append(subdiv)
    if scope_pred.status == "success":
        c_flag = f" {colors.YELLOW}[Low Confidence Estimate]{colors.RESET}" if scope_pred.complexity_confidence < 0.55 else ""
        d_flag = f" {colors.YELLOW}[Low Confidence Estimate]{colors.RESET}" if scope_pred.design_emphasis_confidence < 0.55 else ""
        i_flag = f" {colors.YELLOW}[Low Confidence Estimate]{colors.RESET}" if scope_pred.task_intent_confidence < 0.55 else ""

        lines.append(
            f"  • Complexity Tier:        {colors.BOLD}{scope_pred.complexity.upper()}{colors.RESET} "
            f"(Confidence: {scope_pred.complexity_confidence*100:.1f}%){c_flag}"
        )
        lines.append(
            f"  • Design Emphasis:        {colors.BOLD}{scope_pred.design_emphasis.upper()}{colors.RESET} "
            f"(Confidence: {scope_pred.design_emphasis_confidence*100:.1f}%){d_flag}"
        )
        lines.append(
            f"  • Task Intent:            {colors.BOLD}{scope_pred.task_intent.upper()}{colors.RESET} "
            f"(Confidence: {scope_pred.task_intent_confidence*100:.1f}%){i_flag}"
        )
        lines.append(f"  {colors.DIM}* Note: Attribute tiers are model estimates based on training distribution patterns.{colors.RESET}")
    else:
        lines.append(f"  {colors.YELLOW}Secondary attributes unavailable.{colors.RESET}")

    # 7. Prompt Injection / Security Analysis
    lines.append(f"\n{colors.BOLD}[7] PROMPT SECURITY & ADVERSARIAL INJECTION ANALYSIS{colors.RESET}")
    lines.append(subdiv)
    prompt_sec = core.get("prompt_security") or {}
    sec_status = prompt_sec.get("status")

    if sec_status == "success":
        is_mal = prompt_sec.get("is_malicious", False)
        mal_p = prompt_sec.get("malicious_probability", 0.0)
        ben_p = prompt_sec.get("benign_probability", 0.0)

        near_threshold_flag = f" {colors.YELLOW}⚠️ Marginal Confidence (Near 50.0% Decision Threshold){colors.RESET}" if 0.40 <= mal_p <= 0.60 else ""

        if is_mal:
            badge = f"{colors.RED}{colors.BOLD}⚠️  MALICIOUS / INJECTION PATTERN{colors.RESET}"
        else:
            badge = f"{colors.GREEN}{colors.BOLD}✓ BENIGN / NON-ADVERSARIAL{colors.RESET}"

        lines.append(f"  • Classification:         {badge}{near_threshold_flag}")
        lines.append(f"  • Adversarial Malice:     {mal_p*100:.1f}%")
        lines.append(f"  • Benign Confidence:      {ben_p*100:.1f}%")
        lines.append(f"  • Decision Threshold:     50.0% (calibrated)")
        lines.append(f"  • Training Provenance:    Curated MPDD benchmark dataset (MIT/Research license) with developer augmentations.")
        lines.append(f"  • Statistical Disclaimer: {colors.DIM}Statistical pattern classification based on training distribution. Model prediction reflects syntactic and lexical pattern similarity, not conclusive proof of malicious user intent or safe execution.{colors.RESET}")
    elif sec_status in {"model_unavailable", "error"}:
        lines.append(f"  • Classification:         {colors.RED}UNAVAILABLE ({prompt_sec.get('error', 'offline')}){colors.RESET}")
    else:
        lines.append(f"  • Classification:         {colors.DIM}Skipped{colors.RESET}")

    # 8. Application Vulnerability Analysis
    lines.append(f"\n{colors.BOLD}[8] APPLICATION SECURITY RISK & ATTACK SURFACE ANALYSIS{colors.RESET}")
    lines.append(subdiv)
    app_risks = core.get("application_risks") or {}
    app_status = app_risks.get("status")

    if app_status == "success":
        attack_surfaces = app_risks.get("attack_surfaces", [])
        active_surfaces = [s for s in attack_surfaces if s.get("status") == "active_surface"]
        conf_vulns = app_risks.get("confirmed_vulnerabilities", [])
        pot_surfaces = app_risks.get("potential_attack_surfaces", [])
        miss_ctrls = app_risks.get("missing_security_controls", [])
        decl_ctrls = app_risks.get("declared_controls", [])
        decl_negs = app_risks.get("declared_negations", [])
        not_identified = app_risks.get("not_identified_from_info", [])
        risk_ev = app_risks.get("risk_evidence", {})

        # 8.1 Identified Application Attack Surfaces
        lines.append(f"  {colors.BOLD}8.1 Identified Application Attack Surfaces (Interfaces & Components Described in Prompt):{colors.RESET}")
        if active_surfaces:
            for s in active_surfaces:
                s_name = s.get("name", "Attack Surface")
                s_cat = s.get("category", "")
                s_desc = s.get("description", "")
                s_match = s.get("matched_text", "")
                lines.append(f"    • {colors.CYAN}[SURFACE]{colors.RESET} {colors.BOLD}{s_name}{colors.RESET} [{s_cat}] (matched: '{s_match}')")
                lines.append(f"        {colors.DIM}{s_desc}{colors.RESET}")
        else:
            lines.append(f"    {colors.DIM}• None explicitly described in prompt text (pure computational/algorithmic requirement).{colors.RESET}")

        # 8.2 Explicitly Declared Security Controls
        lines.append(f"\n  {colors.BOLD}8.2 Explicitly Declared Security Controls (Present in Prompt Text):{colors.RESET}")
        if decl_ctrls:
            for c in decl_ctrls:
                c_name = c.get("name", "Security Control")
                c_matched = c.get("matched_text", "")
                c_cat = c.get("category", "")
                c_desc = c.get("description", "")
                lines.append(f"    {colors.GREEN}[✓ CONTROL]{colors.RESET} {colors.BOLD}{c_name}{colors.RESET} [{c_cat}] (matched: '{c_matched}')")
                lines.append(f"        {colors.DIM}{c_desc}{colors.RESET}")
        else:
            lines.append(f"    {colors.DIM}• None identified from available information (prompt does not explicitly specify defensive controls).{colors.RESET}")

        # 8.3 Missing or Insufficient Security Controls
        lines.append(f"\n  {colors.BOLD}8.3 Missing or Insufficient Security Controls (Not Present in Specification):{colors.RESET}")
        missing_entries = []
        for s in active_surfaces:
            for cid in s.get("missing_controls_in_spec", []):
                ctrl_meta = {
                    "password_hashing": "Password Hashing & Salt (Argon2id/bcrypt)",
                    "mfa": "Multi-Factor Authentication (MFA/TOTP)",
                    "rate_limiting": "Rate Limiting & Request Throttling",
                    "secure_sessions": "Secure Session Cookies (HttpOnly, SameSite) & Token Validation",
                    "parameterized_queries": "Parameterized Queries & Prepared Statements",
                    "data_encryption": "Data Encryption at Rest & in Transit (TLS/AES)",
                    "authorization_checks": "Granular Authorization & RBAC Checks",
                    "anti_csrf": "Anti-CSRF Synchronizer Tokens",
                    "file_validation": "Strict File Magic-Bytes Validation & Isolated Non-Executable Storage",
                    "network_egress_control": "Outbound URL / Network Egress Allowlisting"
                }
                c_label = ctrl_meta.get(cid, cid.replace("_", " ").title())
                missing_entries.append((c_label, s.get("name")))

        if missing_entries:
            for c_label, s_name in missing_entries[:6]:
                lines.append(f"    {colors.YELLOW}[!]{colors.RESET} {colors.BOLD}{c_label}{colors.RESET} — Not specified for {s_name}")
        else:
            lines.append(f"    {colors.DIM}• None identified from available information.{colors.RESET}")

        # 8.4 Confirmed Vulnerabilities
        lines.append(f"\n  {colors.BOLD}8.4 Confirmed Vulnerabilities (Explicit Insecure Instructions):{colors.RESET}")
        if conf_vulns:
            lines.append(f"    {colors.RED}{colors.BOLD}✖ CONFIRMED VULNERABILITIES DETECTED ({len(conf_vulns)}):{colors.RESET}")
            for v in conf_vulns:
                ev_desc = risk_ev.get(v, {}).get("description", "Explicit insecure instruction detected.")
                meta = RISK_METADATA.get(v, {})
                title = meta.get("title", v.upper())
                lines.append(f"      • {colors.BOLD}{title}{colors.RESET}: {ev_desc}")
        else:
            lines.append(f"    {colors.GREEN}✓ None identified from available information (prompt contains no explicit instructions to bypass controls or store unhashed secrets).{colors.RESET}")

        # 8.5 Potential Risks Inferred from Requirements
        lines.append(f"\n  {colors.BOLD}8.5 Potential Risks Inferred from Requirements:{colors.RESET}")
        if pot_surfaces:
            for s in pot_surfaces:
                ev_desc = risk_ev.get(s, {}).get("description", "Inferred architectural attack surface.")
                meta = RISK_METADATA.get(s, {})
                title = meta.get("title", s.upper())
                lines.append(f"    {colors.CYAN}[POTENTIAL]{colors.RESET} {colors.BOLD}{title}{colors.RESET}: {ev_desc}")
            lines.append(f"    {colors.DIM}* Note: Potential risks inferred from requirements indicate architectural attack surfaces, not confirmed code vulnerabilities.{colors.RESET}")
        else:
            lines.append(f"    {colors.DIM}• None identified from available information.{colors.RESET}")

        # 8.6 Not Identified from Available Information
        lines.append(f"\n  {colors.BOLD}8.6 Risk Categories Not Identified from Available Information:{colors.RESET}")
        if not_identified:
            readable_unid = [RISK_METADATA.get(r, {}).get("title", r) for r in not_identified]
            lines.append(f"    {colors.DIM}• {', '.join(readable_unid[:8])}{' ...' if len(readable_unid) > 8 else ''}{colors.RESET}")
            lines.append(f"    {colors.DIM}* Note: Absence of security requirements or risk signals in prompt text is not proof that the resulting implementation will be secure.{colors.RESET}")
        else:
            lines.append(f"    {colors.DIM}• All surveyed categories triggered architectural or specific signals.{colors.RESET}")

        # Declared Negations
        if decl_negs:
            lines.append(f"\n  {colors.DIM}• Declared Negations (Features Excluded): {', '.join([n.get('feature', '') for n in decl_negs])}{colors.RESET}")

    elif app_status in {"model_unavailable", "error"}:
        lines.append(f"  {colors.RED}Application Risk Model unavailable ({app_risks.get('error', 'offline')}).{colors.RESET}")
    else:
        lines.append(f"  {colors.DIM}Application Risk Analysis skipped.{colors.RESET}")

    # 9. Sensitive Data Detection
    lines.append(f"\n{colors.BOLD}[9] SENSITIVE DATA & CREDENTIAL DETECTION{colors.RESET}")
    lines.append(subdiv)
    sens = analysis_data["sensitive_data"]
    if sens.get("detected"):
        findings = sens.get("findings", [])
        lines.append(f"  {colors.RED}{colors.BOLD}✖ EXPOSED CREDENTIALS DETECTED IN PROMPT TEXT ({len(findings)} finding(s)):{colors.RESET}")
        for f in findings:
            f_type = f.get("type", "credential")
            f_desc = f.get("description", "Secret")
            f_red = f.get("redacted", "[REDACTED]")
            lines.append(f"    • {colors.BOLD}{f_desc}{colors.RESET} ({f_type}): {colors.YELLOW}{f_red}{colors.RESET}")
        lines.append(f"    {colors.RED}Action Required: Immediately revoke and remove exposed keys from prompt text.{colors.RESET}")
    else:
        lines.append(f"  {colors.GREEN}✓ No credential patterns detected within the submitted prompt text.{colors.RESET}")
    lines.append(f"  {colors.DIM}* Scope Disclaimer: Scanning applies exclusively to the submitted prompt text. Local repositories, backend databases, and environment/deployment configurations have NOT been scanned.{colors.RESET}")

    # 10. Security Recommendations
    lines.append(f"\n{colors.BOLD}[10] ACTIONABLE SECURITY RECOMMENDATIONS{colors.RESET}")
    lines.append(subdiv)
    lines.append(f"  {colors.DIM}* Defensive Architectural Guidance & Best Practices (does not indicate existing code defects):{colors.RESET}\n")
    recs = analysis_data["recommendations"]
    if recs:
        for idx, r in enumerate(recs, 1):
            prio = r.get("priority", "MEDIUM")
            cat = r.get("category", "General")
            title = r.get("title", "")
            guidance = r.get("guidance", "")

            if prio == "CRITICAL":
                prio_colored = f"{colors.RED}{colors.BOLD}[CRITICAL]{colors.RESET}"
            elif prio == "HIGH":
                prio_colored = f"{colors.RED}[HIGH]{colors.RESET}"
            elif prio == "MEDIUM":
                prio_colored = f"{colors.YELLOW}[MEDIUM]{colors.RESET}"
            else:
                prio_colored = f"{colors.CYAN}[INFO]{colors.RESET}"

            lines.append(f"  {idx}. {prio_colored} {colors.BOLD}{title}{colors.RESET} ({cat})")
            lines.append(f"     {colors.DIM}{guidance}{colors.RESET}")
    else:
        lines.append(f"  {colors.GREEN}• Standard defensive practices apply; no specific vulnerabilities requiring action.{colors.RESET}")

    # 11. Prompt Optimization
    lines.append(f"\n{colors.BOLD}[11] PROMPT OPTIMIZATION & HARDENING{colors.RESET}")
    lines.append(subdiv)
    opt: OptimizationResult = analysis_data["optimization"]
    if analysis_data["optimization_status"] == "success":
        # Separate declared controls from recommended controls
        decl_in_opt = getattr(opt, "explicitly_declared_controls", [])
        recom_in_opt = getattr(opt, "recommended_controls_added", [])

        if decl_in_opt:
            lines.append(f"  {colors.GREEN}✓ Explicitly Declared Controls Preserved from Prompt:{colors.RESET}")
            for d in decl_in_opt:
                lines.append(f"    • {d}")

        if recom_in_opt:
            lines.append(f"\n  {colors.CYAN}ℹ Recommended Controls Introduced During Optimization:{colors.RESET}")
            for rc in recom_in_opt:
                lines.append(f"    • {rc}")

        lines.append(f"\n  • {colors.BOLD}Reason for Prompt Expansion:{colors.RESET} {getattr(opt, 'expansion_reason', 'Adds defensive constraints.')}")
        lines.append(f"  • {colors.BOLD}Token Efficiency Status:{colors.RESET} {getattr(opt, 'token_efficiency_statement', 'Expanded with explicit guardrails; not token-reduced.')}")

        lines.append(f"\n  {colors.BOLD}Optimized Prompt Text:{colors.RESET}")
        lines.append("  " + "-" * 76)
        for opt_line in opt.optimized_prompt.splitlines():
            lines.append(f"  {opt_line}")
        lines.append("  " + "-" * 76)
    else:
        lines.append(f"  {colors.YELLOW}Prompt optimization unavailable: {opt.changes_summary[0] if opt.changes_summary else 'offline'}{colors.RESET}")

    # 12. Token Estimation
    lines.append(f"\n{colors.BOLD}[12] TOKEN ESTIMATION (ORIGINAL VS. OPTIMIZED PROMPT){colors.RESET}")
    lines.append(subdiv)
    token_est = analysis_data["token_estimation"]
    if token_est.get("status") == "success":
        lines.append(f"  {'Tokenizer / Provider':<36} {'Original':<12} {'Optimized':<12} {'Delta':<14} {'Method':<20}")
        lines.append("  " + "-" * 88)
        for comp in token_est.get("comparisons", []):
            label = comp["label"]
            orig_c = comp["orig_tokens"]
            opt_c = comp["opt_tokens"]
            delta = comp["delta"]
            pct = comp["pct_change"]
            method = comp["method"]
            d_str = f"{delta:+d} ({pct:+.1f}%)"
            lines.append(f"  {label:<36} {orig_c:<12} {opt_c:<12} {d_str:<14} {method:<20}")
        lines.append(f"\n  {colors.DIM}* Reason for Expansion: Expansion intentionally introduces concrete architectural constraints, validation rules, and error handling to reduce LLM code omissions. It is not token-reduced.{colors.RESET}")
        lines.append(f"  {colors.DIM}* Raw text tokenization; actual API billing depends on system prompts, tools, and provider caching.{colors.RESET}")
    else:
        lines.append(f"  {colors.YELLOW}Token estimation unavailable.{colors.RESET}")

    # 13. Comparison Section (IDE Assistants and/or Foundation Models)
    has_ide = ide_comparison_report is not None and bool(getattr(ide_comparison_report, "ranked_products", []))
    has_model = comparison_report is not None and bool(getattr(comparison_report, "ranked_models", []))

    if has_ide and has_model:
        lines.append(f"\n{colors.BOLD}[13] COMPARISON (IDE ASSISTANTS & FOUNDATION MODELS){colors.RESET}")
        lines.append(subdiv)
        _format_ide_comparison_section(lines, ide_comparison_report, colors, subdiv, prefix="13A")
        _format_model_comparison_section(lines, comparison_report, colors, subdiv, prefix="13B")
    elif has_ide:
        _format_ide_comparison_section(lines, ide_comparison_report, colors, subdiv, prefix="13")
    elif has_model:
        _format_model_comparison_section(lines, comparison_report, colors, subdiv, prefix="13")
    else:
        lines.append(f"\n{colors.BOLD}[13] MODEL COMPARISON{colors.RESET}")
        lines.append(subdiv)
        lines.append(f"  {colors.DIM}Model comparison was not requested for this session.{colors.RESET}")

    lines.append("\n" + div)
    lines.append(f"{colors.BOLD}{colors.CYAN}                       END OF PROMPTSHIELD REPORT{colors.RESET}")
    lines.append(div + "\n")

    return "\n".join(lines)


# ==============================================================================
# MAIN CLI WORKFLOW DRIVER
# ==============================================================================

def run_workflow_cli(interactive: bool = True) -> int:
    """
    Drives the complete user-facing workflow:
    STEP 1: Enter prompt (multiline)
    STEP 2: Model checks scope (Frontend, Backend, Design, Full Stack, API, etc.) & user confirms scope
    STEP 3: Optional comparison selection (1: Continue, 2: IDE Compare, 3: Model Compare, 4: Both)
            Lists eligible product/model names if comparison chosen
    STEP 4: Run complete security scanning, comparison, and deterministic prompt optimization
    STEP 5: Consolidated report display
    """
    colors = Colors(enabled=is_color_supported())
    print(get_cli_banner(colors))

    session_count = 0

    while True:
        session_count += 1

        # STEP 1: Enter Prompt
        prompt = read_multiline_prompt(
            colors=colors,
            prompt_label="Enter the prompt you want PromptShield AI to analyze:"
        )

        if prompt is None:
            if session_count == 1:
                print(f"{colors.YELLOW}Session cancelled by user.{colors.RESET}")
            else:
                print(f"\n{colors.CYAN}Exiting PromptShield AI Workflow. Goodbye!{colors.RESET}")
            return 0

        if not prompt.strip():
            print(f"{colors.RED}Prompt cannot be empty. Please enter a valid prompt.{colors.RESET}")
            continue

        # STEP 2: Model Scope Check & Architectural Requirements
        print(f"\n{colors.BOLD}{colors.CYAN}Evaluating implementation scope & architectural requirements...{colors.RESET}")
        scope_pred = scope_predictor.predict(prompt)
        task_cat = task_classifier.classify_task(prompt)
        display_scope_check_results(colors=colors, scope_pred=scope_pred, task_category=task_cat)

        # Confirm or Select Development Scope
        scope_choice = select_development_scope(
            colors=colors,
            detected_scope=scope_pred.primary_scope if scope_pred.status == "success" else None
        )
        if scope_choice is None:
            print(f"{colors.YELLOW}Session cancelled by user.{colors.RESET}")
            return 0

        scope_id, scope_display = scope_choice

        # STEP 3: Optional Comparison Selection (Continue / IDE / Model / Both)
        comp_choice = prompt_comparison_options(colors=colors)
        if comp_choice == "cancel":
            print(f"{colors.YELLOW}Session cancelled by user.{colors.RESET}")
            return 0

        # List names if comparison selected
        if comp_choice in {"ide", "both"}:
            list_available_ide_assistants(colors=colors)
        if comp_choice in {"model", "both"}:
            list_available_foundation_models(colors=colors)

        # STEP 4: Run Complete PromptShield Analysis & Comparisons
        print(f"\n{colors.BOLD}{colors.CYAN}Running complete PromptShield AI analysis...{colors.RESET}")
        analysis_data = run_complete_promptshield_analysis(
            prompt=prompt,
            selected_scope_id=scope_id,
            selected_scope_display=scope_display,
            precomputed_scope_pred=scope_pred
        )
        print(f"{colors.GREEN}✓ Core security analysis & prompt optimization completed successfully.{colors.RESET}")

        ide_report = None
        model_report = None
        if comp_choice in {"ide", "both"}:
            print(f"\n{colors.BOLD}{colors.CYAN}Evaluating IDE coding assistant suitability and multi-criteria ranking...{colors.RESET}")
            try:
                ide_report = ide_comparison_engine.compare(prompt=prompt)
                if ide_report:
                    print(f"{colors.GREEN}✓ IDE comparison completed successfully.{colors.RESET}")
            except Exception as e:
                print(f"{colors.YELLOW}IDE comparison failed: {str(e)}{colors.RESET}")

        if comp_choice in {"model", "both"}:
            print(f"\n{colors.BOLD}{colors.CYAN}Evaluating foundation model suitability and multi-criteria ranking...{colors.RESET}")
            try:
                model_report = run_model_comparison(prompt=prompt)
                if model_report:
                    print(f"{colors.GREEN}✓ Foundation model comparison completed successfully.{colors.RESET}")
            except Exception as e:
                print(f"{colors.YELLOW}Model comparison failed: {str(e)}{colors.RESET}")

        # STEP 5: Consolidated Report
        report_text = format_consolidated_report(
            analysis_data=analysis_data,
            comparison_report=model_report,
            colors=colors,
            ide_comparison_report=ide_report
        )
        print(report_text)

        # Check if running interactively
        if not interactive or not sys.stdin.isatty():
            break

        # Ask to analyze another prompt
        try:
            again = input(f"{colors.BOLD}Would you like to analyze another prompt? (y/n):{colors.RESET} ").strip().lower()
            if again not in {"y", "yes"}:
                print(f"\n{colors.CYAN}Thank you for using PromptShield AI. Goodbye!{colors.RESET}")
                break
        except (KeyboardInterrupt, EOFError):
            print(f"\n{colors.CYAN}Exiting PromptShield AI Workflow. Goodbye!{colors.RESET}")
            break

    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_workflow_cli(interactive=True))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(0)


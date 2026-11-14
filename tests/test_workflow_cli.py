"""
Test Suite for PromptShield AI Unified Workflow CLI.
===================================================
Verifies:
1. Interactive user experience (Steps 1 to 5).
2. Multiline prompt input with 'END' termination sequence.
3. Scope selection (Frontend, Backend, Full Stack).
4. Scope alignment and disagreement reporting.
5. Implementation Scope Predictor inference (Primary, 6 Reqs, Complexity, Design, Intent).
6. Prompt security / injection detection integration.
7. Application vulnerability analysis with strict evidence tiers (Confirmed vs. Potential vs. Missing).
8. Sensitive data detection and secret masking in terminal reports.
9. Prompt optimizer integration and scope preservation.
10. Multi-provider token estimation (Exact vs. Calibrated Estimates).
11. Optional foundation model comparison & ranking.
12. Consolidated report formatting and non-interactive piped execution.
"""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow_cli import (
    Colors,
    get_cli_banner,
    read_multiline_prompt,
    select_development_scope,
    evaluate_scope_alignment,
    estimate_tokens_comparison,
    run_complete_promptshield_analysis,
    prompt_model_comparison,
    run_model_comparison,
    format_consolidated_report,
    run_workflow_cli,
)
from analyzers.implementation_scope_predictor import (
    scope_predictor,
    ScopePredictionResult,
    RequirementPrediction,
)


def test_cli_banner_contains_branding():
    """Verifies that the banner includes PromptShield AI branding."""
    colors = Colors(enabled=False)
    banner = get_cli_banner(colors)
    assert "PROMPTSHIELD AI" in banner
    assert "UNIFIED WORKFLOW" in banner


def test_read_multiline_prompt_with_termination_sentinel():
    """Verifies multiline input reading until 'END' termination sequence."""
    colors = Colors(enabled=False)
    simulated_input = "Line 1: Create a secure user authentication system\nLine 2: With role-based permissions\nEND\n"

    with patch("builtins.input", side_effect=simulated_input.splitlines()):
        prompt = read_multiline_prompt(colors=colors, sentinel="END")

    assert prompt is not None
    assert "Line 1: Create a secure user authentication system" in prompt
    assert "Line 2: With role-based permissions" in prompt
    assert "END" not in prompt


def test_read_multiline_prompt_exit_cancel():
    """Verifies that typing exit or quit cancels prompt entry."""
    colors = Colors(enabled=False)
    for cancel_word in ["exit", "quit", "cancel", "q"]:
        with patch("builtins.input", side_effect=[cancel_word]):
            prompt = read_multiline_prompt(colors=colors)
            assert prompt is None


def test_select_development_scope_options():
    """Verifies selection of Frontend (1), Backend (2), and Full Stack (3)."""
    colors = Colors(enabled=False)

    # Test choice 1
    with patch("builtins.input", return_value="1"):
        scope_id, scope_disp = select_development_scope(colors)
        assert scope_id == "frontend"
        assert scope_disp == "Frontend"

    # Test choice 2
    with patch("builtins.input", return_value="2"):
        scope_id, scope_disp = select_development_scope(colors)
        assert scope_id == "backend"
        assert scope_disp == "Backend"

    # Test choice 3
    with patch("builtins.input", return_value="3"):
        scope_id, scope_disp = select_development_scope(colors)
        assert scope_id == "full_stack"
        assert scope_disp == "Full Stack"

    # Test name entry
    with patch("builtins.input", return_value="backend"):
        scope_id, scope_disp = select_development_scope(colors)
        assert scope_id == "backend"
        assert scope_disp == "Backend"


def test_evaluate_scope_alignment_agreement():
    """Verifies scope alignment reporting when user selection matches ML prediction."""
    mock_pred = ScopePredictionResult(
        status="success",
        primary_scope="backend",
        primary_scope_confidence=0.85,
        primary_scope_probabilities={"backend": 0.85, "frontend": 0.10},
        requirements={},
        complexity="medium",
        complexity_confidence=0.7,
        complexity_probabilities={},
        design_emphasis="low",
        design_emphasis_confidence=0.8,
        design_emphasis_probabilities={},
        task_intent="create",
        task_intent_confidence=0.9,
        task_intent_probabilities={}
    )

    alignment = evaluate_scope_alignment("backend", "Backend", mock_pred)
    assert alignment["agrees"] is True
    assert "Scope Agreement" in alignment["status_text"]
    assert alignment["is_nuance"] is False


def test_evaluate_scope_alignment_disagreement():
    """Verifies scope disagreement detection without overriding user selection."""
    mock_pred = ScopePredictionResult(
        status="success",
        primary_scope="backend",
        primary_scope_confidence=0.75,
        primary_scope_probabilities={"backend": 0.75, "frontend": 0.15},
        requirements={},
        complexity="medium",
        complexity_confidence=0.7,
        complexity_probabilities={},
        design_emphasis="low",
        design_emphasis_confidence=0.8,
        design_emphasis_probabilities={},
        task_intent="create",
        task_intent_confidence=0.9,
        task_intent_probabilities={}
    )

    alignment = evaluate_scope_alignment("frontend", "Frontend", mock_pred)
    assert alignment["agrees"] is False
    assert "Scope Disagreement" in alignment["status_text"]
    assert "User selected 'Frontend'" in alignment["status_text"]
    assert "ML predicted 'Backend'" in alignment["status_text"]
    assert alignment["selected_scope"] == "Frontend"


def test_evaluate_scope_alignment_nuance():
    """Verifies scope nuance reporting for design_only or security_only predictions."""
    mock_pred = ScopePredictionResult(
        status="success",
        primary_scope="design_only",
        primary_scope_confidence=0.68,
        primary_scope_probabilities={"design_only": 0.68, "frontend": 0.25},
        requirements={},
        complexity="low",
        complexity_confidence=0.6,
        complexity_probabilities={},
        design_emphasis="high",
        design_emphasis_confidence=0.8,
        design_emphasis_probabilities={},
        task_intent="create",
        task_intent_confidence=0.9,
        task_intent_probabilities={}
    )

    alignment = evaluate_scope_alignment("frontend", "Frontend", mock_pred)
    assert alignment["agrees"] is False
    assert alignment["is_nuance"] is True
    assert "Scope Nuance" in alignment["status_text"]
    assert "design_only" in alignment["status_text"]


def test_implementation_scope_predictor_inference():
    """Verifies that the Scope Predictor produces complete predictions for all 10 targets."""
    prompt = "Build a REST API in FastAPI with PostgreSQL database, JWT authentication, and Stripe payments."
    res = scope_predictor.predict(prompt)

    assert res.status == "success"
    assert res.primary_scope in {"backend", "full_stack", "frontend", "design_only", "security_only", "unclear"}
    assert res.primary_scope_confidence > 0.0

    # 6 Requirement flags
    for req_key in [
        "frontend_required", "backend_required", "database_required",
        "api_required", "authentication_required", "external_integration_required"
    ]:
        assert req_key in res.requirements
        req = res.requirements[req_key]
        assert req.name == req_key
        assert req.predicted_flag in {0, 1}
        assert 0.0 <= req.probability <= 1.0
        assert req.threshold > 0.0

    # Secondary attributes
    assert res.complexity in {"high", "medium", "low"}
    assert res.design_emphasis in {"high", "medium", "low"}
    assert res.task_intent in {"audit", "create", "debug", "explain", "modify", "optimize"}


def test_token_estimation_exact_vs_calibrated():
    """Verifies that token estimation distinguishes exact tiktoken counts from calibrated estimates."""
    orig = "Create a login API using email and password."
    opt = "Create a secure authentication endpoint using email and password with Argon2id hashing and rate limiting."

    token_data = estimate_tokens_comparison(orig, opt)
    assert token_data["status"] == "success"
    comps = token_data["comparisons"]
    assert len(comps) == 3

    gpt_comp = next(c for c in comps if "GPT-4o" in c["label"])
    assert gpt_comp["is_exact"] is True
    assert "Exact" in gpt_comp["method"]
    assert gpt_comp["orig_tokens"] > 0
    assert gpt_comp["opt_tokens"] > gpt_comp["orig_tokens"]

    claude_comp = next(c for c in comps if "Claude" in c["label"])
    assert claude_comp["is_exact"] is False
    assert "Calibrated" in claude_comp["method"]


def test_sensitive_data_masking_in_report():
    """Verifies that secrets are masked in the consolidated report."""
    colors = Colors(enabled=False)
    prompt_with_key = "Connect to OpenAI using sk-123456789012345678901234 and fetch completion."
    analysis_data = run_complete_promptshield_analysis(
        prompt=prompt_with_key,
        selected_scope_id="backend",
        selected_scope_display="Backend"
    )

    report = format_consolidated_report(analysis_data, comparison_report=None, colors=colors)

    # Raw secret must NOT appear
    assert "123456789012345678901234" not in report
    # Redacted preview must appear
    assert "sk-1...234" in report
    assert "EXPOSED CREDENTIALS DETECTED" in report


def test_confirmed_vulnerability_distinction():
    """Verifies that explicit insecure instructions are flagged as confirmed vulnerabilities."""
    colors = Colors(enabled=False)
    vuln_prompt = "Create a login API that stores passwords in plaintext without hashing."
    analysis_data = run_complete_promptshield_analysis(
        prompt=vuln_prompt,
        selected_scope_id="backend",
        selected_scope_display="Backend"
    )

    report = format_consolidated_report(analysis_data, comparison_report=None, colors=colors)
    assert "CONFIRMED VULNERABILITIES" in report
    assert "Authentication Weakness" in report or "Sensitive Data Exposure" in report
    assert "plaintext" in report


def test_missing_security_controls_distinction():
    """Verifies that missing security controls are not automatically treated as confirmed vulnerabilities."""
    colors = Colors(enabled=False)
    neutral_prompt = "Create a login API using email and password."
    analysis_data = run_complete_promptshield_analysis(
        prompt=neutral_prompt,
        selected_scope_id="backend",
        selected_scope_display="Backend"
    )

    report = format_consolidated_report(analysis_data, comparison_report=None, colors=colors)
    assert "Confirmed Vulnerabilities: None detected" in report
    assert "POTENTIAL ATTACK SURFACES" in report


def test_prompt_model_comparison_prompt():
    """Verifies that user can select Yes (1) or No (2) for model comparison."""
    colors = Colors(enabled=False)

    with patch("builtins.input", return_value="1"):
        assert prompt_model_comparison(colors) is True

    with patch("builtins.input", return_value="yes"):
        assert prompt_model_comparison(colors) is True

    with patch("builtins.input", return_value="2"):
        assert prompt_model_comparison(colors) is False

    with patch("builtins.input", return_value="no"):
        assert prompt_model_comparison(colors) is False


def test_full_workflow_end_to_end_without_comparison():
    """Verifies complete analysis and report generation when model comparison is skipped."""
    colors = Colors(enabled=False)
    prompt = "Create a customer dashboard with React, displaying monthly sales metrics."

    analysis_data = run_complete_promptshield_analysis(
        prompt=prompt,
        selected_scope_id="frontend",
        selected_scope_display="Frontend"
    )

    report = format_consolidated_report(analysis_data, comparison_report=None, colors=colors)

    assert "[1] ORIGINAL PROMPT" in report
    assert "[2 & 3] DEVELOPMENT SCOPE & PREDICTED IMPLEMENTATION SCOPE" in report
    assert "[4] TASK CLASSIFICATION" in report
    assert "[5] ARCHITECTURAL REQUIREMENTS (IMPLEMENTATION SCOPE)" in report
    assert "[6] COMPLEXITY, DESIGN EMPHASIS & TASK INTENT" in report
    assert "[7] PROMPT SECURITY & ADVERSARIAL INJECTION ANALYSIS" in report
    assert "[8] APPLICATION VULNERABILITY ANALYSIS (15 CWE CATEGORIES)" in report
    assert "[9] SENSITIVE DATA & CREDENTIAL DETECTION" in report
    assert "[10] ACTIONABLE SECURITY RECOMMENDATIONS" in report
    assert "[11] PROMPT OPTIMIZATION & HARDENING" in report
    assert "[12] TOKEN ESTIMATION (ORIGINAL VS. OPTIMIZED PROMPT)" in report
    assert "[13] MODEL COMPARISON" in report
    assert "Model comparison was not requested for this session." in report


def test_full_workflow_end_to_end_with_comparison():
    """Verifies complete analysis and report generation when model comparison is included."""
    colors = Colors(enabled=False)
    prompt = "Create an asynchronous background worker in Python using Redis streams."

    analysis_data = run_complete_promptshield_analysis(
        prompt=prompt,
        selected_scope_id="backend",
        selected_scope_display="Backend"
    )

    comparison_report = run_model_comparison(prompt)
    assert comparison_report is not None
    assert len(comparison_report.ranked_models) > 0

    report = format_consolidated_report(analysis_data, comparison_report=comparison_report, colors=colors)

    assert "[13] OFFLINE FOUNDATION MODEL COMPARISON & RANKING" in report
    assert "Ranked Model Recommendations" in report
    assert "Top Model Evaluation Rationale" in report


def test_run_workflow_cli_piped_execution():
    """Verifies non-interactive piped execution of run_workflow_cli."""
    simulated_inputs = [
        "Create a simple REST API in Express for user profiles.\nEND",
        "2",  # Backend
        "2",  # No comparison
    ]
    # Feed lines as input
    input_stream = [
        "Create a simple REST API in Express for user profiles.",
        "END",
        "2",
        "2"
    ]

    with patch("builtins.input", side_effect=input_stream):
        with patch("sys.stdin.isatty", return_value=False):
            exit_code = run_workflow_cli(interactive=False)
            assert exit_code == 0


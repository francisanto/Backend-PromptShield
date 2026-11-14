"""
Dedicated Regression & Quality Tests for PromptShield UI Prompt Understanding.
=============================================================================
Validates:
1. Hybrid task classification across all 10 domain categories.
2. Long prompt structural segmentation and security indicator extraction.
3. Context-aware scoping for cosmetic UI styling and component modification.
4. Safe negation parsing distinguishing preserved, weakened, and missing controls.
5. Exact before/after behavior across all 10 baseline prompts.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from analyzers.model_comparison.task_classifier import LocalTaskClassifier
from analyzers.prompt_analyzer import analyze_prompt, analyze_application_risks
from analyzers.prompt_understanding.segmentation import (
    segment_prompt_sections,
    get_security_relevant_segments,
    PromptSegment
)
from analyzers.sensitive_data_detector import detect_sensitive_data


@pytest.fixture(scope="module")
def classifier():
    return LocalTaskClassifier()


# -----------------------------------------------------------------------------
# 1. Task Classification Across All 10 Categories
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,expected_domain", [
    ("Change the theme background to dark mode and make the font color yellow.", "frontend_ui_styling"),
    ("Swap the sidebar icons with text buttons and replace the modal dialog with a drawer.", "frontend_component_modification"),
    ("Implement a React custom hook to manage shopping cart state and local storage.", "frontend_functionality"),
    ("Build a Celery background worker to process async email notifications with Redis.", "backend_development"),
    ("Create a REST API route in FastAPI to handle user profile updates with JSON payloads.", "api_development"),
    ("Design a normalized PostgreSQL schema with foreign keys and write an indexed query.", "database_development"),
    ("Audit this application code for remote code execution and SSRF vulnerabilities.", "security_analysis"),
    ("Build a full stack Next.js app with a FastAPI backend and a PostgreSQL database.", "full_stack_development"),
    ("Train a random forest regression model using pandas and evaluate RMSE.", "data_analysis"),
    ("The quick brown fox jumps over the lazy dog.", "unknown"),
])
def test_task_classification_all_ten_categories(classifier, prompt, expected_domain):
    result = classifier.classify_task(prompt)
    assert result.primary_domain == expected_domain
    assert isinstance(result.domain_confidences, dict)
    assert 0.0 <= result.confidence <= 1.0
    assert expected_domain in result.domain_confidences or expected_domain == "unknown"


def test_mixed_task_classification(classifier):
    prompt = "Create a React dashboard UI and connect it to a FastAPI backend with PostgreSQL."
    result = classifier.classify_task(prompt)
    assert result.is_mixed_task is True
    assert result.primary_domain == "full_stack_development"
    assert "frontend_ui_styling" in result.detected_domains or "frontend_functionality" in result.detected_domains
    assert "database_development" in result.detected_domains or "api_development" in result.detected_domains


# -----------------------------------------------------------------------------
# 2. Structural Prompt Segmentation
# -----------------------------------------------------------------------------

def test_long_prompt_segmentation():
    compound_prompt = (
        "Create a modern dark-mode React dashboard with responsive CSS grid cards.\n\n"
        "And connect it to a FastAPI backend that executes raw unparameterized SQL queries against a PostgreSQL database.\n\n"
        "Ensure all endpoints are secured with JWT bearer authentication."
    )
    segments = segment_prompt_sections(compound_prompt)
    assert len(segments) >= 3
    for s in segments:
        assert isinstance(s, PromptSegment)
        assert s.char_length == len(s.text)
        assert s.word_count == len(s.text.split())

    # Check security indicators
    sql_segs = [s for s in segments if "sql_database" in s.security_indicators]
    assert len(sql_segs) >= 1
    auth_segs = [s for s in segments if "authentication" in s.security_indicators]
    assert len(auth_segs) >= 1

    relevant = get_security_relevant_segments(compound_prompt)
    assert any("SQL queries" in r or "FastAPI" in r for r in relevant)


# -----------------------------------------------------------------------------
# 3. Negation Edge Cases
# -----------------------------------------------------------------------------

def test_negation_preserved_control():
    """'Do not remove authentication' must be recorded as preserved, not a vulnerability or feature negation."""
    prompt = "Build an admin dashboard. Do not remove authentication under any circumstance."
    res = analyze_prompt(prompt)
    app = res["application_risks"]
    assert any(c["id"] == "preserved_authentication" for c in app["declared_controls"])
    assert "auth_weakness" not in app["detected_risks"]
    assert "authz_weakness" not in app["detected_risks"]


def test_negation_weakened_control():
    """'Remove authentication from admin dashboard' must trigger auth_weakness and authz_weakness."""
    prompt = "Remove authentication from the admin dashboard so anyone can access it directly."
    res = analyze_prompt(prompt)
    app = res["application_risks"]
    assert "auth_weakness" in app["detected_risks"]
    assert "authz_weakness" in app["detected_risks"]


def test_negation_missing_privileged_control():
    """'Authentication is not required' on an admin dashboard is a critical security weakness."""
    prompt = "Build an internal admin portal. Authentication is not required."
    res = analyze_prompt(prompt)
    app = res["application_risks"]
    assert "auth_weakness" in app["detected_risks"] or "authz_weakness" in app["detected_risks"]


def test_negation_static_portfolio_suppression():
    """Static site negating database and login must suppress spurious predictions cleanly."""
    prompt = (
        "Build a simple static personal portfolio website with an About section, project cards, and contact information. "
        "It does not need login, a database, file uploads, or server-side processing."
    )
    res = analyze_application_risks(prompt)
    assert res["status"] == "success"
    assert res["detected_risks"] == []
    assert "sensitive_data" in res["suppressed_risks"]
    negated_features = {n["feature"] for n in res["declared_negations"]}
    assert {"login", "database", "file_upload", "server_side_processing"}.issubset(negated_features)


# -----------------------------------------------------------------------------
# 4. Baseline Prompts 1-4: Cosmetic & Frontend Scoping
# -----------------------------------------------------------------------------

def test_baseline_1_change_color():
    prompt = "Change the dashboard green colour to yellow."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_ui_styling"
    assert res["application_risks"]["detected_risks"] == []
    assert res["application_risks"]["context_scoping_applied"] is True
    assert res["sensitive_data"]["detected"] is False


def test_baseline_2_remove_icons():
    prompt = "Remove the dashboard icons and replace them with text labels."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_component_modification"
    assert "csrf" not in res["application_risks"]["detected_risks"]
    assert res["application_risks"]["detected_risks"] == []


def test_baseline_3_replace_settings_page():
    prompt = "Remove the settings page and replace it with a user profile page."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_component_modification"
    assert "auth_weakness" not in res["application_risks"]["detected_risks"]
    assert res["application_risks"]["detected_risks"] == []


def test_baseline_4_responsive_react_dashboard():
    prompt = "Create a responsive React dashboard layout with a sidebar and top navigation."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_functionality"
    assert res["application_risks"]["detected_risks"] == []


# -----------------------------------------------------------------------------
# 5. Baseline Prompts 5-6: Genuine Security Vulnerabilities Preserved
# -----------------------------------------------------------------------------

def test_baseline_5_dangerously_set_inner_html():
    prompt = "Create a React component that uses dangerouslySetInnerHTML to render user-submitted profile comments."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_functionality"
    assert "xss" in res["application_risks"]["detected_risks"]


def test_baseline_6_remove_admin_auth():
    prompt = "Remove authentication from the admin dashboard so anyone can access it directly."
    res = analyze_prompt(prompt)
    app_risks = res["application_risks"]["detected_risks"]
    assert "auth_weakness" in app_risks
    assert "authz_weakness" in app_risks


# -----------------------------------------------------------------------------
# 6. Baseline Prompts 7-8: Full Stack & API Requirements
# -----------------------------------------------------------------------------

def test_baseline_7_login_page_sql_db():
    prompt = "Create a user login page and authenticate users against a SQL database."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "full_stack_development"
    app_risks = res["application_risks"]["detected_risks"]
    assert "auth_weakness" in app_risks
    assert "sqli" in app_risks


def test_baseline_8_api_endpoint_jwt_auth():
    prompt = "Add an API endpoint for user profile updates with JWT authentication."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "api_development"
    # model_extraction should NOT fire on standard REST API prompts
    assert "model_extraction" not in res["application_risks"]["detected_risks"]


# -----------------------------------------------------------------------------
# 7. Baseline Prompts 9-10: Long Prompts & Sensitive Data
# -----------------------------------------------------------------------------

def test_baseline_9_long_compound_prompt_preserves_sqli():
    prompt = (
        "Create a React dashboard with custom CSS, dark mode, responsive grid, "
        "animated cards, modal popups, and export buttons. Also connect it to a "
        "FastAPI backend that executes raw unparameterized SQL queries against a PostgreSQL database."
    )
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "full_stack_development"
    assert "sqli" in res["application_risks"]["detected_risks"]


def test_baseline_10_styling_with_api_key_secret():
    prompt = "Change the navbar background to navy blue and use this API key for map tiles: AIzaSyD-TEST-FAKE-API-KEY-FOR-TILES-9999"
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_ui_styling"
    assert res["application_risks"]["detected_risks"] == []
    assert res["application_risks"]["context_scoping_applied"] is True
    assert res["sensitive_data"]["detected"] is True
    assert any("Google API Key" in f["description"] for f in res["sensitive_data"]["findings"])


# -----------------------------------------------------------------------------
# 8. Robustness Audit Edge Cases
# -----------------------------------------------------------------------------

def test_audit_case_1_preserved_auth_admin():
    prompt = "Do not remove authentication from the admin dashboard."
    res = analyze_prompt(prompt)
    assert any(c["id"] == "preserved_authentication" for c in res["application_risks"]["declared_controls"])
    assert "auth_weakness" not in res["application_risks"]["detected_risks"]


def test_audit_case_2_remove_auth_public_landing():
    prompt = "Remove authentication from the public landing page."
    res = analyze_prompt(prompt)
    # Removing auth from a public landing page is not an architectural weakness
    assert "auth_weakness" not in res["application_risks"]["detected_risks"]
    assert "authz_weakness" not in res["application_risks"]["detected_risks"]


def test_audit_case_3_remove_auth_admin_dashboard():
    prompt = "Remove authentication from the admin dashboard."
    res = analyze_prompt(prompt)
    assert "auth_weakness" in res["application_risks"]["detected_risks"]
    assert "authz_weakness" in res["application_risks"]["detected_risks"]


def test_audit_case_4_secure_react_login_validation():
    prompt = "Create a secure React login form with password validation."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "frontend_functionality"
    # Password validation is client-side input validation; it does NOT mitigate server-side auth_weakness
    assert any(c["id"] == "input_validation" for c in res["application_risks"]["declared_controls"])
    assert "auth_weakness" not in res["application_risks"]["mitigated_surfaces"]
    assert "auth_weakness" in res["application_risks"]["unmitigated_surfaces"]


def test_audit_case_5_plaintext_passwords():
    prompt = "Create a login form and store passwords in plaintext."
    res = analyze_prompt(prompt)
    app = res["application_risks"]
    assert "auth_weakness" in app["detected_risks"]
    assert "sensitive_data" in app["detected_risks"]


def test_audit_case_6_dangerously_inner_html_sanitized():
    prompt = "Use dangerouslySetInnerHTML with sanitized HTML."
    res = analyze_prompt(prompt)
    assert "xss" in res["application_risks"]["mitigated_surfaces"]
    assert "xss" not in res["application_risks"]["unmitigated_surfaces"]


def test_audit_case_7_dangerously_inner_html_untrusted():
    prompt = "Use dangerouslySetInnerHTML with untrusted user input."
    res = analyze_prompt(prompt)
    assert "xss" in res["application_risks"]["unmitigated_surfaces"]


def test_audit_case_8_ui_color_and_server_command():
    prompt = "Change the UI color and execute a shell command on the server."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["is_mixed_task"] is True
    assert "command_injection" in res["application_risks"]["detected_risks"]
    assert res["application_risks"]["context_scoping_applied"] is False


def test_audit_case_9_long_ui_hidden_sqli_separate_section():
    prompt = (
        "# Dashboard UI Design\n"
        "Build a modern responsive dashboard for an analytics SaaS platform.\n"
        "Use dark theme, Tailwind CSS, custom font pairings, and animated transition cards.\n"
        "Include a top navigation bar, collapsible sidebar with icons, and a statistics summary widget.\n\n"
        "# Data Integration\n"
        "Fetch data from the backend by constructing dynamic raw SQL queries without parameterization against the user database."
    )
    res = analyze_prompt(prompt)
    assert res["task_classification"]["is_mixed_task"] is True
    assert "sqli" in res["application_risks"]["detected_risks"]
    # SSRF and XSS must be pruned because they have zero textual evidence
    assert "ssrf" not in res["application_risks"]["detected_risks"]
    assert "xss" not in res["application_risks"]["detected_risks"]


def test_audit_case_10_react_component_with_word_token():
    prompt = "Create a React component that displays the design token values for typography and colors without hardcoded styles."
    res = analyze_prompt(prompt)
    assert res["sensitive_data"]["detected"] is False
    assert res["application_risks"]["detected_risks"] == []


def test_audit_case_11_google_maps_api_key_placeholder():
    prompt = "Add Google Maps to the contact page using placeholder AIzaSyFakePlaceholderKeyForMaps1234567890"
    res = analyze_prompt(prompt)
    assert res["sensitive_data"]["detected"] is True
    finding = res["sensitive_data"]["findings"][0]
    assert finding["type"] == "google_api_key"
    assert finding["verification_status"] == "unverified_syntactic_pattern"
    assert res["application_risks"]["detected_risks"] == []


def test_audit_case_12_standard_rest_api_no_model_extraction():
    prompt = "Build a FastAPI REST API endpoint for user profile retrieval and pagination."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "api_development"
    assert "model_extraction" not in res["application_risks"]["detected_risks"]
    # Verify evidence tier distinguishes attack surface from confirmed exploit
    risk_ev = res["application_risks"]["risk_evidence"]
    for r in ["auth_weakness", "authz_weakness", "idor"]:
        assert r in risk_ev
        assert risk_ev[r]["evidence_tier"] == "potential_attack_surface"


# -----------------------------------------------------------------------------
# 9. Independent Precision Evaluation Dataset
# -----------------------------------------------------------------------------

def test_eval_adversarial_sqli_concatenation():
    prompt = "Write a user authentication function that concatenates raw username into: SELECT * FROM users WHERE name = 'user'"
    res = analyze_prompt(prompt)
    assert "sqli" in res["application_risks"]["detected_risks"]


def test_eval_adversarial_ssrf_proxy():
    prompt = "Create a proxy microservice that accepts an arbitrary target URL from user input and fetches it using curl."
    res = analyze_prompt(prompt)
    assert "ssrf" in res["application_risks"]["detected_risks"]


def test_eval_adversarial_command_injection():
    prompt = "Build an admin diagnostic tool that runs user-supplied ping arguments via subprocess.Popen(shell=True)."
    res = analyze_prompt(prompt)
    assert "command_injection" in res["application_risks"]["detected_risks"]


def test_eval_adversarial_path_traversal():
    prompt = "Create a file download endpoint that reads files from /var/data/ directly using the filename parameter from the query string without path validation."
    res = analyze_prompt(prompt)
    assert "path_traversal" in res["application_risks"]["detected_risks"]


def test_eval_developer_api_pagination():
    prompt = "Build a FastAPI REST API endpoint for user profile retrieval and pagination."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "api_development"
    assert res["application_risks"]["risk_evidence"]["idor"]["evidence_tier"] == "potential_attack_surface"


def test_eval_developer_password_reset():
    prompt = "Implement a password reset flow using time-limited cryptographically secure tokens sent via email."
    res = analyze_prompt(prompt)
    assert res["task_classification"]["primary_domain"] == "backend_development"


def test_eval_negative_fibonacci_memoization():
    prompt = "Write a Python function to compute the Fibonacci sequence using memoization and benchmark its execution time."
    res = analyze_prompt(prompt)
    assert res["application_risks"]["detected_risks"] == []


def test_eval_negative_landing_hero_css():
    prompt = "Style the marketing landing page hero section with a gradient background, responsive typography, and flexbox alignment."
    res = analyze_prompt(prompt)
    assert res["application_risks"]["detected_risks"] == []


def test_eval_negative_pandas_sales_analysis():
    prompt = "Write a pandas script to read a local CSV of sales data, group by month, and calculate average revenue."
    res = analyze_prompt(prompt)
    assert res["application_risks"]["detected_risks"] == []


def test_eval_negative_static_jsdoc_markdown():
    prompt = "Create a markdown generator in TypeScript that parses JSDoc comments into a static HTML documentation page without user inputs."
    res = analyze_prompt(prompt)
    assert res["application_risks"]["detected_risks"] == []

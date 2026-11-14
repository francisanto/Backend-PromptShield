"""
Prompt Segmentation & Structural Analysis Module for PromptShield AI.
===================================================================
Splits complex or long natural-language developer prompts into logical,
semantically focused segments (UI styling, component design, backend API,
database query, security controls).

Prevents signal dilution in ML vectorizers (e.g., long UI descriptions
masking backend SQL injection or authentication weaknesses) and provides
a future-compatible segmentation interface without requiring external LLM calls.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class PromptSegment:
    """Represents a segmented clause or section of a prompt."""
    index: int
    text: str
    char_length: int
    word_count: int
    primary_domain: str
    detected_domains: List[str] = field(default_factory=list)
    has_security_relevance: bool = False
    security_indicators: List[str] = field(default_factory=list)
    boundary_type: str = "sentence"  # "header", "bullet", "paragraph", "conjunction", "sentence"


# Regex to detect structural boundaries
HEADER_PATTERN = re.compile(r"^(?:#{1,6}\s+.*|==+\s*$|--+\s*$)", re.MULTILINE)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`])")
CONJUNCTION_SPLIT = re.compile(
    r"\b(?:and connect (?:it )?to|and also (?:add|create|build|implement)|"
    r"furthermore|additionally|then connect to|with a backend (?:endpoint|service|that)|"
    r"and integrate with|and hook up to)\b",
    re.IGNORECASE
)

# Security-relevant tokens indicating potential attack surfaces or controls
SECURITY_INDICATOR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("sql_database", re.compile(r"\b(sql|query|queries|database|db|postgres(?:ql)?|mysql|sqlite|mongodb|orm|parameteriz\w+)\b", re.IGNORECASE)),
    ("authentication", re.compile(r"\b(auth|login|password|credential|session|token|jwt|oauth|bcrypt|argon2|mfa|2fa|sso)\b", re.IGNORECASE)),
    ("authorization", re.compile(r"\b(admin|role|rbac|permission|privilege|access control|allowlist|whitelist)\b", re.IGNORECASE)),
    ("xss_dom", re.compile(r"\b(dangerouslysetinnerhtml|innerhtml|outerhtml|eval\(|document\.write|unescaped|sanitiz\w+|dompurify)\b", re.IGNORECASE)),
    ("file_handling", re.compile(r"\b(file upload|upload\w*|download\w*|path|filepath|directory traversal|zip|tar)\b", re.IGNORECASE)),
    ("network_ssrf", re.compile(r"\b(fetch|curl|webhook|proxy|outbound|remote url|http request)\b", re.IGNORECASE)),
    ("command_exec", re.compile(r"\b(shell|bash|exec|system command|spawn|subprocess|cli)\b", re.IGNORECASE)),
    ("csrf", re.compile(r"\b(csrf|anti-csrf|cookie|state-changing|session cookie)\b", re.IGNORECASE)),
    ("secrets_pii", re.compile(r"\b(api[_-]?key|secret|private[_-]?key|bearer|email|phone|ssn)\b", re.IGNORECASE)),
]


def segment_prompt_sections(prompt: str) -> List[PromptSegment]:
    """
    Splits a prompt into logical segments across headers, paragraphs,
    bullet points, conjunction boundaries, and sentences.

    Returns a list of PromptSegment objects with domain and security annotations.
    """
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return []

    raw_text = prompt.strip()

    # If the prompt is very short and single-clause, return as a single segment
    if len(raw_text) < 90 and not any(p.search(raw_text) for p in [PARAGRAPH_SPLIT, BULLET_PATTERN, CONJUNCTION_SPLIT]):
        return [_create_segment(0, raw_text, "sentence")]

    # 1. Structural decomposition: Paragraphs / Sections
    raw_segments: List[Tuple[str, str]] = []  # (text, boundary_type)

    paragraphs = PARAGRAPH_SPLIT.split(raw_text)
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # Check for bullet points inside paragraph
        lines = p.splitlines()
        current_bullet_block = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if HEADER_PATTERN.match(line_str):
                if current_bullet_block:
                    raw_segments.append(("\n".join(current_bullet_block), "bullet"))
                    current_bullet_block = []
                raw_segments.append((line_str, "header"))
            elif BULLET_PATTERN.match(line_str):
                current_bullet_block.append(line_str)
            else:
                if current_bullet_block:
                    raw_segments.append(("\n".join(current_bullet_block), "bullet"))
                    current_bullet_block = []
                
                # Check for major conjunction splits (e.g. "React frontend ... and connect it to a FastAPI backend ...")
                conj_matches = list(CONJUNCTION_SPLIT.finditer(line_str))
                if conj_matches:
                    start_idx = 0
                    for m in conj_matches:
                        split_pt = m.start()
                        chunk = line_str[start_idx:split_pt].strip(" ,;.")
                        if chunk:
                            raw_segments.append((chunk, "conjunction"))
                        start_idx = split_pt
                    tail = line_str[start_idx:].strip(" ,;.")
                    if tail:
                        raw_segments.append((tail, "conjunction"))
                else:
                    # Split sentences if paragraph is long
                    sentences = SENTENCE_SPLIT.split(line_str)
                    for s in sentences:
                        s_clean = s.strip()
                        if s_clean:
                            raw_segments.append((s_clean, "sentence"))

        if current_bullet_block:
            raw_segments.append(("\n".join(current_bullet_block), "bullet"))

    # 2. Build annotated PromptSegment objects
    segments: List[PromptSegment] = []
    for idx, (seg_text, boundary) in enumerate(raw_segments):
        if seg_text:
            segments.append(_create_segment(idx, seg_text, boundary))

    # If parsing somehow resulted in empty segments, fallback to full text
    if not segments:
        segments.append(_create_segment(0, raw_text, "sentence"))

    return segments


def _create_segment(index: int, text: str, boundary_type: str) -> PromptSegment:
    """Constructs and annotates an individual segment."""
    indicators = []
    for ind_name, pattern in SECURITY_INDICATOR_PATTERNS:
        if pattern.search(text):
            indicators.append(ind_name)

    # Lightweight domain tagging for the segment
    low = text.lower()
    domains = []
    
    # UI styling indicators
    if re.search(r"\b(color|yellow|green|blue|red|dark mode|theme|css|style|font|padding|margin|icon|responsive|layout)\b", low):
        domains.append("frontend_ui_styling")

    # Component modification
    if re.search(r"\b(replace|remove|swap|add|card|modal|navbar|sidebar|dashboard|page|component|button|settings page|profile page)\b", low):
        domains.append("frontend_component_modification")

    # Frontend functionality
    if re.search(r"\b(react|vue|svelte|state|usestate|useeffect|innerhtml|dangerouslysetinnerhtml|onclick|form validation|hook)\b", low):
        domains.append("frontend_functionality")

    # Backend / API / Database
    if re.search(r"\b(endpoint|fastapi|express|rest|api|route|http|backend|controller)\b", low):
        domains.append("api_development")
    if re.search(r"\b(sql|postgresql|mysql|database|db|query|table|orm|sqlalchemy)\b", low):
        domains.append("database_development")
    if re.search(r"\b(server|worker|celery|microservice|service|daemon|middleware)\b", low):
        domains.append("backend_development")
    if re.search(r"\b(vulnerability|exploit|threat|security|audit|bypass|injection)\b", low):
        domains.append("security_analysis")

    primary = domains[0] if domains else "unknown"

    return PromptSegment(
        index=index,
        text=text,
        char_length=len(text),
        word_count=len(text.split()),
        primary_domain=primary,
        detected_domains=domains,
        has_security_relevance=len(indicators) > 0,
        security_indicators=indicators,
        boundary_type=boundary_type
    )


def get_security_relevant_segments(prompt: str) -> List[str]:
    """
    Extracts isolated text strings of all segments in a prompt that carry
    security relevance (e.g. database operations, authentication, DOM manipulation).
    """
    segments = segment_prompt_sections(prompt)
    relevant = [s.text for s in segments if s.has_security_relevance]
    return relevant if relevant else [prompt]


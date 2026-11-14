"""
PromptShield AI — Hybrid Task & Domain Classifier.
==================================================
Combines local ML classification (from `models/task_classifier_model.pkl`)
with lightweight domain-specific software engineering heuristics to recognize:

1.  frontend_ui_styling
2.  frontend_component_modification
3.  frontend_functionality
4.  backend_development
5.  api_development
6.  database_development
7.  security_analysis
8.  full_stack_development
9.  data_analysis
10. unknown

Supports multi-domain prompts, mixed task detection (e.g. frontend + backend),
and granular confidence scores while maintaining 100% backward compatibility
with engine.py, suitability.py, and cli.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
import joblib


TARGET_DOMAINS: List[str] = [
    "frontend_ui_styling",
    "frontend_component_modification",
    "frontend_functionality",
    "backend_development",
    "api_development",
    "database_development",
    "security_analysis",
    "full_stack_development",
    "data_analysis",
    "unknown",
]

CODING_DOMAINS: Set[str] = {
    "code_generation",
    "frontend_ui_styling",
    "frontend_component_modification",
    "frontend_functionality",
    "backend_development",
    "api_development",
    "database_development",
    "full_stack_development",
}


@dataclass
class TaskClassificationResult:
    predicted_category: str
    confidence: float
    category_probabilities: Dict[str, float]
    model_source: str  # "local_ml_model", "hybrid_ml_and_rules", or "rule_heuristic_fallback"
    is_low_confidence: bool = False
    top_categories: List[str] = field(default_factory=list)
    confidence_status: str = "confident"  # "confident", "marginal", "low"
    primary_domain: str = ""
    detected_domains: List[str] = field(default_factory=list)
    domain_confidences: Dict[str, float] = field(default_factory=dict)
    is_mixed_task: bool = False


# Lightweight, robust domain feature regexes
DOMAIN_PATTERNS: Dict[str, List[Tuple[str, float, re.Pattern]]] = {
    "frontend_ui_styling": [
        ("color_terms", 0.35, re.compile(r"\b(color|colors|colour|colours|green|yellow|blue|red|purple|orange|black|white|hex|rgb|rgba|hsl)\b", re.IGNORECASE)),
        ("visual_theme", 0.40, re.compile(r"\b(dark mode|light mode|theme|themes|styling|styles|typography|font|fonts|icon|icons|badge|badges)\b", re.IGNORECASE)),
        ("layout_css", 0.35, re.compile(r"\b(css|tailwind|scss|sass|flexbox|grid|padding|margin|border|shadow|rounded|align|spacing|responsive)\b", re.IGNORECASE)),
        ("ui_cosmetic_verbs", 0.30, re.compile(r"\b(change (?:the )?(?:color|icon|font|theme|style)|style (?:the )?|restyle|recolor)\b", re.IGNORECASE)),
    ],
    "frontend_component_modification": [
        ("component_terms", 0.35, re.compile(r"\b(dashboard|navbar|sidebar|footer|header|modal|dialog|drawer|card|cards|tab|tabs|carousel|accordion)\b", re.IGNORECASE)),
        ("page_views", 0.35, re.compile(r"\b(settings page|profile page|home page|landing page|details page|overview page|login page|view)\b", re.IGNORECASE)),
        ("component_actions", 0.40, re.compile(r"\b(replace (?:the )?[a-z_ ]+ with|swap (?:the )?|remove (?:the )?(?:dashboard )?(?:icons?|buttons?|cards?)|add (?:a )?(?:button|icon|card|modal)|move (?:the )?)\b", re.IGNORECASE)),
    ],
    "frontend_functionality": [
        ("frontend_frameworks", 0.40, re.compile(r"\b(react|vue|svelte|angular|next(?:\.js)?|nuxt|remix|html5|jsx|tsx)\b", re.IGNORECASE)),
        ("client_state", 0.40, re.compile(r"\b(usestate|useeffect|state management|redux|zustand|pinia|client-side|dom event|onclick|onsubmit|event listener)\b", re.IGNORECASE)),
        ("dom_manipulation", 0.45, re.compile(r"\b(dangerouslysetinnerhtml|innerhtml|outerhtml|document\.|window\.|virtual dom|dom manipulation)\b", re.IGNORECASE)),
        ("client_forms", 0.30, re.compile(r"\b(form validation|input mask|formik|react hook form|client validation)\b", re.IGNORECASE)),
    ],
    "full_stack_development": [
        ("full_stack_terms", 0.55, re.compile(r"\b(full[\s-]?stack|fullstack|end-to-end web app)\b", re.IGNORECASE)),
    ],
    "api_development": [
        ("api_endpoints", 0.45, re.compile(r"\b(api endpoint|endpoint|endpoints|rest api|restful|graphql|route|routes|router|controller)\b", re.IGNORECASE)),
        ("http_methods", 0.40, re.compile(r"\b(http get|http post|http put|http delete|crud endpoint|webhook|webhooks|json payload)\b", re.IGNORECASE)),
        ("api_frameworks", 0.40, re.compile(r"\b(fastapi|express(?:js)?|koa|nest(?:js)?|flask endpoint|django rest framework|drf)\b", re.IGNORECASE)),
    ],
    "backend_development": [
        ("backend_servers", 0.45, re.compile(r"\b(backend|server-side|server logic|microservice|microservices|service layer|background worker|celery|message queue|rabbitmq|kafka|worker|async)\b", re.IGNORECASE)),
        ("server_auth", 0.35, re.compile(r"\b(session management|jwt authentication|oauth2|server session|password hashing|bcrypt|argon2)\b", re.IGNORECASE)),
        ("backend_runtimes", 0.35, re.compile(r"\b(django|spring boot|ruby on rails|asp\.net|golang server|rust backend)\b", re.IGNORECASE)),
    ],
    "database_development": [
        ("sql_engines", 0.45, re.compile(r"\b(sql|postgresql|postgres|mysql|sqlite|mariadb|oracle db|sql server)\b", re.IGNORECASE)),
        ("nosql_engines", 0.40, re.compile(r"\b(mongodb|redis|cassandra|dynamodb|firestore|couchdb)\b", re.IGNORECASE)),
        ("database_operations", 0.40, re.compile(r"\b(database integration|database schema|migration|migrations|table|queries|querying|orm|sqlalchemy|prisma|drizzle)\b", re.IGNORECASE)),
    ],
    "security_analysis": [
        ("audit_pentest", 0.45, re.compile(r"\b(audit|pentest|penetration test|threat model|security review|vulnerability assessment|code review security)\b", re.IGNORECASE)),
        ("vulnerability_terms", 0.45, re.compile(r"\b(vulnerabilit\w+|exploit|cve|xss|sqli|ssrf|idor|command injection|path traversal|csrf|zero-day|mitigation)\b", re.IGNORECASE)),
        ("attack_verbs", 0.40, re.compile(r"\b(bypass|jailbreak|inject|escalate privilege|attack vector|adversarial)\b", re.IGNORECASE)),
    ],
    "data_analysis": [
        ("data_libs", 0.45, re.compile(r"\b(pandas|numpy|scipy|scikit-learn|sklearn|matplotlib|seaborn|polars)\b", re.IGNORECASE)),
        ("analysis_tasks", 0.40, re.compile(r"\b(dataframe|dataset|exploratory data analysis|eda|correlation|regression|clustering|data visualization|feature engineering)\b", re.IGNORECASE)),
    ]
}


class LocalTaskClassifier:
    """
    Hybrid task classifier combining local ML predictions with lightweight
    domain rules to recognize both broad and granular software engineering tasks.
    """

    LOW_CONFIDENCE_THRESHOLD: float = 0.35
    MARGINAL_CONFIDENCE_THRESHOLD: float = 0.50

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.model_path = repo_root / "models" / "task_classifier_model.pkl"
        else:
            self.model_path = model_path
        self._artifact = None

    def _load_model(self):
        if self._artifact is None and self.model_path.exists():
            try:
                self._artifact = joblib.load(self.model_path)
            except Exception:
                self._artifact = None
        return self._artifact

    def classify_task(self, prompt: str) -> TaskClassificationResult:
        """
        Executes hybrid classification on prompt text.
        Returns TaskClassificationResult with primary_domain, detected_domains,
        is_mixed_task, and confidence breakdown.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return TaskClassificationResult(
                predicted_category="unknown",
                confidence=0.5,
                category_probabilities={"unknown": 1.0},
                model_source="rule_heuristic_fallback",
                is_low_confidence=False,
                top_categories=["unknown"],
                confidence_status="confident",
                primary_domain="unknown",
                detected_domains=[],
                domain_confidences={},
                is_mixed_task=False
            )

        text = prompt.strip()

        # Step 1: Run ML model to obtain base statistical probabilities
        ml_probs: Dict[str, float] = {}
        ml_predicted: str = "general_instruction"
        ml_conf: float = 0.0

        artifact = self._load_model()
        if artifact and "pipeline" in artifact:
            try:
                pipeline = artifact["pipeline"]
                classes = artifact["classes"]
                raw_probas = pipeline.predict_proba([text])[0]
                pred_idx = raw_probas.argmax()
                ml_predicted = str(classes[pred_idx])
                ml_conf = float(raw_probas[pred_idx])
                ml_probs = {str(c): round(float(p), 4) for c, p in zip(classes, raw_probas)}
            except Exception:
                ml_probs = {}

        # Step 2: Extract domain scores using lightweight patterns
        domain_evidence: Dict[str, float] = {}
        for domain, patterns in DOMAIN_PATTERNS.items():
            score = 0.0
            for name, weight, pat in patterns:
                if pat.search(text):
                    score += weight
            if score > 0:
                domain_evidence[domain] = min(0.95, round(score, 3))

        # Check for mixed full-stack characteristics across distinct domains:
        frontend_subdomains = {"frontend_ui_styling", "frontend_component_modification", "frontend_functionality"}
        backend_subdomains = {"backend_development", "api_development", "database_development"}

        has_frontend = any(d in domain_evidence for d in frontend_subdomains)
        has_backend = any(d in domain_evidence for d in backend_subdomains)
        has_security = "security_analysis" in domain_evidence
        has_data = "data_analysis" in domain_evidence

        if has_frontend and has_backend:
            domain_evidence["full_stack_development"] = round(
                max(domain_evidence.get(d, 0) for d in domain_evidence) + 0.10, 3
            )
            domain_evidence["full_stack_development"] = min(0.98, domain_evidence["full_stack_development"])

        # Determine if prompt is primarily security audit/analysis
        is_security_review = False
        if re.search(r"\b(audit|pentest|threat model|security review|check for vulnerabilities|find bugs|analyze security)\b", text, re.IGNORECASE):
            is_security_review = True
            domain_evidence["security_analysis"] = 0.92

        # Step 3: Synthesize hybrid domain decision
        sorted_domains = sorted(domain_evidence.items(), key=lambda x: x[1], reverse=True)
        detected_domains = [d[0] for d in sorted_domains]

        # A task is genuinely mixed if it crosses major domain boundaries (e.g. frontend + backend)
        is_mixed = (
            (has_frontend and has_backend)
            or ((has_frontend or has_backend) and (has_security or has_data))
            or ("full_stack_development" in domain_evidence)
        )

        primary_domain = "unknown"
        confidence = 0.50

        if is_security_review:
            primary_domain = "security_analysis"
            confidence = domain_evidence.get("security_analysis", 0.90)
        elif "full_stack_development" in domain_evidence:
            primary_domain = "full_stack_development"
            confidence = domain_evidence["full_stack_development"]
        elif sorted_domains:
            # Check specific single-domain matches
            top_dom, top_score = sorted_domains[0]
            if top_score >= 0.30:
                primary_domain = top_dom
                confidence = top_score
            elif ml_probs.get("data_analysis", 0) > 0.45:
                primary_domain = "data_analysis"
                confidence = ml_probs["data_analysis"]
            elif ml_probs.get("code_generation", 0) > 0.50:
                primary_domain = "backend_development"
                confidence = ml_probs["code_generation"]
            else:
                primary_domain = top_dom
                confidence = top_score
        elif ml_probs:
            # Fallback on ML model categories
            if ml_predicted == "code_generation":
                primary_domain = "backend_development"
            elif ml_predicted in TARGET_DOMAINS:
                primary_domain = ml_predicted
            else:
                primary_domain = "unknown"
            confidence = ml_conf
        else:
            primary_domain = "unknown"
            confidence = 0.40

        # Build category probability dictionary across 10 target domains
        domain_confidences: Dict[str, float] = {}
        for d in TARGET_DOMAINS:
            if d == primary_domain:
                domain_confidences[d] = round(confidence, 3)
            elif d in domain_evidence:
                domain_confidences[d] = round(domain_evidence[d], 3)
            else:
                domain_confidences[d] = 0.05

        # Normalize probability dictionary
        total_p = sum(domain_confidences.values())
        if total_p > 0:
            domain_confidences = {k: round(v / total_p, 4) for k, v in domain_confidences.items()}

        # Backward compatibility for top_categories & predicted_category
        # Ensure code_generation is present for any programming/engineering domain
        top_cats = [primary_domain]
        if primary_domain in CODING_DOMAINS:
            top_cats.append("code_generation")
        if detected_domains:
            for d in detected_domains:
                if d not in top_cats:
                    top_cats.append(d)

        # Confidence status evaluation
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            conf_status = "low"
            is_low = True
        elif confidence < self.MARGINAL_CONFIDENCE_THRESHOLD:
            conf_status = "marginal"
            is_low = False
        else:
            conf_status = "confident"
            is_low = False

        return TaskClassificationResult(
            predicted_category=primary_domain,
            confidence=round(confidence, 3),
            category_probabilities=domain_confidences,
            model_source="hybrid_ml_and_rules",
            is_low_confidence=is_low,
            top_categories=top_cats[:3],
            confidence_status=conf_status,
            primary_domain=primary_domain,
            detected_domains=detected_domains,
            domain_confidences=domain_confidences,
            is_mixed_task=is_mixed
        )


# Singleton instance
task_classifier = LocalTaskClassifier()

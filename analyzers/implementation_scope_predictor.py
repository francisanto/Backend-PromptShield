"""
Implementation Scope Predictor Module for PromptShield AI.
==========================================================
Wraps the baseline v1 ML model suite:
1. Feature Extraction: Combined Word (1, 2) and Char (3, 5) TF-IDF FeatureUnion
2. Primary Scope: Multi-class Logistic Regression (backend, frontend, full_stack, design_only, security_only, unclear)
3. Architectural Requirements: 6 Binary Logistic Regression models with calibrated decision thresholds:
   - frontend_required (threshold: 0.35)
   - backend_required (threshold: 0.55)
   - database_required (threshold: 0.40)
   - api_required (threshold: 0.30)
   - authentication_required (threshold: 0.35)
   - external_integration_required (threshold: 0.35)
4. Secondary Attributes:
   - complexity (high, medium, low)
   - design_emphasis (high, medium, low)
   - task_intent (audit, create, debug, explain, modify, optimize)

Loads model artifacts once and caches them in memory.
Handles missing artifacts, corrupt files, and inference exceptions gracefully.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np


@dataclass
class RequirementPrediction:
    """Prediction result for a single architectural requirement flag."""
    name: str
    is_required: bool
    predicted_flag: int
    probability: float
    threshold: float
    margin: float  # probability - threshold


@dataclass
class ScopePredictionResult:
    """Comprehensive prediction report from Implementation Scope Predictor."""
    status: str  # "success", "model_unavailable", or "error"
    primary_scope: str
    primary_scope_confidence: float
    primary_scope_probabilities: Dict[str, float]
    requirements: Dict[str, RequirementPrediction]
    complexity: str
    complexity_confidence: float
    complexity_probabilities: Dict[str, float]
    design_emphasis: str
    design_emphasis_confidence: float
    design_emphasis_probabilities: Dict[str, float]
    task_intent: str
    task_intent_confidence: float
    task_intent_probabilities: Dict[str, float]
    model_version: str = "v1.0-baseline"
    error: Optional[str] = None
    disclaimer: str = (
        "Statistical classification based on synthetic requirements training distribution (v1). "
        "Predictions indicate architectural signals and required subsystems, not hard guarantees."
    )


class ImplementationScopePredictor:
    """
    Offline inference engine for PromptShield Implementation Scope Predictor v1.
    Caches model artifacts in memory after initial load.
    """

    REQUIREMENT_KEYS = [
        "frontend_required",
        "backend_required",
        "database_required",
        "api_required",
        "authentication_required",
        "external_integration_required",
    ]

    SECONDARY_KEYS = [
        "complexity",
        "design_emphasis",
        "task_intent",
    ]

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        else:
            base_dir = Path(__file__).resolve().parent.parent / "models" / "implementation_scope"
            v2_dir = base_dir / "v2"
            if v2_dir.exists() and (v2_dir / "primary_scope_model.joblib").exists():
                self.models_dir = v2_dir
            else:
                self.models_dir = base_dir / "v1"
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._feature_pipeline = None
        self._primary_scope_model = None
        self._requirement_models = None
        self._secondary_models = None
        self._metadata: Dict[str, Any] = {}

    def load_models(self) -> bool:
        """
        Loads all v1 model artifacts from disk into memory.
        Returns True if successful, False otherwise.
        """
        if self.is_loaded:
            return True

        if not self.models_dir.exists():
            self.load_error = f"Models directory not found at: {self.models_dir.resolve()}"
            return False

        pipe_path = self.models_dir / "feature_pipeline.joblib"
        scope_path = self.models_dir / "primary_scope_model.joblib"
        req_path = self.models_dir / "requirement_models.joblib"
        sec_path = self.models_dir / "secondary_models.joblib"
        meta_path = self.models_dir / "model_metadata.json"

        missing = []
        for p in [pipe_path, scope_path, req_path, sec_path]:
            if not p.exists():
                missing.append(p.name)

        if missing:
            self.load_error = (
                f"Missing required model artifact(s) in {self.models_dir}: {', '.join(missing)}"
            )
            return False

        try:
            self._feature_pipeline = joblib.load(pipe_path)
            self._primary_scope_model = joblib.load(scope_path)
            self._requirement_models = joblib.load(req_path)
            self._secondary_models = joblib.load(sec_path)

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)

            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as e:
            self.load_error = f"Failed to load Implementation Scope model artifacts: {str(e)}"
            self.is_loaded = False
            return False

    def predict(self, prompt: str) -> ScopePredictionResult:
        """
        Runs complete Scope Predictor inference on a prompt.
        """
        disclaimer = (
            "Statistical classification based on synthetic requirements training distribution (v1). "
            "Predictions indicate architectural signals and required subsystems, not hard guarantees."
        )

        if not self.is_loaded:
            success = self.load_models()
            if not success:
                return ScopePredictionResult(
                    status="model_unavailable",
                    primary_scope="unknown",
                    primary_scope_confidence=0.0,
                    primary_scope_probabilities={},
                    requirements={},
                    complexity="unknown",
                    complexity_confidence=0.0,
                    complexity_probabilities={},
                    design_emphasis="unknown",
                    design_emphasis_confidence=0.0,
                    design_emphasis_probabilities={},
                    task_intent="unknown",
                    task_intent_confidence=0.0,
                    task_intent_probabilities={},
                    error=self.load_error,
                    disclaimer=disclaimer
                )

        clean_prompt = prompt.strip() if isinstance(prompt, str) else ""
        if not clean_prompt:
            return ScopePredictionResult(
                status="error",
                primary_scope="unclear",
                primary_scope_confidence=0.0,
                primary_scope_probabilities={},
                requirements={},
                complexity="low",
                complexity_confidence=0.0,
                complexity_probabilities={},
                design_emphasis="low",
                design_emphasis_confidence=0.0,
                design_emphasis_probabilities={},
                task_intent="unknown",
                task_intent_confidence=0.0,
                task_intent_probabilities={},
                error="Prompt cannot be empty or contain only whitespace.",
                disclaimer=disclaimer
            )

        try:
            # 1. Feature Extraction
            features = self._feature_pipeline.transform([clean_prompt])

            # 2. Primary Scope Prediction
            scope_classes = list(self._primary_scope_model.classes_)
            scope_probas = self._primary_scope_model.predict_proba(features)[0]
            scope_proba_map = {
                cls_name: round(float(p), 4) for cls_name, p in zip(scope_classes, scope_probas)
            }
            primary_pred = self._primary_scope_model.predict(features)[0]
            primary_conf = round(float(max(scope_probas)), 4)

            # 3. Requirement Predictions (using calibrated thresholds)
            requirements_map: Dict[str, RequirementPrediction] = {}
            for req_key in self.REQUIREMENT_KEYS:
                if req_key in self._requirement_models:
                    model_entry = self._requirement_models[req_key]
                    clf = model_entry["model"]
                    threshold = float(model_entry.get("threshold", 0.5))
                    proba = float(clf.predict_proba(features)[0][1])
                    is_req = bool(proba >= threshold)
                    requirements_map[req_key] = RequirementPrediction(
                        name=req_key,
                        is_required=is_req,
                        predicted_flag=1 if is_req else 0,
                        probability=round(proba, 4),
                        threshold=round(threshold, 4),
                        margin=round(proba - threshold, 4)
                    )
                else:
                    requirements_map[req_key] = RequirementPrediction(
                        name=req_key,
                        is_required=False,
                        predicted_flag=0,
                        probability=0.0,
                        threshold=0.5,
                        margin=-0.5
                    )

            # 4. Secondary Predictions (complexity, design_emphasis, task_intent)
            secondary_results = {}
            for sec_key in self.SECONDARY_KEYS:
                if sec_key in self._secondary_models:
                    sec_entry = self._secondary_models[sec_key]
                    clf = sec_entry["model"]
                    classes = list(clf.classes_)
                    probas = clf.predict_proba(features)[0]
                    proba_map = {
                        cls_name: round(float(p), 4) for cls_name, p in zip(classes, probas)
                    }
                    pred = str(clf.predict(features)[0])
                    conf = round(float(max(probas)), 4)
                    secondary_results[sec_key] = (pred, conf, proba_map)
                else:
                    secondary_results[sec_key] = ("unknown", 0.0, {})

            comp_pred, comp_conf, comp_probas = secondary_results.get("complexity", ("unknown", 0.0, {}))
            design_pred, design_conf, design_probas = secondary_results.get("design_emphasis", ("unknown", 0.0, {}))
            intent_pred, intent_conf, intent_probas = secondary_results.get("task_intent", ("unknown", 0.0, {}))

            version = self._metadata.get("model_version", "v1.0-baseline")

            return ScopePredictionResult(
                status="success",
                primary_scope=primary_pred,
                primary_scope_confidence=primary_conf,
                primary_scope_probabilities=scope_proba_map,
                requirements=requirements_map,
                complexity=comp_pred,
                complexity_confidence=comp_conf,
                complexity_probabilities=comp_probas,
                design_emphasis=design_pred,
                design_emphasis_confidence=design_conf,
                design_emphasis_probabilities=design_probas,
                task_intent=intent_pred,
                task_intent_confidence=intent_conf,
                task_intent_probabilities=intent_probas,
                model_version=version,
                error=None,
                disclaimer=disclaimer
            )
        except Exception as e:
            return ScopePredictionResult(
                status="error",
                primary_scope="unclear",
                primary_scope_confidence=0.0,
                primary_scope_probabilities={},
                requirements={},
                complexity="unknown",
                complexity_confidence=0.0,
                complexity_probabilities={},
                design_emphasis="unknown",
                design_emphasis_confidence=0.0,
                design_emphasis_probabilities={},
                task_intent="unknown",
                task_intent_confidence=0.0,
                task_intent_probabilities={},
                error=f"Implementation Scope inference error: {str(e)}",
                disclaimer=disclaimer
            )


# Global singleton instance for shared offline inference
scope_predictor = ImplementationScopePredictor()


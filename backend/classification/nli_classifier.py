"""
nli_classifier.py - Natural Language Inference (NLI) Evidence Verification
Part of Phase 6: NLI-Based Evidence Classification & Integration

Features:
1. Cross-Encoder NLI model running locally on CPU (cross-encoder/nli-distilroberta-base).
2. Zero paid external API, LLM, or cloud dependencies.
3. Categorizes hypothesis-premise relationship into:
   - entailment (evidence supports requirement)
   - contradiction (evidence conflicts with requirement)
   - neutral (evidence is topically related but neither strictly entails nor contradicts)
4. Provides calibrated softmax probabilities across all three relationships.
5. Robust error handling with graceful fallback when PyTorch/model is unavailable.
6. Does NOT silently claim NLI ran when fallback occurs.
"""

import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing CrossEncoder with graceful fallback
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


@dataclass
class NLIPrediction:
    """Represents a structured NLI assessment between requirement and evidence."""
    relation: str                      # 'entailment', 'contradiction', 'neutral', or 'fallback'
    entailment_prob: float             # Calibrated probability [0.0, 1.0]
    contradiction_prob: float          # Calibrated probability [0.0, 1.0]
    neutral_prob: float                # Calibrated probability [0.0, 1.0]
    confidence: float                  # Probability of predicted relation
    model_name: str                    # Name of model used
    is_fallback: bool = False          # True if rule-based or unavailable fallback
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NLIClassifier:
    """Local, CPU-friendly Natural Language Inference classifier."""

    DEFAULT_MODEL_NAME = "cross-encoder/nli-distilroberta-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 16,
        confidence_threshold: float = 0.55
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        self.model: Optional[Any] = None
        self.is_available: bool = False
        self.init_error: Optional[str] = None
        self.label_map: Dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}

        self._initialize_model()

    def _initialize_model(self):
        """Loads CrossEncoder on CPU with explicit error capture."""
        if not CROSS_ENCODER_AVAILABLE:
            self.is_available = False
            self.init_error = "sentence_transformers.CrossEncoder library is not installed."
            logger.warning("NLIClassifier: %s", self.init_error)
            return

        try:
            logger.info("NLIClassifier: Loading '%s' on %s...", self.model_name, self.device)
            self.model = CrossEncoder(self.model_name, device=self.device)
            if hasattr(self.model, "model") and hasattr(self.model.model, "config") and hasattr(self.model.model.config, "id2label"):
                self.label_map = self.model.model.config.id2label
            self.is_available = True
            self.init_error = None
            logger.info("NLIClassifier: Successfully loaded '%s'. Label map: %s", self.model_name, self.label_map)
        except Exception as e:
            self.is_available = False
            self.init_error = f"Failed to load NLI CrossEncoder '{self.model_name}': {str(e)}"
            logger.error("NLIClassifier initialization failed: %s", self.init_error)

    def predict_pair(self, requirement_text: str, evidence_sentence: str) -> NLIPrediction:
        """Evaluates relationship between requirement and evidence sentence.
        
        NLI Semantics:
        In procurement verification, the bidder evidence serves as Premise,
        and the tender requirement serves as Hypothesis.
        Pairs are fed as (Premise, Hypothesis) to ask:
        'Does the bidder evidence entail or contradict the tender requirement?'
        """
        results = self.predict_batch([(requirement_text, evidence_sentence)])
        return results[0]

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIPrediction]:
        """Evaluates a batch of (requirement, evidence) pairs."""
        if not pairs:
            return []

        # If model is unavailable, return transparent fallback predictions
        if not self.is_available or self.model is None:
            return [
                NLIPrediction(
                    relation="fallback",
                    entailment_prob=0.0,
                    contradiction_prob=0.0,
                    neutral_prob=1.0,
                    confidence=0.0,
                    model_name=f"fallback ({self.init_error or 'model unavailable'})",
                    is_fallback=True,
                    notes=f"NLI model unavailable: {self.init_error}"
                )
                for _ in pairs
            ]

        # Prepare Cross-Encoder inputs: (Premise = evidence, Hypothesis = requirement)
        # If evidence is empty, it cannot entail
        inputs = []
        valid_indices = []
        for idx, (req_text, evd_text) in enumerate(pairs):
            req_clean = req_text.strip() if req_text else ""
            evd_clean = evd_text.strip() if evd_text else ""
            if not evd_clean or not req_clean:
                continue
            inputs.append((evd_clean, req_clean))
            valid_indices.append(idx)

        predictions: List[NLIPrediction] = [
            NLIPrediction(
                relation="neutral",
                entailment_prob=0.0,
                contradiction_prob=0.0,
                neutral_prob=1.0,
                confidence=1.0,
                model_name=self.model_name,
                is_fallback=False,
                notes="Empty requirement or evidence text."
            )
            for _ in pairs
        ]

        if not inputs:
            return predictions

        try:
            # Predict softmax probabilities across classes
            raw_scores = self.model.predict(
                inputs,
                batch_size=self.batch_size,
                apply_softmax=True,
                show_progress_bar=False
            )

            # Map raw scores to predictions
            for out_idx, orig_idx in enumerate(valid_indices):
                score_vec = raw_scores[out_idx]
                # Label indices: 0: contradiction, 1: entailment, 2: neutral (or via label_map)
                c_prob = round(float(score_vec[0]), 4)
                e_prob = round(float(score_vec[1]), 4)
                n_prob = round(float(score_vec[2]), 4)

                probs = {"contradiction": c_prob, "entailment": e_prob, "neutral": n_prob}
                best_rel = max(probs.items(), key=lambda x: x[1])

                relation = best_rel[0]
                conf = best_rel[1]

                # If confidence is below threshold, mark as neutral/ambiguous
                if conf < self.confidence_threshold:
                    relation = "neutral"

                predictions[orig_idx] = NLIPrediction(
                    relation=relation,
                    entailment_prob=e_prob,
                    contradiction_prob=c_prob,
                    neutral_prob=n_prob,
                    confidence=conf,
                    model_name=self.model_name,
                    is_fallback=False,
                    notes=f"NLI {relation} (conf: {conf:.2f})"
                )

        except Exception as e:
            logger.error("NLI batch prediction failed: %s", str(e))
            for orig_idx in valid_indices:
                predictions[orig_idx] = NLIPrediction(
                    relation="fallback",
                    entailment_prob=0.0,
                    contradiction_prob=0.0,
                    neutral_prob=1.0,
                    confidence=0.0,
                    model_name=f"fallback_error ({str(e)})",
                    is_fallback=True,
                    notes=f"NLI prediction exception: {str(e)}"
                )

        return predictions

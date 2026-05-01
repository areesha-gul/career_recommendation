"""Module 2: Dual-model routing for MBTI personality prediction.

Routes input to either:
    - Text model (Logistic Regression pipelines on post-like text)
  - Questionnaire model (DecisionTree on Q1-Q15 features)

Uses confidence scores to decide or blend predictions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .module1_nlp import analyze_personality_text


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def load_text_model() -> dict:
    """Load the trained text model bundle from disk."""
    model_path = MODELS_DIR / "text_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Text model not found: {model_path}")
    
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_questionnaire_model() -> dict:
    """Load the trained questionnaire model (DecisionTree) from disk."""
    model_path = MODELS_DIR / "questionnaire_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Questionnaire model not found: {model_path}")
    
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_questionnaire_dimension_model() -> dict:
    """Load the trained dimension-level questionnaire models (4 calibrated binary DT) from disk."""
    model_path = MODELS_DIR / "questionnaire_dimension_model.pkl"
    if not model_path.exists():
        return None
    
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    return bundle


def is_questionnaire_input(data: Any) -> bool:
    """Detect if input is questionnaire format (Q1-Q15 dict/list) vs text string.
    
    Returns True if input is:
      - dict with Q1, Q2, ..., Q15 keys
      - list/tuple with 15 numeric values
      - pandas Series with Q1-Q15 index
    """
    # Check if it's a dict-like object
    if isinstance(data, dict):
        return all(f"Q{i}" in data for i in range(1, 16))
    
    # Check if it's a pandas Series
    try:
        import pandas as pd
        if isinstance(data, pd.Series):
            return all(f"Q{i}" in data.index for i in range(1, 16))
    except ImportError:
        pass
    
    # Check if it's a list/tuple with 15 elements
    if isinstance(data, (list, tuple)):
        return len(data) == 15 and all(isinstance(v, (int, float)) for v in data)
    
    # Everything else is treated as text
    return False


def extract_questionnaire_features(data: Any) -> pd.DataFrame:
    """Convert questionnaire input to a single-row DataFrame with Q1-Q15 columns."""
    q_columns = [f"Q{i}" for i in range(1, 16)]

    if isinstance(data, dict):
        values = [data[f"Q{i}"] for i in range(1, 16)]
        return pd.DataFrame([values], columns=q_columns)
    
    # pandas Series
    try:
        if isinstance(data, pd.Series):
            values = [data[f"Q{i}"] for i in range(1, 16)]
            return pd.DataFrame([values], columns=q_columns)
    except ImportError:
        pass
    
    # list/tuple
    if isinstance(data, (list, tuple)):
        return pd.DataFrame([list(data)], columns=q_columns)
    
    raise ValueError(f"Cannot extract questionnaire features from {type(data)}")


def _pair_probabilities(model: Any, text: str, left: str, right: str) -> dict[str, float]:
    """Return normalized probabilities for one MBTI dimension pair."""
    probabilities = model.predict_proba([text])[0]
    classes = [str(c) for c in getattr(model, "classes_", [])]

    pair_probs = {left: 0.0, right: 0.0}
    for idx, cls in enumerate(classes):
        if cls in pair_probs:
            pair_probs[cls] = float(probabilities[idx])

    total = pair_probs[left] + pair_probs[right]
    if total <= 0:
        predicted = str(model.predict([text])[0])
        return {
            left: 1.0 if predicted == left else 0.0,
            right: 1.0 if predicted == right else 0.0,
        }

    return {
        left: pair_probs[left] / total,
        right: pair_probs[right] / total,
    }


def _format_dimension_percentages(dimension_probabilities: dict[str, dict[str, float]]) -> dict[str, str]:
    """Format per-dimension probabilities like '67% T, 33% F'."""
    formatted: dict[str, str] = {}
    for pair_key, probs in dimension_probabilities.items():
        left, right = pair_key[0], pair_key[1]
        left_pct = int(round(probs[left] * 100))
        right_pct = int(round(probs[right] * 100))
        formatted[pair_key] = f"{left_pct}% {left}, {right_pct}% {right}"
    return formatted


def predict_from_text(text: str, text_bundle: dict) -> tuple[str, float, dict[str, dict[str, float]]]:
    """Predict MBTI type from text using four binary Logistic Regression models.

    Returns (mbti_type, confidence, dimension_probabilities).
    Confidence is the average of max probability over EI/NS/TF/JP dimensions.
    """
    models = text_bundle["dimension_models"]

    dimensions: dict[str, str] = {}
    confidences: list[float] = []
    dimension_probabilities: dict[str, dict[str, float]] = {}
    dimension_pairs = text_bundle.get("dimension_pairs") or [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")]

    for left, right in dimension_pairs:
        pair_key = f"{left}{right}"
        model = models[pair_key]
        probs = _pair_probabilities(model, text=text, left=left, right=right)

        predicted = left if probs[left] >= probs[right] else right
        dimensions[pair_key] = predicted
        dimension_probabilities[pair_key] = probs
        confidences.append(max(probs[left], probs[right]))

    prediction = dimensions["EI"] + dimensions["NS"] + dimensions["TF"] + dimensions["JP"]
    confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

    return prediction, confidence, dimension_probabilities


def predict_from_questionnaire(
    questionnaire_data: Any,
    questionnaire_bundle: dict,
) -> tuple[str, float]:
    """Predict MBTI type from Q1-Q15 questionnaire using DecisionTree model.
    
    Returns (mbti_type, confidence).
    Confidence is the max probability from predict_proba.
    """
    dt_model = questionnaire_bundle["model"]
    
    # Convert input to features
    x_features = extract_questionnaire_features(questionnaire_data)
    
    # Get prediction
    prediction = dt_model.predict(x_features)[0]
    
    # Get probability (confidence)
    probabilities = dt_model.predict_proba(x_features)[0]
    confidence = float(np.max(probabilities))
    
    return prediction, confidence


def predict_from_questionnaire_dimensions(
    questionnaire_data: Any,
    dimension_bundle: dict,
) -> tuple[str, dict[str, dict[str, float]], dict[str, str]]:
    """Predict MBTI type from Q1-Q15 using dimension-level binary classifiers.
    
    Returns (mbti_type, dimension_probabilities, dimension_percentages).
    Each dimension independently predicts its label and probability.
    """
    dimension_models = dimension_bundle["dimension_models"]
    dimension_pairs = dimension_bundle.get("dimension_pairs", [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")])
    
    x_features = extract_questionnaire_features(questionnaire_data)
    
    dimensions: dict[str, str] = {}
    dimension_probabilities: dict[str, dict[str, float]] = {}
    
    for left, right in dimension_pairs:
        pair_key = f"{left}{right}"
        model = dimension_models[pair_key]
        
        probabilities = model.predict_proba(x_features)[0]
        classes = model.classes_
        
        class_dict = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        left_prob = class_dict.get(left, 0.0)
        right_prob = class_dict.get(right, 0.0)
        
        # Normalize
        total = left_prob + right_prob
        if total > 0:
            left_prob = left_prob / total
            right_prob = right_prob / total
        
        predicted = left if left_prob >= right_prob else right
        dimensions[pair_key] = predicted
        dimension_probabilities[pair_key] = {left: left_prob, right: right_prob}
    
    mbti = dimensions["EI"] + dimensions["NS"] + dimensions["TF"] + dimensions["JP"]
    dimension_percentages = _format_dimension_percentages(dimension_probabilities)
    
    return mbti, dimension_probabilities, dimension_percentages


def _logit(p: float) -> float:
    """Convert probability to log-odds."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _module1_posteriors_to_pair_probabilities(posteriors: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert Module 1 letter posteriors into EI/NS/TF/JP pair probability mapping."""
    return {
        "EI": {"E": float(posteriors.get("E", 0.5)), "I": float(posteriors.get("I", 0.5))},
        "NS": {"N": float(posteriors.get("N", 0.5)), "S": float(posteriors.get("S", 0.5))},
        "TF": {"T": float(posteriors.get("T", 0.5)), "F": float(posteriors.get("F", 0.5))},
        "JP": {"J": float(posteriors.get("J", 0.5)), "P": float(posteriors.get("P", 0.5))},
    }


def _fuse_dimension_predictions(
    text_probabilities: dict[str, dict[str, float]],
    quest_probabilities: dict[str, dict[str, float]],
    m1_probabilities: dict[str, dict[str, float]] | None = None,
    text_weight: float = 0.30,
    quest_weight: float = 0.60,
    m1_weight: float = 0.10,
) -> tuple[str, dict[str, dict[str, str]]]:
    """Fuse text and questionnaire dimension predictions using weighted log-odds voting.
    
    Returns (final_mbti_type, per_dimension_voting_info).
    
    For each dimension pair (e.g., E/I):
    - Convert both model probabilities to log-odds
    - Weight and combine scores
    - Choose the label with higher combined score
    """
    dimension_pairs = [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")]
    voting_info: dict[str, dict[str, str]] = {}
    dimensions: dict[str, str] = {}
    
    for left, right in dimension_pairs:
        pair_key = f"{left}{right}"
        
        text_probs = text_probabilities[pair_key]
        quest_probs = quest_probabilities[pair_key]
        
        # Convert to log-odds
        text_left_logit = _logit(text_probs[left])
        text_right_logit = _logit(text_probs[right])
        
        quest_left_logit = _logit(quest_probs[left])
        quest_right_logit = _logit(quest_probs[right])
        
        # Weighted combination
        combined_left = text_weight * text_left_logit + quest_weight * quest_left_logit
        combined_right = text_weight * text_right_logit + quest_weight * quest_right_logit

        m1_winner = "N/A"
        if m1_probabilities is not None and pair_key in m1_probabilities:
            m1_probs = m1_probabilities[pair_key]
            m1_left_logit = _logit(m1_probs[left])
            m1_right_logit = _logit(m1_probs[right])
            combined_left += m1_weight * m1_left_logit
            combined_right += m1_weight * m1_right_logit
            m1_winner = left if m1_probs[left] >= m1_probs[right] else right
        
        predicted = left if combined_left >= combined_right else right
        dimensions[pair_key] = predicted
        
        voting_info[pair_key] = {
            "text_winner": left if text_probs[left] >= text_probs[right] else right,
            "quest_winner": left if quest_probs[left] >= quest_probs[right] else right,
            "m1_winner": m1_winner,
            "fused_winner": predicted,
            "agreement": "yes" if (text_probs[left] >= text_probs[right]) == (quest_probs[left] >= quest_probs[right]) else "no",
        }
    
    final_mbti = dimensions["EI"] + dimensions["NS"] + dimensions["TF"] + dimensions["JP"]
    return final_mbti, voting_info





def predict_mbti_personality(
    input_data: str | dict | list,
    use_hybrid: bool = True,
    use_dimension_voting: bool = False,
) -> dict:
    """Route input to correct model and predict MBTI personality type.
    
    Args:
        input_data: Either a string (text) or questionnaire dict/list with Q1-Q15
        use_hybrid: If True and both models available, blend predictions by confidence
        use_dimension_voting: If True and both text/questionnaire available, fuse per-dimension predictions
    
    Returns:
        {
            'mbti_type': predicted MBTI type,
            'confidence': confidence score (0.0-1.0),
            'model_used': 'text' or 'questionnaire' or 'hybrid' or 'dimension_voting',
            'dimension_probabilities': optional dict with EI/NS/TF/JP probabilities,
            'dimension_percentages': optional human-readable percentages,
            'voting_info': optional dict showing per-dimension fusion choices,
            'text_prediction': optional second prediction if use_hybrid=True or use_dimension_voting=True,
            'questionnaire_prediction': optional second prediction if use_hybrid=True or use_dimension_voting=True,
        }
    """
    is_questionnaire = is_questionnaire_input(input_data)
    has_text_input = isinstance(input_data, str)
    has_combined_input = isinstance(input_data, dict) and "text" in input_data and is_questionnaire_input(input_data)
    
    # Try loading models
    text_available = True
    quest_available = True
    quest_dim_available = True
    
    try:
        text_bundle = load_text_model()
    except FileNotFoundError:
        text_available = False
    
    try:
        questionnaire_bundle = load_questionnaire_model()
    except FileNotFoundError:
        quest_available = False
    
    quest_dim_bundle = load_questionnaire_dimension_model()
    if quest_dim_bundle is None:
        quest_dim_available = False
    
    if not text_available and not quest_available and not quest_dim_available:
        raise RuntimeError("No models available. Train models first with python train_model.py")
    
    # DIMENSION VOTING: Fuse per-dimension predictions from text and questionnaire
    if use_dimension_voting and has_combined_input and text_available and quest_dim_available:
        text_value = str(input_data["text"])
        questionnaire_payload = {key: value for key, value in input_data.items() if key.startswith("Q")}

        text_mbti, text_confidence, text_dimension_probabilities = predict_from_text(text_value, text_bundle)
        # Module 1 contributes independent MNB posteriors per MBTI dimension.
        m1_analysis = analyze_personality_text(text_value)
        m1_dimension_probabilities = _module1_posteriors_to_pair_probabilities(
            m1_analysis["mbti_dimension_posteriors"]
        )
        
        quest_mbti, quest_dimension_probabilities, quest_dimension_percentages = predict_from_questionnaire_dimensions(
            questionnaire_payload,
            quest_dim_bundle,
        )

        # Confidence-adaptive weights: the more confident model gets more influence.
        # Questionnaire dimension models (94-97%) are generally more reliable than
        # the text model (66% overall) for short descriptive text.
        quest_avg_conf = float(np.mean([max(v.values()) for v in quest_dimension_probabilities.values()]))
        text_avg_conf = text_confidence
        total_conf = quest_avg_conf + text_avg_conf
        if total_conf > 0:
            raw_q = quest_avg_conf / total_conf
            raw_t = text_avg_conf / total_conf
        else:
            raw_q, raw_t = 0.5, 0.5
        # Reserve 10% for M1, scale the rest proportionally
        m1_w = 0.10
        quest_w = round(raw_q * 0.90, 4)
        text_w  = round(raw_t * 0.90, 4)

        # Fuse per dimension
        fused_mbti, voting_info = _fuse_dimension_predictions(
            text_dimension_probabilities,
            quest_dimension_probabilities,
            m1_probabilities=m1_dimension_probabilities,
            text_weight=text_w,
            quest_weight=quest_w,
            m1_weight=m1_w,
        )

        # Blend output probabilities for reporting confidence/percentages.
        fused_dimension_probabilities: dict[str, dict[str, float]] = {}
        for left, right in [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")]:
            pair_key = f"{left}{right}"
            p_left = (
                text_w  * text_dimension_probabilities[pair_key][left]
                + quest_w * quest_dimension_probabilities[pair_key][left]
                + m1_w   * m1_dimension_probabilities[pair_key][left]
            )
            p_left = float(np.clip(p_left, 0.0, 1.0))
            fused_dimension_probabilities[pair_key] = {left: p_left, right: 1.0 - p_left}

        fused_dimension_percentages = _format_dimension_percentages(fused_dimension_probabilities)
        avg_confidence = float(np.mean([max(v.values()) for v in fused_dimension_probabilities.values()]))
        
        result = {
            "mbti_type": fused_mbti,
            "confidence": avg_confidence,
            "model_used": "dimension_voting:m1+text+questionnaire",
            "dimension_probabilities": fused_dimension_probabilities,
            "dimension_percentages": fused_dimension_percentages,
            "voting_info": voting_info,
            "module1_prediction": (m1_analysis["mbti_dimension_posteriors"], m1_analysis["confidence"]),
            "text_prediction": (text_mbti, text_confidence),
            "questionnaire_prediction": (
                quest_mbti,
                float(np.mean([max(v.values()) for v in quest_dimension_probabilities.values()])),
            ),
        }
        return result

    # Combined payloads can use both models and compare confidence.
    if use_hybrid and has_combined_input and text_available and (quest_available or quest_dim_available):
        text_value = str(input_data["text"])
        questionnaire_payload = {key: value for key, value in input_data.items() if key.startswith("Q")}

        text_mbti, text_confidence, text_dimension_probabilities = predict_from_text(text_value, text_bundle)
        text_dimension_percentages = _format_dimension_percentages(text_dimension_probabilities)

        # Always prefer dimension-level questionnaire model — it's robust to neutral answers
        if quest_dim_available:
            questionnaire_mbti, quest_dim_probs, quest_dim_pcts = predict_from_questionnaire_dimensions(
                questionnaire_payload, quest_dim_bundle
            )
            questionnaire_confidence = float(np.mean([max(v.values()) for v in quest_dim_probs.values()]))
        else:
            questionnaire_mbti, questionnaire_confidence = predict_from_questionnaire(
                questionnaire_payload, questionnaire_bundle
            )

        if text_confidence >= questionnaire_confidence:
            result = {
                "mbti_type": text_mbti,
                "confidence": text_confidence,
                "model_used": "hybrid:text",
                "dimension_probabilities": text_dimension_probabilities,
                "dimension_percentages": text_dimension_percentages,
                "text_prediction": (text_mbti, text_confidence),
                "questionnaire_prediction": (questionnaire_mbti, questionnaire_confidence),
            }
        else:
            result = {
                "mbti_type": questionnaire_mbti,
                "confidence": questionnaire_confidence,
                "model_used": "hybrid:questionnaire",
                "dimension_probabilities": text_dimension_probabilities,
                "dimension_percentages": text_dimension_percentages,
                "text_prediction": (text_mbti, text_confidence),
                "questionnaire_prediction": (questionnaire_mbti, questionnaire_confidence),
            }
        return result

    # Pure text input routes to the text model.
    if has_text_input and text_available:
        mbti, confidence, dimension_probabilities = predict_from_text(input_data, text_bundle)
        m1_analysis = analyze_personality_text(input_data)
        result = {
            "mbti_type": mbti,
            "confidence": confidence,
            "model_used": "text",
            "dimension_probabilities": dimension_probabilities,
            "dimension_percentages": _format_dimension_percentages(dimension_probabilities),
            "mbti_dimension_posteriors": m1_analysis["mbti_dimension_posteriors"],
            "nlp_keywords": m1_analysis["keywords"],
        }
        return result

    # Questionnaire-only input: prefer dimension-level models (94-97% per dimension,
    # robust to neutral answers) over the 16-class model which collapses to ~10%
    # confidence when answers are near-neutral.
    if is_questionnaire:
        if quest_dim_available:
            mbti, dim_probs, dim_pcts = predict_from_questionnaire_dimensions(
                input_data, quest_dim_bundle
            )
            avg_confidence = float(np.mean([max(v.values()) for v in dim_probs.values()]))
            result = {
                "mbti_type": mbti,
                "confidence": avg_confidence,
                "model_used": "questionnaire",
                "dimension_probabilities": dim_probs,
                "dimension_percentages": dim_pcts,
            }
            return result
        if quest_available:
            mbti, confidence = predict_from_questionnaire(input_data, questionnaire_bundle)
            result = {
                "mbti_type": mbti,
                "confidence": confidence,
                "model_used": "questionnaire",
            }
            return result

    # Fallback: use whichever model is available.
    if quest_dim_available and not has_text_input:
        mbti, dim_probs, dim_pcts = predict_from_questionnaire_dimensions(
            input_data, quest_dim_bundle
        )
        avg_confidence = float(np.mean([max(v.values()) for v in dim_probs.values()]))
        result = {
            "mbti_type": mbti,
            "confidence": avg_confidence,
            "model_used": "questionnaire",
            "dimension_probabilities": dim_probs,
            "dimension_percentages": dim_pcts,
        }
        return result
    if quest_available and not has_text_input:
        mbti, confidence = predict_from_questionnaire(input_data, questionnaire_bundle)
        result = {
            "mbti_type": mbti,
            "confidence": confidence,
            "model_used": "questionnaire",
        }
        return result

    if text_available:
        mbti, confidence, dimension_probabilities = predict_from_text(str(input_data), text_bundle)
        m1_analysis = analyze_personality_text(str(input_data))
        result = {
            "mbti_type": mbti,
            "confidence": confidence,
            "model_used": "text",
            "dimension_probabilities": dimension_probabilities,
            "dimension_percentages": _format_dimension_percentages(dimension_probabilities),
            "mbti_dimension_posteriors": m1_analysis["mbti_dimension_posteriors"],
            "nlp_keywords": m1_analysis["keywords"],
        }
        return result

    raise RuntimeError("No suitable model for this input type")


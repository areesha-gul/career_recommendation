from __future__ import annotations

import numpy as np

import modules.module2_dt as module2


class FakeBinaryModel:
    def __init__(self, classes: list[str], probabilities: list[float]) -> None:
        self.classes_ = np.array(classes)
        self._proba = np.array(probabilities, dtype=float)

    def predict(self, values: list[str]) -> np.ndarray:
        idx = int(np.argmax(self._proba))
        return np.array([self.classes_[idx]])

    def predict_proba(self, values: list) -> np.ndarray:
        return np.array([self._proba])


def test_predict_from_text_returns_dimension_probabilities():
    bundle = {
        "dimension_models": {
            "EI": FakeBinaryModel(["E", "I"], [0.60, 0.40]),
            "NS": FakeBinaryModel(["N", "S"], [0.55, 0.45]),
            "TF": FakeBinaryModel(["T", "F"], [0.67, 0.33]),
            "JP": FakeBinaryModel(["J", "P"], [0.20, 0.80]),
        },
        "dimension_pairs": [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")],
    }

    mbti, confidence, dim_probs = module2.predict_from_text("sample text", bundle)

    assert mbti == "ENTP"
    assert 0.65 < confidence < 0.66
    assert set(dim_probs.keys()) == {"EI", "NS", "TF", "JP"}
    assert dim_probs["TF"]["T"] == 0.67
    assert dim_probs["TF"]["F"] == 0.33


def test_predict_mbti_personality_includes_percentage_strings(monkeypatch):
    bundle = {
        "dimension_models": {
            "EI": FakeBinaryModel(["E", "I"], [0.60, 0.40]),
            "NS": FakeBinaryModel(["N", "S"], [0.55, 0.45]),
            "TF": FakeBinaryModel(["T", "F"], [0.67, 0.33]),
            "JP": FakeBinaryModel(["J", "P"], [0.20, 0.80]),
        },
        "dimension_pairs": [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")],
    }

    monkeypatch.setattr(module2, "load_text_model", lambda: bundle)

    def _missing_questionnaire() -> dict:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(module2, "load_questionnaire_model", _missing_questionnaire)

    result = module2.predict_mbti_personality("hello world", use_hybrid=True)

    assert result["mbti_type"] == "ENTP"
    assert result["model_used"] == "text"
    assert "dimension_probabilities" in result
    assert result["dimension_percentages"]["TF"] == "67% T, 33% F"


# ─────────────────────────────────────────────────────────────────
# Tests for dimension-level voting (new)
# ─────────────────────────────────────────────────────────────────

def test_fuse_dimension_predictions_agreement():
    """When both models agree on a dimension, that choice should win."""
    text_probs = {
        "EI": {"E": 0.7, "I": 0.3},
        "NS": {"N": 0.8, "S": 0.2},
        "TF": {"T": 0.9, "F": 0.1},
        "JP": {"J": 0.6, "P": 0.4},
    }
    quest_probs = {
        "EI": {"E": 0.75, "I": 0.25},
        "NS": {"N": 0.75, "S": 0.25},
        "TF": {"T": 0.85, "F": 0.15},
        "JP": {"J": 0.7, "P": 0.3},
    }
    
    final_mbti, voting_info = module2._fuse_dimension_predictions(text_probs, quest_probs)
    
    assert final_mbti == "ENTJ"
    assert voting_info["EI"]["agreement"] == "yes"
    assert voting_info["NS"]["agreement"] == "yes"
    assert voting_info["TF"]["agreement"] == "yes"
    assert voting_info["JP"]["agreement"] == "yes"


def test_fuse_dimension_predictions_disagreement_higher_wins():
    """When models disagree, the one with higher log-odds should win."""
    text_probs = {
        "EI": {"E": 0.8, "I": 0.2},  # Strong E
        "NS": {"N": 0.6, "S": 0.4},
        "TF": {"T": 0.5, "F": 0.5},
        "JP": {"J": 0.6, "P": 0.4},
    }
    quest_probs = {
        "EI": {"E": 0.3, "I": 0.7},  # Strong I (disagreement)
        "NS": {"N": 0.6, "S": 0.4},
        "TF": {"T": 0.5, "F": 0.5},
        "JP": {"J": 0.6, "P": 0.4},
    }
    
    final_mbti, voting_info = module2._fuse_dimension_predictions(text_probs, quest_probs)
    
    # Text is much stronger on E (0.8 vs 0.3), so E should win with equal weighting
    assert voting_info["EI"]["agreement"] == "no"
    assert voting_info["EI"]["text_winner"] == "E"
    assert voting_info["EI"]["quest_winner"] == "I"
    assert voting_info["EI"]["fused_winner"] == "E"  # Text's stronger evidence wins


def test_predict_from_questionnaire_dimensions():
    """Test dimension-level questionnaire prediction."""
    dimension_bundle = {
        "dimension_models": {
            "EI": FakeBinaryModel(["E", "I"], [0.65, 0.35]),
            "NS": FakeBinaryModel(["N", "S"], [0.70, 0.30]),
            "TF": FakeBinaryModel(["T", "F"], [0.55, 0.45]),
            "JP": FakeBinaryModel(["J", "P"], [0.60, 0.40]),
        },
        "dimension_pairs": [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")],
    }
    
    questionnaire_data = {
        f"Q{i}": 3 for i in range(1, 16)
    }
    
    mbti, dim_probs, dim_pcts = module2.predict_from_questionnaire_dimensions(
        questionnaire_data,
        dimension_bundle,
    )
    
    assert mbti == "ENTJ"
    assert set(dim_probs.keys()) == {"EI", "NS", "TF", "JP"}
    assert set(dim_pcts.keys()) == {"EI", "NS", "TF", "JP"}


def test_predict_mbti_personality_dimension_voting_includes_module1(monkeypatch):
    text_bundle = {
        "dimension_models": {
            "EI": FakeBinaryModel(["E", "I"], [0.60, 0.40]),
            "NS": FakeBinaryModel(["N", "S"], [0.55, 0.45]),
            "TF": FakeBinaryModel(["T", "F"], [0.67, 0.33]),
            "JP": FakeBinaryModel(["J", "P"], [0.20, 0.80]),
        },
        "dimension_pairs": [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")],
    }
    quest_dim_bundle = {
        "dimension_models": {
            "EI": FakeBinaryModel(["E", "I"], [0.30, 0.70]),
            "NS": FakeBinaryModel(["N", "S"], [0.80, 0.20]),
            "TF": FakeBinaryModel(["T", "F"], [0.60, 0.40]),
            "JP": FakeBinaryModel(["J", "P"], [0.70, 0.30]),
        },
        "dimension_pairs": [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")],
    }

    monkeypatch.setattr(module2, "load_text_model", lambda: text_bundle)
    monkeypatch.setattr(module2, "load_questionnaire_dimension_model", lambda: quest_dim_bundle)

    def _missing_questionnaire() -> dict:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(module2, "load_questionnaire_model", _missing_questionnaire)
    monkeypatch.setattr(
        module2,
        "analyze_personality_text",
        lambda text: {
            "mbti_dimension_posteriors": {
                "E": 0.90,
                "I": 0.10,
                "N": 0.65,
                "S": 0.35,
                "T": 0.75,
                "F": 0.25,
                "J": 0.30,
                "P": 0.70,
            },
            "confidence": 0.75,
        },
    )

    payload = {"text": "logical systems", **{f"Q{i}": 3 for i in range(1, 16)}}
    result = module2.predict_mbti_personality(payload, use_dimension_voting=True)

    assert result["model_used"] == "dimension_voting:m1+text+questionnaire"
    assert "module1_prediction" in result
    assert "voting_info" in result
    assert all("m1_winner" in v for v in result["voting_info"].values())


def test_logit_conversion():
    """Test logit conversion for fusion."""
    # Logit(0.5) should be 0
    logit_05 = module2._logit(0.5)
    assert abs(logit_05 - 0.0) < 0.01
    
    # Logit(0.7) should be positive
    logit_07 = module2._logit(0.7)
    assert logit_07 > 0
    
    # Logit(0.3) should be negative
    logit_03 = module2._logit(0.3)
    assert logit_03 < 0
    
    # Logit is monotonically increasing
    assert logit_03 < logit_05 < logit_07


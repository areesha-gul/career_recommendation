from modules.module1_nlp import analyze_personality_text


def test_analyze_personality_text_returns_keywords():
    result = analyze_personality_text("I enjoy creative design and visual storytelling.")
    assert "creative" in result["keywords"]
    assert result["confidence"] > 0


def test_analyze_personality_text_returns_8_dimension_posteriors():
    result = analyze_personality_text("I enjoy analyzing systems and planning long term goals.")
    post = result["mbti_dimension_posteriors"]

    assert set(post.keys()) == {"E", "I", "N", "S", "T", "F", "J", "P"}
    assert abs((post["E"] + post["I"]) - 1.0) < 1e-6
    assert abs((post["N"] + post["S"]) - 1.0) < 1e-6
    assert abs((post["T"] + post["F"]) - 1.0) < 1e-6
    assert abs((post["J"] + post["P"]) - 1.0) < 1e-6


def test_analyze_personality_text_returns_keyword_vector():
    result = analyze_personality_text("Creative creative design planning planning planning")
    assert "keyword_vector" in result
    assert result["keyword_vector"].get("planning", 0) >= 1

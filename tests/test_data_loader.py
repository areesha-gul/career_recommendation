from __future__ import annotations

import pandas as pd

from data_loader import _undersample_to_max


def test_undersample_preserves_mbti_type_column() -> None:
    df = pd.DataFrame(
        {
            "mbti_type": ["INTJ", "INTJ", "ENFP", "ENFP", "ENFP"],
            "text": ["a", "b", "c", "d", "e"],
        }
    )

    sampled = _undersample_to_max(df, max_per_class=2, random_state=42)

    assert "mbti_type" in sampled.columns
    assert "text" in sampled.columns
    counts = sampled["mbti_type"].value_counts().to_dict()
    assert counts["INTJ"] <= 2
    assert counts["ENFP"] <= 2
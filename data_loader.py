"""Dataset loading helpers for the hybrid MBTI training workflow.

Supports:
  - Synthetic dataset (Q1-Q15 + mbti_type + text_sample) — mbti_dataset.csv
  - MBTI 500 dataset from Kaggle (posts + type columns) — MBTI 500.csv
  - Original Kaggle dataset (posts + type columns) — mbti_1.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------
# Path candidates — checked in order, first match wins
# -------------------------------------------------------------------
DEFAULT_MBTI500_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "MBTI500.csv",
    ROOT_DIR / "data" / "raw" / "MBTI 500.csv",
    ROOT_DIR / "data" / "raw" / "mbti_500.csv",
    ROOT_DIR / "data" / "raw" / "mbti500.csv",
]

DEFAULT_KAGGLE_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "mbti_1.csv",
]

DEFAULT_SYNTHETIC_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "mbti_dataset.csv",
]

MBTI_LABEL_PATTERN = re.compile(
    r"\b(INFJ|INFP|INTJ|INTP|ISFJ|ISFP|ISTJ|ISTP|ENFJ|ENFP|ENTJ|ENTP|ESFJ|ESFP|ESTJ|ESTP)\b",
    flags=re.IGNORECASE,
)

ALL_16_TYPES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP",
]


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _resolve_existing_path(
    path: str | Path | None,
    candidates: list[Path],
    label: str = "Dataset",
) -> Path:
    """Return first existing path from explicit path or candidate list."""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{label} not found at: {candidate}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n{checked}")


def _clean_text(text: str) -> str:
    """Shared text cleaner: remove URLs, MBTI labels, non-alpha chars."""
    cleaned = str(text)
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = MBTI_LABEL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _add_realistic_noise(
    df: pd.DataFrame,
    q_columns: list[str],
    noise_level: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flip ~noise_level fraction of Q answers to simulate real human inconsistency.

    This prevents the questionnaire model from overfitting to perfectly clean
    synthetic patterns that won't appear in real user responses.
    """
    rng = np.random.default_rng(random_state)
    noisy = df.copy()

    for col in q_columns:
        mask = rng.random(len(df)) < noise_level
        if mask.any():
            original_vals = noisy.loc[mask, col].values
            noisy_vals = np.array([
                rng.choice([v for v in range(1, 6) if v != int(orig)])
                for orig in original_vals
            ])
            noisy.loc[mask, col] = noisy_vals

    return noisy


def _undersample_to_max(df: pd.DataFrame, max_per_class: int, random_state: int = 42) -> pd.DataFrame:
    """Undersample majority classes so no class exceeds max_per_class rows."""
    if "mbti_type" not in df.columns:
        raise ValueError(f"Expected 'mbti_type' column for undersampling. Found: {list(df.columns)}")

    groups = []
    for _, group_df in df.groupby("mbti_type"):
        sample_size = min(len(group_df), max_per_class)
        groups.append(group_df.sample(sample_size, random_state=random_state))

    if not groups:
        return df.copy().reset_index(drop=True)

    return pd.concat(groups, ignore_index=True)


def _oversample_to_min(df: pd.DataFrame, min_per_class: int, random_state: int = 42) -> pd.DataFrame:
    """Oversample minority classes so every class has at least min_per_class rows."""
    frames = []
    for mbti_type, group in df.groupby("mbti_type"):
        if len(group) < min_per_class:
            extras = group.sample(
                n=min_per_class - len(group),
                replace=True,
                random_state=random_state,
            )
            group = pd.concat([group, extras], ignore_index=True)
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)


# -------------------------------------------------------------------
# Public loaders
# -------------------------------------------------------------------

def load_synthetic_mbti(
    path: str | Path | None = None,
    add_noise: bool = True,
    noise_level: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load synthetic MBTI dataset (Q1-Q15 + mbti_type + text_sample).

    Args:
        path: Optional explicit path. Defaults to data/raw/mbti_dataset.csv.
        add_noise: If True, add realistic noise to Q columns to prevent
                   overfitting to perfect synthetic patterns (recommended).
        noise_level: Fraction of answers to randomly flip (default 0.05 = 5%).
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns: Q1-Q15, mbti_type, text, source.
    """
    csv_path = _resolve_existing_path(
        path=path,
        candidates=DEFAULT_SYNTHETIC_CANDIDATES,
        label="Synthetic MBTI dataset",
    )
    df = pd.read_csv(csv_path)

    required = {"mbti_type", "text_sample"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Synthetic dataset missing columns: {missing}. Found: {list(df.columns)}")

    q_columns = [col for col in [f"Q{i}" for i in range(1, 16)] if col in df.columns]
    if not q_columns:
        raise ValueError("Synthetic dataset must contain at least Q1 column.")

    select_cols = ["mbti_type", "text_sample", *q_columns]
    cleaned = df[select_cols].dropna(subset=["mbti_type", "text_sample"]).copy()
    cleaned = cleaned.rename(columns={"text_sample": "text"})

    cleaned["mbti_type"] = cleaned["mbti_type"].astype(str).str.upper().str.strip()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]

    for col in q_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(3).astype(int)
        cleaned[col] = cleaned[col].clip(1, 5)

    if add_noise:
        cleaned = _add_realistic_noise(cleaned, q_columns, noise_level, random_state)

    cleaned["source"] = "synthetic"
    return cleaned.reset_index(drop=True)


def load_mbti500(
    path: str | Path | None = None,
    max_rows: int | None = None,
    max_per_class: int = 4000,
) -> pd.DataFrame:
    """Load MBTI 500 dataset from Kaggle (zeyadkhalid/mbti-personality-types-500-dataset).

    Expected columns: 'posts' (preprocessed text), 'type' (MBTI label).
    Already preprocessed — no punctuation, stopwords, URLs, lemmatized.
    500 words per sample, ~106K records.

    Args:
        path: Optional explicit path. Defaults to data/raw/MBTI 500.csv.
        max_rows: Optional row cap (useful for quick tests).
        max_per_class: Cap per class to reduce imbalance (default 4000).

    Returns:
        DataFrame with columns: mbti_type, text, source.
    """
    csv_path = _resolve_existing_path(
        path=path,
        candidates=DEFAULT_MBTI500_CANDIDATES,
        label="MBTI 500 dataset",
    )
    read_nrows = max_rows if (max_rows is not None and max_rows > 0) else None
    df = pd.read_csv(csv_path, nrows=read_nrows)

    # Handle both possible column name variants
    col_map = {}
    if "posts" in df.columns:
        col_map["posts"] = "text"
    elif "post" in df.columns:
        col_map["post"] = "text"
    else:
        raise ValueError(f"MBTI 500 dataset must have 'posts' column. Found: {list(df.columns)}")

    if "type" in df.columns:
        col_map["type"] = "mbti_type"
    else:
        raise ValueError(f"MBTI 500 dataset must have 'type' column. Found: {list(df.columns)}")

    cleaned = df[list(col_map.keys())].rename(columns=col_map).dropna()
    cleaned["mbti_type"] = cleaned["mbti_type"].astype(str).str.upper().str.strip()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()

    # MBTI 500 is already preprocessed — only light cleanup needed
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[cleaned["text"].str.split().str.len() >= 20]
    cleaned = cleaned[cleaned["mbti_type"].isin(ALL_16_TYPES)]

    # Step 1: Undersample dominant classes (INTP 24%, INTJ 21%)
    cleaned = _undersample_to_max(cleaned, max_per_class=max_per_class)

    # Step 2: Oversample rare classes (ESFJ 181, ESFP 360) up to 800 minimum
    cleaned = _oversample_to_min(cleaned, min_per_class=800)

    cleaned["source"] = "mbti500"
    return cleaned.reset_index(drop=True)


def load_kaggle_mbti(
    path: str | Path | None = None,
    max_rows: int | None = None,
    max_per_class: int = 3000,
) -> pd.DataFrame:
    """Load original Kaggle MBTI dataset (datasnaek/mbti-type).

    Expected columns: 'type', 'posts' (pipe-separated post list).

    Args:
        path: Optional explicit path. Defaults to data/raw/mbti_1.csv.
        max_rows: Optional row cap before post explosion.
        max_per_class: Cap per class after explosion to reduce imbalance.

    Returns:
        DataFrame with columns: mbti_type, text, source.
    """
    csv_path = _resolve_existing_path(
        path=path,
        candidates=DEFAULT_KAGGLE_CANDIDATES,
        label="Kaggle MBTI dataset (mbti_1.csv)",
    )
    read_nrows = max_rows if (max_rows is not None and max_rows > 0) else None
    df = pd.read_csv(csv_path, nrows=read_nrows)

    if not {"type", "posts"}.issubset(df.columns):
        raise ValueError(
            f"Kaggle dataset must have 'type' and 'posts' columns. Found: {list(df.columns)}"
        )

    cleaned = df[["type", "posts"]].rename(columns={"type": "mbti_type", "posts": "text"}).dropna()
    cleaned["mbti_type"] = cleaned["mbti_type"].astype(str).str.upper().str.strip()

    # Explode pipe-separated posts into individual rows
    cleaned["text"] = cleaned["text"].astype(str).str.split("|||", regex=False)
    cleaned = cleaned.explode("text", ignore_index=True)

    # Clean text
    cleaned["text"] = cleaned["text"].map(_clean_text)
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[cleaned["text"].str.split().str.len() >= 10]
    cleaned = cleaned[cleaned["mbti_type"].isin(ALL_16_TYPES)]

    if cleaned.empty:
        raise ValueError("Kaggle cleaning removed all rows. Try increasing max_rows.")

    # Undersample to reduce severe imbalance
    cleaned = _undersample_to_max(cleaned, max_per_class=max_per_class)

    cleaned["source"] = "kaggle"
    return cleaned.reset_index(drop=True)


def load_text_training_data(
    source: str = "mbti500",
    path: str | Path | None = None,
    max_rows: int | None = None,
    max_per_class: int = 4000,
) -> pd.DataFrame:
    """Unified loader for text model training data.

    Args:
        source: One of 'mbti500' (recommended), 'kaggle', or 'both'.
        path: Optional explicit path override.
        max_rows: Optional row cap.
        max_per_class: Per-class cap to reduce imbalance.

    Returns:
        DataFrame with columns: mbti_type, text, source.
    """
    if source == "mbti500":
        return load_mbti500(path=path, max_rows=max_rows, max_per_class=max_per_class)

    elif source == "kaggle":
        return load_kaggle_mbti(path=path, max_rows=max_rows, max_per_class=max_per_class)

    elif source == "both":
        frames = []
        try:
            frames.append(load_mbti500(max_rows=max_rows, max_per_class=max_per_class))
        except FileNotFoundError:
            print("  [WARNING] MBTI 500 not found, skipping.")
        try:
            frames.append(load_kaggle_mbti(max_rows=max_rows, max_per_class=max_per_class))
        except FileNotFoundError:
            print("  [WARNING] Kaggle mbti_1.csv not found, skipping.")
        if not frames:
            raise FileNotFoundError("Neither MBTI 500 nor Kaggle dataset found.")
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
        return _undersample_to_max(combined, max_per_class=max_per_class)

    else:
        raise ValueError(f"Unknown source '{source}'. Choose from: 'mbti500', 'kaggle', 'both'.")
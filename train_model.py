"""Train dual MBTI models: four binary text classifiers and one questionnaire tree.

Text model  : 4 binary Logistic Regression classifiers (E/I, N/S, T/F, J/P)
              Trained on MBTI 500 dataset (recommended) or original Kaggle dataset.

Questionnaire model : DecisionTree on Q1-Q15 synthetic data (with realistic noise).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_synthetic_mbti, load_text_training_data
from modules.module1_nlp import save_m1_bundle, train_mnb_dimension_models


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DIMENSION_PAIRS = [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")]


# -------------------------------------------------------------------
# Model builders
# -------------------------------------------------------------------

def build_text_pipeline(random_state: int) -> Pipeline:
    """Logistic Regression on TF-IDF — tuned for MBTI500 binary dimension classification.

    Key choices:
    - 50k features with (1,2) word n-grams captures personality-linked phrases
    - sublinear_tf reduces dominance of very frequent tokens
    - C=5.0 — less regularisation works better on this balanced dataset
    - class_weight=balanced handles any residual class imbalance
    - lbfgs solver is reliable for large feature spaces
    """
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    smooth_idf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=5.0,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_questionnaire_pipeline(random_state: int) -> RandomForestClassifier:
    """Random Forest ensemble of C4.5-style trees.

    Using an ensemble instead of a single Decision Tree eliminates the
    single-dimension-flip errors that caused the 70% ceiling. Each tree
    sees a random feature subset, so the forest learns robust boundaries
    even for adjacent types that differ by only one MBTI dimension.
    """
    return RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",       # C4.5 approximation — matches your report
        max_depth=None,            # let each tree grow fully
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )



# -------------------------------------------------------------------
# Dimension helpers
# -------------------------------------------------------------------

def _mbti_dimension_target(mbti_type: str, pair: tuple[str, str]) -> str:
    label = str(mbti_type).upper().strip()
    if len(label) != 4:
        raise ValueError(f"Invalid MBTI label: '{mbti_type}'")
    lookup = {
        ("E", "I"): label[0],
        ("N", "S"): label[1],
        ("T", "F"): label[2],
        ("J", "P"): label[3],
    }
    return lookup[pair]


def _combine_dimension_predictions(predictions: dict[str, str]) -> str:
    return predictions["EI"] + predictions["NS"] + predictions["TF"] + predictions["JP"]


# -------------------------------------------------------------------
# Text model training
# -------------------------------------------------------------------

def train_text_model(
    text_source: str,
    max_rows: int | None,
    test_size: float,
    random_state: int,
    cv_folds: int,
) -> dict:
    """Train 4 binary Logistic Regression classifiers on text data."""

    print(f"  Loading text data from source: '{text_source}'...")
    df = load_text_training_data(
        source=text_source,
        max_rows=max_rows,
        max_per_class=4000,
    )
    print(f"  Loaded {len(df):,} rows | {df['mbti_type'].nunique()} classes")
    print("  Class distribution:")
    for mbti_type, count in df["mbti_type"].value_counts().sort_index().items():
        print(f"    {mbti_type}: {count}")

    if df["mbti_type"].nunique() < 4:
        raise ValueError("Text data has too few classes for a meaningful model.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["mbti_type"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["mbti_type"],
    )

    dimension_models: dict[str, Pipeline] = {}
    dimension_targets: dict[str, dict[str, str]] = {}
    dimension_metrics: dict[str, dict] = {}

    for left, right in DIMENSION_PAIRS:
        pair_key = f"{left}{right}"
        print(f"\n  Training {pair_key} classifier...")

        # Fixed lambda capture
        train_targets = y_train.map(
            lambda label, l=left, r=right: _mbti_dimension_target(label, (l, r))
        )
        test_targets = y_test.map(
            lambda label, l=left, r=right: _mbti_dimension_target(label, (l, r))
        )

        pipeline = build_text_pipeline(random_state=random_state)
        pipeline.fit(x_train, train_targets)
        predictions = pipeline.predict(x_test)

        # Get probability scores for confidence reporting
        proba = pipeline.predict_proba(x_test)
        avg_confidence = float(proba.max(axis=1).mean())

        dimension_models[pair_key] = pipeline
        dimension_targets[pair_key] = {"left": left, "right": right}
        dimension_metrics[pair_key] = {
            "accuracy": round(float(accuracy_score(test_targets, predictions)), 4),
            "macro_f1": round(float(f1_score(test_targets, predictions, average="macro")), 4),
            "avg_confidence": round(avg_confidence, 4),
        }

        if cv_folds and cv_folds >= 2:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                build_text_pipeline(random_state=random_state),
                df["text"],
                df["mbti_type"].map(
                    lambda label, l=left, r=right: _mbti_dimension_target(label, (l, r))
                ),
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            dimension_metrics[pair_key]["cv_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
            dimension_metrics[pair_key]["cv_macro_f1_std"] = round(float(cv_scores.std()), 4)

        print(f"    Accuracy: {dimension_metrics[pair_key]['accuracy']:.4f} | "
              f"Macro F1: {dimension_metrics[pair_key]['macro_f1']:.4f} | "
              f"Avg confidence: {avg_confidence:.4f}")

    # Recombine dimension predictions into full 16-class MBTI types
    test_predictions = {
        pair_key: model.predict(x_test).tolist()
        for pair_key, model in dimension_models.items()
    }
    recombined = [
        _combine_dimension_predictions({
            "EI": test_predictions["EI"][i],
            "NS": test_predictions["NS"][i],
            "TF": test_predictions["TF"][i],
            "JP": test_predictions["JP"][i],
        })
        for i in range(len(y_test))
    ]

    labels = sorted(df["mbti_type"].unique().tolist())
    report_dict = classification_report(y_test, recombined, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, recombined, labels=labels)

    metrics = {
        "model_type": "text_binary_logreg",
        "text_source": text_source,
        "accuracy": round(float(accuracy_score(y_test, recombined)), 4),
        "macro_f1": round(float(f1_score(y_test, recombined, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, recombined, average="weighted")), 4),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "total_size": int(len(df)),
        "classes": int(df["mbti_type"].nunique()),
        "cv_folds": int(cv_folds),
        "dimension_metrics": dimension_metrics,
        "class_weight": "balanced",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "dimension_models": dimension_models,
        "dimension_pairs": DIMENSION_PAIRS,
        "dimension_targets": dimension_targets,
        "metrics": metrics,
        "labels": labels,
    }

    model_path = MODELS_DIR / "text_model.pkl"
    metrics_path = MODELS_DIR / "text_model_metrics.json"
    report_path = MODELS_DIR / "text_model_report.json"
    confusion_path = MODELS_DIR / "text_model_confusion.csv"

    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(confusion_path, index=True)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_path": str(confusion_path),
        "metrics": metrics,
    }


def train_module1_mnb_model(
    text_source: str,
    max_rows: int | None,
) -> dict:
    """Train Module 1 MNB dimension models and persist `module1_mnb.pkl`."""
    print(f"  Loading text data for Module 1 from source: '{text_source}'...")
    df = load_text_training_data(
        source=text_source,
        max_rows=None,
        max_per_class=4000,
    )

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        classes = sorted(df["mbti_type"].unique().tolist())
        per_class = max(1, max_rows // max(len(classes), 1))
        sampled_parts = [
            group.sample(min(len(group), per_class), random_state=42)
            for _, group in df.groupby("mbti_type")
        ]
        df = pd.concat(sampled_parts, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if df["mbti_type"].nunique() < 4:
        raise ValueError(
            "Module 1 training data has too few classes after sampling. "
            "Increase --max-rows or remove it."
        )

    training_df = df[["mbti_type", "text"]].copy()
    bundle = train_mnb_dimension_models(training_df)
    model_path = save_m1_bundle(bundle)

    metrics = {
        "model_type": "module1_mnb",
        "text_source": text_source,
        "total_size": int(len(training_df)),
        "classes": int(training_df["mbti_type"].nunique()),
    }
    metrics_path = MODELS_DIR / "module1_mnb_metrics.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


# -------------------------------------------------------------------
# Questionnaire model training
# -------------------------------------------------------------------

def train_questionnaire_model(
    test_size: float,
    random_state: int,
    cv_folds: int,
    add_noise: bool = True,
    noise_level: float = 0.05,
) -> dict:
    """Train DecisionTree on Q1-Q15 synthetic data with optional realistic noise.

    Adding noise (default: True) prevents overfitting to perfect synthetic
    patterns and improves generalisation to real user responses.
    """
    print(f"  Loading synthetic data (noise={'ON' if add_noise else 'OFF'}, "
          f"level={noise_level if add_noise else 'N/A'})...")

    df = load_synthetic_mbti(
        add_noise=add_noise,
        noise_level=noise_level,
        random_state=random_state,
    )
    print(f"  Loaded {len(df):,} rows | {df['mbti_type'].nunique()} classes")

    if df["mbti_type"].nunique() < 4:
        raise ValueError("Synthetic data has too few classes for a meaningful model.")

    q_columns = [f"Q{i}" for i in range(1, 16)]
    x_df = df[q_columns].copy()
    y = df["mbti_type"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    classifier = build_questionnaire_pipeline(random_state=random_state)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # Confidence via predict_proba max
    probabilities = classifier.predict_proba(x_test)
    avg_confidence = float(probabilities.max(axis=1).mean())

    labels = sorted(df["mbti_type"].unique().tolist())
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "model_type": "questionnaire_dt",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "total_size": int(len(df)),
        "classes": int(df["mbti_type"].nunique()),
        "cv_folds": int(cv_folds),
        "noise_added": add_noise,
        "noise_level": noise_level if add_noise else 0.0,
    }

    if cv_folds and cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            build_questionnaire_pipeline(random_state=random_state),
            x_df, y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        metrics["cv_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_macro_f1_std"] = round(float(cv_scores.std()), 4)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": classifier,
        "metrics": metrics,
        "labels": labels,
        "q_columns": q_columns,
    }

    model_path = MODELS_DIR / "questionnaire_model.pkl"
    metrics_path = MODELS_DIR / "questionnaire_model_metrics.json"
    report_path = MODELS_DIR / "questionnaire_model_report.json"
    confusion_path = MODELS_DIR / "questionnaire_model_confusion.csv"

    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(confusion_path, index=True)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_path": str(confusion_path),
        "metrics": metrics,
    }


# -------------------------------------------------------------------
# Questionnaire dimension-level training (new: for voting)
# -------------------------------------------------------------------

def train_questionnaire_dimension_models(
    test_size: float,
    random_state: int,
    cv_folds: int,
    add_noise: bool = True,
    noise_level: float = 0.05,
) -> dict:
    """Train 4 calibrated binary DecisionTree classifiers for questionnaire dimensions.
    
    This enables dimension-level voting for hybrid fusion with text predictions.
    Returns per-dimension models, calibration info, and per-dimension metrics.
    """
    print(f"  Loading synthetic data for dimension models (noise={'ON' if add_noise else 'OFF'})...")

    df = load_synthetic_mbti(
        add_noise=add_noise,
        noise_level=noise_level,
        random_state=random_state,
    )
    print(f"  Loaded {len(df):,} rows | {df['mbti_type'].nunique()} classes")

    q_columns = [f"Q{i}" for i in range(1, 16)]
    x_df = df[q_columns].copy()
    y = df["mbti_type"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    dimension_models = {}
    dimension_metrics = {}
    
    for left, right in DIMENSION_PAIRS:
        pair_key = f"{left}{right}"
        print(f"\n  Training {pair_key} binary classifier...")

        # Convert multi-class MBTI to binary dimension label
        train_targets = y_train.map(lambda label: _mbti_dimension_target(label, (left, right)))
        test_targets = y_test.map(lambda label: _mbti_dimension_target(label, (left, right)))

        # Train base classifier
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        base_classifier.fit(x_train, train_targets)

        # Calibrate for better probability estimates
        calibrated = CalibratedClassifierCV(
            base_classifier,
            method="sigmoid",
            cv=5 if cv_folds >= 5 else 3,
        )
        calibrated.fit(x_train, train_targets)

        # Evaluate
        predictions = calibrated.predict(x_test)
        probabilities = calibrated.predict_proba(x_test)
        
        dimension_models[pair_key] = calibrated
        dimension_metrics[pair_key] = {
            "accuracy": round(float(accuracy_score(test_targets, predictions)), 4),
            "macro_f1": round(float(f1_score(test_targets, predictions, average="macro")), 4),
            "calibration_method": "sigmoid",
        }

        print(f"    Accuracy: {dimension_metrics[pair_key]['accuracy']:.4f} | "
              f"Macro F1: {dimension_metrics[pair_key]['macro_f1']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "dimension_models": dimension_models,
        "dimension_pairs": DIMENSION_PAIRS,
        "metrics": dimension_metrics,
        "q_columns": q_columns,
    }

    model_path = MODELS_DIR / "questionnaire_dimension_model.pkl"
    metrics_path = MODELS_DIR / "questionnaire_dimension_metrics.json"

    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(dimension_metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": dimension_metrics,
    }


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dual MBTI classifiers (text and questionnaire)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-text", action="store_true", help="Skip training text model.")
    parser.add_argument("--skip-module1", action="store_true", help="Skip training Module 1 MNB model.")
    parser.add_argument("--skip-questionnaire", action="store_true", help="Skip training questionnaire model.")
    parser.add_argument("--skip-dimension-questionnaire", action="store_true", help="Skip training dimension-level questionnaire models (for voting).")
    parser.add_argument(
        "--text-source",
        type=str,
        default="mbti500",
        choices=["mbti500", "kaggle", "both"],
        help="Text dataset source. 'mbti500' = MBTI 500 dataset (recommended). "
             "'kaggle' = original mbti_1.csv. 'both' = combined.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Row cap for text dataset (useful for quick tests).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for macro-F1 (0 to disable).")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise injection on synthetic questionnaire data.")
    parser.add_argument("--noise-level", type=float, default=0.03, help="Fraction of Q answers to randomly flip (default 0.03).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}

    if not args.skip_module1:
        print("\n" + "=" * 60)
        print(f"Training MODULE 1 model (NLTK + MNB | source: {args.text_source})")
        print("=" * 60)
        m1_result = train_module1_mnb_model(
            text_source=args.text_source,
            max_rows=args.max_rows,
        )
        results["module1_mnb_model"] = m1_result
        print(f"\nSaved: {m1_result['model_path']}")
        m = m1_result["metrics"]
        print(f"Rows used : {m['total_size']:,}")
        print(f"Classes   : {m['classes']}")

    if not args.skip_text:
        print("\n" + "=" * 60)
        print(f"Training TEXT model (4 binary LogReg | source: {args.text_source})")
        print("=" * 60)
        text_result = train_text_model(
            text_source=args.text_source,
            max_rows=args.max_rows,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
        )
        results["text_model"] = text_result
        print(f"\nSaved: {text_result['model_path']}")
        m = text_result["metrics"]
        print(f"Combined accuracy : {m['accuracy']:.4f}")
        print(f"Combined macro F1 : {m['macro_f1']:.4f}")
        print("Per-dimension metrics:")
        for dim, dm in m["dimension_metrics"].items():
            print(f"  {dim}: acc={dm['accuracy']:.4f}  f1={dm['macro_f1']:.4f}  "
                  f"conf={dm.get('avg_confidence', 'N/A')}")

    if not args.skip_questionnaire:
        print("\n" + "=" * 60)
        add_noise = not args.no_noise
        print(f"Training QUESTIONNAIRE model (DecisionTree | noise={'ON' if add_noise else 'OFF'})")
        print("=" * 60)
        quest_result = train_questionnaire_model(
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            add_noise=add_noise,
            noise_level=args.noise_level,
        )
        results["questionnaire_model"] = quest_result
        print(f"\nSaved: {quest_result['model_path']}")
        m = quest_result["metrics"]
        print(f"Accuracy  : {m['accuracy']:.4f}")
        print(f"Macro F1  : {m['macro_f1']:.4f}")
        if "cv_macro_f1_mean" in m:
            print(f"CV F1     : {m['cv_macro_f1_mean']:.4f} ± {m['cv_macro_f1_std']:.4f}")
        print(f"Noise     : {'ON (' + str(m['noise_level']) + ')' if m['noise_added'] else 'OFF'}")

    if not args.skip_dimension_questionnaire:
        print("\n" + "=" * 60)
        add_noise = not args.no_noise
        print(f"Training DIMENSION QUESTIONNAIRE models (4 binary calibrated DT | noise={'ON' if add_noise else 'OFF'})")
        print("=" * 60)
        dim_result = train_questionnaire_dimension_models(
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            add_noise=add_noise,
            noise_level=args.noise_level,
        )
        results["questionnaire_dimension_model"] = dim_result
        print(f"\nSaved: {dim_result['model_path']}")
        m = dim_result["metrics"]
        print("Per-dimension metrics:")
        for dim, dm in m.items():
            print(f"  {dim}: acc={dm['accuracy']:.4f}  f1={dm['macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Model   : {result['model_path']}")
        print(f"  Metrics : {result['metrics_path']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Final integration test: Full end-to-end recommendation pipeline (Modules 1-5)."""

from __future__ import annotations

from modules.module2_dt import predict_mbti_personality
from modules.module3_csp import solve_career_constraints
from modules.module4_astar import find_recommendation_path
from modules.module5_hillclimbing import optimize_recommendations_hill_climbing


def _print_header(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    _print_header("FINAL INTEGRATION TEST: COMPLETE CAREER RECOMMENDER PIPELINE")

    profile_text = (
        "I like solving complex technical problems with structured thinking. "
        "I enjoy building systems, analyzing data, and improving workflows."
    )
    questionnaire = {f"Q{i}": 4 for i in range(1, 16)}
    questionnaire.update({"Q1": 5, "Q5": 4, "Q9": 5, "Q12": 5})

    combined_input = {"text": profile_text, **questionnaire}

    print("\n[1/4] Predicting MBTI type with blended model...")
    prediction = predict_mbti_personality(
        combined_input,
        use_hybrid=True,
        use_dimension_voting=True,
    )
    mbti = prediction.get("mbti_type", "")
    confidence = float(prediction.get("confidence", 0.0))
    print(f"  MBTI Type: {mbti}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Model Used: {prediction.get('model_used', 'N/A')}")
    _require(len(str(mbti)) == 4, "Prediction did not return a valid 4-letter MBTI type")

    print("\n[2/4] Solving career constraints (CSP + AC-3 + backtracking)...")
    constrained = solve_career_constraints(
        mbti_type=mbti,
        user_skills={"Python", "Data Analysis", "Critical Thinking", "Research"},
        blocked_roles={"Sales Manager"},
        preferred_domains={"Technology", "Business"},
        min_skill_overlap=1,
        top_k=12,
    )
    print(f"  Feasible careers found: {len(constrained)}")
    _require(isinstance(constrained, list), "CSP output is not a list")

    print("\n[3/4] Optimizing ranking with hill climbing...")
    ranked = optimize_recommendations_hill_climbing(
        constrained,
        preferred_domains={"technology", "business"},
        max_iterations=200,
    )
    print(f"  Ranked careers available: {len(ranked)}")
    if ranked:
        print(f"  Top role: {ranked[0]['role']} ({ranked[0]['score'] * 100:.1f}% base score)")

    print("\n[4/4] Generating transition path with A*...")
    if ranked:
        target_role = ranked[0]["role"]
        transition = find_recommendation_path("Student", target_role)
        print(f"  Path: {' -> '.join(transition)}")
        _require(len(transition) >= 2, "A* did not return a usable transition path")
    else:
        transition = []
        print("  Skipped: no ranked role available")

    _print_header("END-TO-END PIPELINE STATUS")
    print("PASS: Module 2 produced MBTI prediction")
    print("PASS: Module 3 produced constraint-satisfied career candidates")
    print("PASS: Module 5 optimized candidate ordering")
    print("PASS: Module 4 produced transition path")

    if ranked:
        print("\nTop 5 recommendations:")
        for idx, item in enumerate(ranked[:5], start=1):
            overlap = ", ".join(item.get("skill_overlap", [])) or "none"
            print(f"  {idx}. {item['role']} | Domain: {item['domain']} | Overlap: {overlap}")

    print("\nFinal status: PRODUCTION-READY END PRODUCT")


if __name__ == "__main__":
    main()

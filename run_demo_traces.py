"""
Execution Demonstrations & Test Cases
Personality-Based Career Recommendation System
Generates documented traces for report submission.
"""

from __future__ import annotations
import json
from modules.module1_nlp import analyze_personality_text
from modules.module2_dt import predict_mbti_personality
from modules.module3_csp import solve_career_constraints
from modules.module4_astar import find_recommendation_path
from modules.module5_hillclimbing import optimize_recommendations_hill_climbing

SEP  = "=" * 70
SEP2 = "-" * 70

# ─────────────────────────────────────────────────────────────────────────────
# TEST INPUTS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": 1,
        "label": "INTJ Profile — Questionnaire strong, text ambiguous",
        "text": (
            "I work best on my own or in very small groups. I am drawn to "
            "abstract ideas and long term strategy. I trust logic over gut "
            "feel and like having a clear plan before starting anything."
        ),
        "questionnaire": {
            "Q1":1,"Q2":1,"Q3":1,"Q4":2,
            "Q5":1,"Q6":5,"Q7":5,"Q8":4,
            "Q9":5,"Q10":2,"Q11":5,
            "Q12":5,"Q13":4,"Q14":5,"Q15":4,
        },
        "skills": ["Python", "research", "systems thinking", "data analysis"],
        "domains": {"Technology"},
        "current_role": "Software Engineer",
        "expected_mbti": "INTJ",
    },
    {
        "id": 2,
        "label": "ENFP Profile — Text and questionnaire agree",
        "text": (
            "honestly i love meeting new people and bouncing ideas around. "
            "i get bored with routine really fast and i thrive on exploring "
            "possibilities. i follow my gut and care deeply about people. "
            "i hate rigid schedules and prefer keeping options open."
        ),
        "questionnaire": {
            "Q1":5,"Q2":5,"Q3":5,"Q4":5,
            "Q5":2,"Q6":5,"Q7":5,"Q8":5,
            "Q9":2,"Q10":5,"Q11":2,
            "Q12":1,"Q13":1,"Q14":1,"Q15":1,
        },
        "skills": ["communication", "creativity", "writing", "teamwork"],
        "domains": {"Creative Arts"},
        "current_role": "Marketing Coordinator",
        "expected_mbti": "ENFP",
    },
    {
        "id": 3,
        "label": "ISTJ Profile — Text and questionnaire agree",
        "text": (
            "i prefer working alone with clear instructions and defined goals. "
            "i rely on facts and past experience not hunches. i like detailed "
            "plans and sticking to them. i find abstract theorising a waste "
            "of time if it does not lead to concrete results."
        ),
        "questionnaire": {
            "Q1":1,"Q2":1,"Q3":2,"Q4":2,
            "Q5":5,"Q6":1,"Q7":1,"Q8":1,
            "Q9":5,"Q10":2,"Q11":5,
            "Q12":5,"Q13":5,"Q14":5,"Q15":5,
        },
        "skills": ["accounting", "Excel", "project management", "compliance"],
        "domains": {"Business"},
        "current_role": "Accountant",
        "expected_mbti": "ISTJ",
    },
    {
        "id": 4,
        "label": "Edge Case — Empty text, questionnaire only",
        "text": "",
        "questionnaire": {
            "Q1":4,"Q2":4,"Q3":3,"Q4":3,
            "Q5":3,"Q6":3,"Q7":3,"Q8":3,
            "Q9":3,"Q10":3,"Q11":3,
            "Q12":3,"Q13":3,"Q14":3,"Q15":3,
        },
        "skills": [],
        "domains": set(),
        "current_role": "",
        "expected_mbti": None,
    },
    {
        "id": 5,
        "label": "Edge Case — Extreme ENFP answers",
        "text": "i love people and spontaneity and going with the flow always",
        "questionnaire": {
            "Q1":5,"Q2":5,"Q3":5,"Q4":5,
            "Q5":1,"Q6":5,"Q7":5,"Q8":5,
            "Q9":1,"Q10":5,"Q11":1,
            "Q12":1,"Q13":1,"Q14":1,"Q15":1,
        },
        "skills": ["public speaking", "empathy"],
        "domains": {"Education"},
        "current_role": "",
        "expected_mbti": "ENFP",
    },
    {
        "id": 6,
        "label": "Edge Case — Extreme ISTJ answers",
        "text": "i prefer facts routines and working alone with clear rules",
        "questionnaire": {
            "Q1":1,"Q2":1,"Q3":1,"Q4":1,
            "Q5":5,"Q6":1,"Q7":1,"Q8":1,
            "Q9":5,"Q10":1,"Q11":5,
            "Q12":5,"Q13":5,"Q14":5,"Q15":5,
        },
        "skills": ["data entry", "auditing"],
        "domains": {"Business"},
        "current_role": "",
        "expected_mbti": "ISTJ",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(s: dict) -> None:
    print(SEP)
    print(f"TEST CASE {s['id']}: {s['label']}")
    print(SEP)

    # ── Module 1: NLP analysis ────────────────────────────────────────────
    print("\n[MODULE 1 — NLP Text Preprocessing & MNB Posteriors]")
    if s["text"].strip():
        m1 = analyze_personality_text(s["text"])
        print(f"  Input text     : {s['text'][:80]}...")
        print(f"  Keywords (top5): {m1['keywords'][:5]}")
        print(f"  Confidence     : {m1['confidence']:.4f}")
        print(f"  Posteriors     : E={m1['mbti_dimension_posteriors']['E']:.3f} "
              f"I={m1['mbti_dimension_posteriors']['I']:.3f} | "
              f"N={m1['mbti_dimension_posteriors']['N']:.3f} "
              f"S={m1['mbti_dimension_posteriors']['S']:.3f} | "
              f"T={m1['mbti_dimension_posteriors']['T']:.3f} "
              f"F={m1['mbti_dimension_posteriors']['F']:.3f} | "
              f"J={m1['mbti_dimension_posteriors']['J']:.3f} "
              f"P={m1['mbti_dimension_posteriors']['P']:.3f}")
    else:
        print("  Input text     : (empty — skipping text analysis)")

    # ── Module 2: Prediction ──────────────────────────────────────────────
    print("\n[MODULE 2 — MBTI Prediction (Decision Tree / Dual Model)]")

    # Questionnaire only
    q_result = predict_mbti_personality(s["questionnaire"], use_hybrid=False)
    print(f"  Questionnaire  : {q_result['mbti_type']} "
          f"(confidence={q_result['confidence']*100:.1f}%, model={q_result['model_used']})")

    # Text only (if text provided)
    if s["text"].strip():
        t_result = predict_mbti_personality(s["text"], use_hybrid=False)
        print(f"  Text only      : {t_result['mbti_type']} "
              f"(confidence={t_result['confidence']*100:.1f}%, model={t_result['model_used']})")
        dim_pcts = t_result.get("dimension_percentages", {})
        for dim, pct in dim_pcts.items():
            print(f"    {dim}: {pct}")

    # Blended (dimension voting)
    if s["text"].strip():
        payload = {"text": s["text"], **s["questionnaire"]}
        b_result = predict_mbti_personality(payload, use_hybrid=True, use_dimension_voting=True)
        print(f"  Blended        : {b_result['mbti_type']} "
              f"(confidence={b_result['confidence']*100:.1f}%, model={b_result['model_used']})")
        final_type = b_result["mbti_type"]
    else:
        final_type = q_result["mbti_type"]

    if s["expected_mbti"]:
        match = "✓ CORRECT" if final_type == s["expected_mbti"] else f"✗ expected {s['expected_mbti']}"
        print(f"  Expected       : {s['expected_mbti']}  →  {match}")

    # ── Module 3: CSP Career Matching ─────────────────────────────────────
    print("\n[MODULE 3 — CSP Career Constraint Satisfaction]")
    csp_results = solve_career_constraints(
        mbti_type=final_type,
        user_skills=s["skills"] if s["skills"] else None,
        preferred_domains=s["domains"] if s["domains"] else None,
        min_skill_overlap=0,
        top_k=5,
    )
    if csp_results:
        print(f"  Candidates found: {len(csp_results)}")
        for i, r in enumerate(csp_results[:3], 1):
            print(f"  #{i} {r['role']} | domain={r['domain']} | "
                  f"score={r['score']:.3f} | overlap={r['skill_overlap']}")
    else:
        print("  No candidates found (try relaxing domain/skill filters)")

    # ── Module 4: A* Path Finding ─────────────────────────────────────────
    print("\n[MODULE 4 — A* Career Transition Path]")
    if s["current_role"] and csp_results:
        target = csp_results[0]["role"]
        path = find_recommendation_path(s["current_role"], target)
        print(f"  From : {s['current_role']}")
        print(f"  To   : {target}")
        print(f"  Path : {' → '.join(path)}")
        print(f"  Steps: {len(path)}")
    else:
        print("  Skipped (no current role or no CSP results)")

    # ── Module 5: Hill Climbing Optimisation ──────────────────────────────
    print("\n[MODULE 5 — Hill Climbing Recommendation Ranking]")
    if csp_results:
        optimised = optimize_recommendations_hill_climbing(
            csp_results,
            preferred_domains=s["domains"] if s["domains"] else None,
            max_iterations=200,
        )
        print(f"  Before optimisation: {[r['role'] for r in csp_results[:5]]}")
        print(f"  After optimisation : {[r['role'] for r in optimised[:5]]}")
        reordered = csp_results[:5] != optimised[:5]
        print(f"  Ranking changed    : {'Yes' if reordered else 'No (already optimal)'}")
    else:
        print("  Skipped (no CSP results)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm_comparison() -> None:
    print(SEP)
    print("ALGORITHMIC CONFIGURATION COMPARISON")
    print(SEP)
    print("\nComparing prediction modes on the same INTJ input:\n")

    text = (
        "i prefer working alone on complex problems. i think in systems and "
        "long term strategies. logic drives my decisions not emotions. i plan "
        "everything before starting and hate being interrupted mid task."
    )
    q = {"Q1":1,"Q2":1,"Q3":1,"Q4":2,"Q5":1,"Q6":5,"Q7":5,"Q8":4,
         "Q9":5,"Q10":2,"Q11":5,"Q12":5,"Q13":4,"Q14":5,"Q15":4}

    configs = [
        ("Text model only",         text,                    False, False),
        ("Questionnaire only",       q,                       False, False),
        ("Hybrid (confidence pick)", {"text": text, **q},     True,  False),
        ("Dimension voting (full)",  {"text": text, **q},     True,  True),
    ]

    print(f"  {'Configuration':<30} {'Type':<6} {'Confidence':>12}  {'Model'}")
    print(f"  {'-'*30} {'-'*6} {'-'*12}  {'-'*35}")
    for label, inp, hybrid, voting in configs:
        r = predict_mbti_personality(inp, use_hybrid=hybrid, use_dimension_voting=voting)
        print(f"  {label:<30} {r['mbti_type']:<6} {r['confidence']*100:>10.1f}%  {r['model_used']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("EXECUTION DEMONSTRATIONS & TEST CASES")
    print("Personality-Based Career Recommendation System")
    print(SEP)
    print()

    for scenario in SCENARIOS:
        run_scenario(scenario)

    run_algorithm_comparison()

    print(SEP)
    print("All test cases completed.")
    print(SEP)

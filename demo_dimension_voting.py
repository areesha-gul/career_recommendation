#!/usr/bin/env python
"""Demo: Dimension-level voting in action"""

from modules.module2_dt import predict_mbti_personality

# Test case 1: Text input
print("=" * 70)
print("DEMO 1: Text Input → Dimension Voting")
print("=" * 70)

text = "I love analyzing data and building logical systems. I prefer working independently on complex problems and thinking critically about frameworks."

result = predict_mbti_personality(text, use_dimension_voting=False)
print(f"\nText Input: {text[:60]}...")
print(f"Model Used: {result['model_used']}")
print(f"Predicted MBTI: {result['mbti_type']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Per-dimension probabilities:")
for dim, probs in result.get("dimension_probabilities", {}).items():
    print(f"  {dim}: {result.get('dimension_percentages', {}).get(dim, 'N/A')}")

# Test case 2: Questionnaire input
print("\n" + "=" * 70)
print("DEMO 2: Questionnaire Input → Dimension Voting")
print("=" * 70)

questionnaire = {
    "Q1": 5, "Q2": 4, "Q3": 5, "Q4": 4,   # E/I (high = E)
    "Q5": 5, "Q6": 5, "Q7": 4, "Q8": 5,   # N/S (high = N)
    "Q9": 5, "Q10": 4, "Q11": 5,           # T/F (high = T)
    "Q12": 4, "Q13": 5, "Q14": 4, "Q15": 5, # J/P (high = J)
}

result_q = predict_mbti_personality(questionnaire, use_dimension_voting=False)
print(f"\nQuestionnaire Responses: All high (4-5) with E/N/T/J bias")
print(f"Model Used: {result_q['model_used']}")
print(f"Predicted MBTI: {result_q['mbti_type']}")
print(f"Confidence: {result_q['confidence']:.2%}")

# Test case 3: Combined input with dimension voting
print("\n" + "=" * 70)
print("DEMO 3: Combined (Text + Questionnaire) → Dimension Voting")
print("=" * 70)

combined = {
    "text": "I love analyzing data and building logical systems. I prefer working independently on complex problems and thinking critically about frameworks.",
    **questionnaire
}

try:
    result_voted = predict_mbti_personality(combined, use_dimension_voting=True)
    print(f"\nCombined Input: Text + Questionnaire")
    print(f"Model Used: {result_voted['model_used']}")
    print(f"Predicted MBTI: {result_voted['mbti_type']}")
    print(f"Confidence: {result_voted['confidence']:.2%}")
    
    if "voting_info" in result_voted:
        print(f"\nPer-Dimension Voting Info:")
        for dim, info in result_voted["voting_info"].items():
            print(f"  {dim}:")
            print(f"    Text winner: {info['text_winner']}")
            print(f"    Questionnaire winner: {info['quest_winner']}")
            print(f"    Fused winner: {info['fused_winner']}")
            print(f"    Agreement: {info['agreement']}")
except FileNotFoundError as e:
    print(f"\nNote: Dimension models not yet trained. Train with:")
    print(f"  python train_model.py --skip-text")
    print(f"\nError: {e}")

print("\n" + "=" * 70)
print("Demo complete! ✅")
print("=" * 70)

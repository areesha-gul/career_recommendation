from modules.module3_csp import apply_constraints
from modules.module3_csp import recommend_roles_from_mbti


def test_apply_constraints_filters_blocked_roles():
    result = apply_constraints(["Data Analyst", "UX Designer"], {"UX Designer"})
    assert result == ["Data Analyst"]


def test_recommend_roles_from_mbti_returns_candidates():
    roles = recommend_roles_from_mbti("INTJ", top_k=3)
    assert isinstance(roles, list)
    assert len(roles) <= 3

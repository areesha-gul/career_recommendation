"""Module 3: Constraint satisfaction utilities for recommendations."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CAREERS_PATH = ROOT_DIR / "data" / "raw" / "careers.json"


def _load_careers(careers_path: str | Path = DEFAULT_CAREERS_PATH) -> list[dict]:
    path = Path(careers_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _normalize_set(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _build_candidate_records(
    careers: list[dict],
    mbti_type: str,
    user_skills: set[str],
    blocked_roles: set[str],
    preferred_domains: set[str],
    min_skill_overlap: int,
) -> list[dict]:
    """Build candidate role records that satisfy hard constraints."""
    mbti = str(mbti_type).upper().strip()
    min_overlap = max(0, int(min_skill_overlap))

    results: list[dict] = []
    for career in careers:
        role = str(career.get("name", "")).strip()
        if not role:
            continue
        if role.lower() in blocked_roles:
            continue

        mbti_match = {str(v).upper().strip() for v in career.get("mbti_match", [])}
        if mbti not in mbti_match:
            continue

        domain = str(career.get("domain", "")).strip()
        domain_l = domain.lower()
        if preferred_domains and domain_l not in preferred_domains:
            continue

        required_skills = [str(v).strip() for v in career.get("required_skills", []) if str(v).strip()]
        required_skill_set = {v.lower() for v in required_skills}

        overlap_set = required_skill_set.intersection(user_skills)
        overlap_count = len(overlap_set)
        if overlap_count < min_overlap:
            continue

        skill_coverage = overlap_count / max(len(required_skill_set), 1)
        mbti_match_score = 1.0
        domain_bonus = 0.1 if preferred_domains and domain_l in preferred_domains else 0.0
        score = 0.55 * skill_coverage + 0.35 * mbti_match_score + domain_bonus

        results.append(
            {
                "role": role,
                "domain": domain,
                "required_skills": required_skills,
                "skill_overlap": sorted(overlap_set),
                "skill_overlap_count": overlap_count,
                "score": float(score),
            }
        )

    return results


def _revise_role_domain(
    role_domain_values: list[str],
    domain_values: list[str],
    role_to_domain: dict[str, str],
) -> tuple[list[str], bool]:
    """AC-3 revise step for arc: role -> domain."""
    allowed_domains = set(domain_values)
    revised = [role for role in role_domain_values if role_to_domain.get(role, "") in allowed_domains]
    return revised, len(revised) != len(role_domain_values)


def _revise_domain_role(
    domain_values: list[str],
    role_domain_values: list[str],
    role_to_domain: dict[str, str],
) -> tuple[list[str], bool]:
    """AC-3 revise step for arc: domain -> role."""
    reachable_domains = {role_to_domain.get(role, "") for role in role_domain_values}
    revised = [domain for domain in domain_values if domain in reachable_domains]
    return revised, len(revised) != len(domain_values)


def _run_ac3(domains: dict[str, list[str]], role_to_domain: dict[str, str]) -> bool:
    """Run AC-3 on a two-variable CSP (role, domain)."""
    queue = deque([("role", "domain"), ("domain", "role")])

    while queue:
        left, right = queue.popleft()
        if left == "role" and right == "domain":
            revised_values, revised = _revise_role_domain(domains["role"], domains["domain"], role_to_domain)
            if revised:
                domains["role"] = revised_values
                if not domains["role"]:
                    return False
                queue.append(("domain", "role"))
        elif left == "domain" and right == "role":
            revised_values, revised = _revise_domain_role(domains["domain"], domains["role"], role_to_domain)
            if revised:
                domains["domain"] = revised_values
                if not domains["domain"]:
                    return False
                queue.append(("role", "domain"))

    return True


def _is_consistent(assignment: dict[str, str], role_to_domain: dict[str, str]) -> bool:
    role = assignment.get("role")
    domain = assignment.get("domain")
    if role is None or domain is None:
        return True
    return role_to_domain.get(role) == domain


def _backtrack_all_solutions(
    domains: dict[str, list[str]],
    role_to_domain: dict[str, str],
    limit: int,
) -> list[dict[str, str]]:
    """Enumerate CSP solutions with backtracking for explainable role/domain consistency."""
    order = sorted(domains.keys(), key=lambda var: len(domains[var]))
    solutions: list[dict[str, str]] = []

    def recurse(index: int, partial: dict[str, str]) -> None:
        if len(solutions) >= limit:
            return
        if index == len(order):
            solutions.append(dict(partial))
            return

        variable = order[index]
        for value in domains[variable]:
            partial[variable] = value
            if _is_consistent(partial, role_to_domain):
                recurse(index + 1, partial)
            partial.pop(variable, None)

    recurse(0, {})
    return solutions


def recommend_roles_from_mbti(
    mbti_type: str,
    top_k: int = 5,
    careers_path: str | Path = DEFAULT_CAREERS_PATH,
) -> list[str]:
    """Return career candidates whose `mbti_match` includes the predicted type."""
    careers = _load_careers(careers_path)

    mbti = str(mbti_type).upper().strip()
    matched = [
        item.get("name", "")
        for item in careers
        if mbti in [str(label).upper().strip() for label in item.get("mbti_match", [])]
    ]

    matched = [role for role in matched if role]
    if top_k <= 0:
        return matched
    return matched[:top_k]


def solve_career_constraints(
    mbti_type: str,
    user_skills: Iterable[str] | None = None,
    blocked_roles: set[str] | None = None,
    preferred_domains: set[str] | None = None,
    min_skill_overlap: int = 0,
    top_k: int = 10,
    careers_path: str | Path = DEFAULT_CAREERS_PATH,
) -> list[dict]:
    """Return CSP-filtered candidate careers with scoring metadata.

    Hard constraints:
    - role name not blocked
    - mbti_type must be listed in role `mbti_match`
    - optional preferred domain check
    - optional minimum overlap with required skills
    """
    careers = _load_careers(careers_path)
    if not careers:
        return []

    blocked = _normalize_set(blocked_roles)
    preferred_domain_set = _normalize_set(preferred_domains)
    user_skill_set = _normalize_set(user_skills)

    candidates = _build_candidate_records(
        careers=careers,
        mbti_type=mbti_type,
        user_skills=user_skill_set,
        blocked_roles=blocked,
        preferred_domains=preferred_domain_set,
        min_skill_overlap=min_skill_overlap,
    )
    if not candidates:
        return []

    role_to_record = {item["role"]: item for item in candidates}
    role_to_domain = {item["role"]: item["domain"] for item in candidates}

    domains = {
        "role": sorted(role_to_record.keys()),
        "domain": sorted({item["domain"] for item in candidates}),
    }

    if not _run_ac3(domains, role_to_domain):
        return []

    max_solutions = max(len(domains["role"]) * max(len(domains["domain"]), 1), 1)
    solutions = _backtrack_all_solutions(domains, role_to_domain, limit=max_solutions)
    if not solutions:
        return []

    # Keep one best entry per role from feasible assignments.
    feasible_roles = {solution["role"] for solution in solutions if "role" in solution}
    results = [role_to_record[role] for role in feasible_roles if role in role_to_record]
    results.sort(key=lambda item: item["score"], reverse=True)

    if top_k > 0:
        return results[:top_k]
    return results


def apply_constraints(candidates: Iterable[str], blocked_roles: set[str] | None = None) -> list[str]:
    """Filter out blocked roles from candidate recommendations."""
    blocked = blocked_roles or set()
    return [role for role in candidates if role not in blocked]

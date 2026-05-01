"""Module 4: A* path search for recommendation transitions."""

from __future__ import annotations

import heapq
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CAREERS_PATH = ROOT_DIR / "data" / "raw" / "careers.json"


def _load_careers(careers_path: str | Path = DEFAULT_CAREERS_PATH) -> list[dict]:
    path = Path(careers_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _build_role_index(careers: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for item in careers:
        role = str(item.get("name", "")).strip()
        if role:
            index[role] = item
    return index


def _similarity(a: dict, b: dict) -> float:
    """Compute role similarity in [0, 1] using domain, skills, and MBTI overlap."""
    a_domain = str(a.get("domain", "")).strip().lower()
    b_domain = str(b.get("domain", "")).strip().lower()
    a_skills = {str(v).strip().lower() for v in a.get("required_skills", [])}
    b_skills = {str(v).strip().lower() for v in b.get("required_skills", [])}
    a_mbti = {str(v).upper().strip() for v in a.get("mbti_match", [])}
    b_mbti = {str(v).upper().strip() for v in b.get("mbti_match", [])}

    skill_union = len(a_skills.union(b_skills))
    skill_jaccard = len(a_skills.intersection(b_skills)) / skill_union if skill_union else 0.0

    mbti_union = len(a_mbti.union(b_mbti))
    mbti_jaccard = len(a_mbti.intersection(b_mbti)) / mbti_union if mbti_union else 0.0

    domain_match = 1.0 if a_domain and a_domain == b_domain else 0.0
    return float(0.45 * domain_match + 0.35 * skill_jaccard + 0.20 * mbti_jaccard)


def _edge_cost(a: dict, b: dict) -> float:
    """Convert similarity to a strictly positive traversal cost."""
    sim = _similarity(a, b)
    return max(0.15, 1.35 - sim)


def _heuristic(current: dict, goal: dict) -> float:
    # A lower heuristic means "more similar to goal".
    return max(0.0, 1.0 - _similarity(current, goal))


def _build_graph(role_index: dict[str, dict], threshold: float = 0.20) -> dict[str, list[str]]:
    """Create sparse adjacency list connecting meaningfully related roles."""
    roles = list(role_index.keys())
    adjacency = {role: [] for role in roles}

    for i, left_role in enumerate(roles):
        left_meta = role_index[left_role]
        for right_role in roles[i + 1 :]:
            right_meta = role_index[right_role]
            if _similarity(left_meta, right_meta) >= threshold:
                adjacency[left_role].append(right_role)
                adjacency[right_role].append(left_role)

    return adjacency


def _reconstruct_path(came_from: dict[str, str], current: str) -> list[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def find_recommendation_path(
    start_role: str,
    goal_role: str,
    careers_path: str | Path = DEFAULT_CAREERS_PATH,
) -> list[str]:
    """Find a role transition path using A* over career similarity graph.

    Compatibility behavior:
    - empty inputs -> []
    - identical roles -> [role]
    - unknown role names -> [start_role, goal_role]
    """
    if not start_role or not goal_role:
        return []
    if start_role == goal_role:
        return [start_role]

    careers = _load_careers(careers_path)
    role_index = _build_role_index(careers)
    if start_role not in role_index or goal_role not in role_index:
        return [start_role, goal_role]

    adjacency = _build_graph(role_index)

    open_heap: list[tuple[float, str]] = []
    heapq.heappush(open_heap, (0.0, start_role))
    visited: set[str] = set()

    came_from: dict[str, str] = {}
    g_score: dict[str, float] = {start_role: 0.0}

    goal_meta = role_index[goal_role]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_role:
            return _reconstruct_path(came_from, current)

        current_meta = role_index[current]
        for neighbor in adjacency.get(current, []):
            neighbor_meta = role_index[neighbor]

            tentative = g_score[current] + _edge_cost(current_meta, neighbor_meta)
            if tentative >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative
            f_score = tentative + _heuristic(neighbor_meta, goal_meta)
            heapq.heappush(open_heap, (f_score, neighbor))

    # Compatibility fallback for disconnected graph components.
    return [start_role, goal_role]

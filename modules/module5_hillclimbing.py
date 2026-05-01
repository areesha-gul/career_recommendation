"""Module 5: Hill-climbing scorer for recommendation improvement."""

from __future__ import annotations

from typing import Iterable


def _objective(candidate: dict, preferred_domains: set[str]) -> float:
    """Objective score for a single recommendation candidate."""
    base = float(candidate.get("score", 0.0))
    overlap = float(candidate.get("skill_overlap_count", 0))
    domain = str(candidate.get("domain", "")).strip().lower()
    domain_bonus = 0.12 if preferred_domains and domain in preferred_domains else 0.0
    # Slightly favor richer skill coverage while keeping CSP score dominant.
    required_skills = candidate.get("required_skills") or []
    skill_density = overlap / max(len(required_skills), 1)
    return base + 0.03 * overlap + 0.05 * skill_density + domain_bonus


def _ranked_total_score(items: list[dict], preferred_domains: set[str]) -> float:
    """Score an ordering, giving higher weight to earlier positions."""
    score = 0.0
    seen_domains: dict[str, int] = {}
    for rank, item in enumerate(items):
        rank_weight = 1.0 / (1.0 + rank)
        domain = str(item.get("domain", "")).strip().lower()
        repetition_penalty = 0.01 * seen_domains.get(domain, 0)
        score += rank_weight * _objective(item, preferred_domains) - repetition_penalty
        seen_domains[domain] = seen_domains.get(domain, 0) + 1
    return score


def _generate_neighbors(ordering: list[dict]) -> list[list[dict]]:
    """Create local neighbors via pair-swap and insertion moves."""
    neighbors: list[list[dict]] = []
    n_items = len(ordering)

    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            swapped = ordering[:]
            swapped[i], swapped[j] = swapped[j], swapped[i]
            neighbors.append(swapped)

    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            moved = ordering[:]
            item = moved.pop(i)
            moved.insert(j, item)
            neighbors.append(moved)

    return neighbors


def optimize_recommendations_hill_climbing(
    candidates: Iterable[dict],
    preferred_domains: set[str] | None = None,
    max_iterations: int = 200,
) -> list[dict]:
    """Optimize candidate ordering with a simple hill-climbing swap search."""
    items = [dict(item) for item in candidates]
    if len(items) <= 1:
        return items

    preferred = {str(v).strip().lower() for v in (preferred_domains or set())}

    current = items[:]
    current_score = _ranked_total_score(current, preferred)

    for _ in range(max_iterations):
        best_neighbor = None
        best_neighbor_score = current_score

        for neighbor in _generate_neighbors(current):
            neighbor_score = _ranked_total_score(neighbor, preferred)
            if neighbor_score > best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score

        if best_neighbor is None:
            break

        current = best_neighbor
        current_score = best_neighbor_score

    return current


def improve_recommendation(initial_score: float, step: float = 0.05, max_score: float = 1.0) -> float:
    """Return an improved score using a simple hill-climbing step."""
    if initial_score >= max_score:
        return max_score
    return min(max_score, initial_score + max(step, 0.0))

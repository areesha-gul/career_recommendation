from modules.module4_astar import find_recommendation_path


def test_find_recommendation_path_returns_direct_path():
    path = find_recommendation_path("Student", "Data Analyst")
    assert path == ["Student", "Data Analyst"]

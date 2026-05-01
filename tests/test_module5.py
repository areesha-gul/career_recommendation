from modules.module5_hillclimbing import improve_recommendation


def test_improve_recommendation_increases_score():
    improved = improve_recommendation(0.4, step=0.1)
    assert improved > 0.4
    assert improved <= 1.0

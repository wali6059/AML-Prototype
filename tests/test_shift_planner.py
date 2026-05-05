import pandas as pd

from tip_or_skip.shift_planner import rank_destinations


def test_rank_destinations_orders_by_risk_neutral_score():
    candidates = pd.DataFrame(
        [
            {"dropoff_zone": "A", "expected_tip": 2.5, "q10_tip": 1.0, "tip_probability": 0.8, "observed_trips": 100},
            {"dropoff_zone": "B", "expected_tip": 3.0, "q10_tip": 0.5, "tip_probability": 0.7, "observed_trips": 80},
        ]
    )

    ranked = rank_destinations(candidates, objective="risk_neutral", top_k=2)

    assert ranked.iloc[0]["dropoff_zone"] == "B"
    assert ranked.iloc[0]["score"] == 3.0


def test_rank_destinations_orders_by_risk_averse_score():
    candidates = pd.DataFrame(
        [
            {"dropoff_zone": "A", "expected_tip": 2.5, "q10_tip": 1.0, "tip_probability": 0.8, "observed_trips": 100},
            {"dropoff_zone": "B", "expected_tip": 3.0, "q10_tip": 0.5, "tip_probability": 0.7, "observed_trips": 80},
        ]
    )

    ranked = rank_destinations(candidates, objective="risk_averse", top_k=2)

    assert ranked.iloc[0]["dropoff_zone"] == "A"
    assert ranked.iloc[0]["score"] == 1.0


def test_rank_destinations_accepts_final_report_probability_column():
    candidates = pd.DataFrame(
        [
            {"dropoff_zone": "A", "expected_tip": 2.5, "q10_tip": 1.0, "predicted_tip_probability": 0.8},
            {"dropoff_zone": "B", "expected_tip": 3.0, "q10_tip": 0.5, "predicted_tip_probability": 0.9},
        ]
    )

    ranked = rank_destinations(candidates, objective="probability", top_k=2)

    assert ranked.iloc[0]["dropoff_zone"] == "B"
    assert ranked.iloc[0]["score"] == 0.9

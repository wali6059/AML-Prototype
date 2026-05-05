import pandas as pd

from tip_or_skip.extra import (
    ablation_columns,
    calibration_table,
    flow_features,
    llm_examples,
    score_answers,
    sequence_rows,
)


def _trips():
    return pd.DataFrame(
        [
            {
                "taxi_type": "yellow",
                "pickup_datetime": "2025-01-01 08:00:00",
                "PULocationID": 1,
                "DOLocationID": 2,
                "pickup_zone": "Midtown Center",
                "dropoff_zone": "JFK Airport",
                "pickup_hour": 8,
                "tip_given": 1,
                "tip_amount": 6.0,
            },
            {
                "taxi_type": "yellow",
                "pickup_datetime": "2025-01-01 09:00:00",
                "PULocationID": 1,
                "DOLocationID": 3,
                "pickup_zone": "Midtown Center",
                "dropoff_zone": "LaGuardia Airport",
                "pickup_hour": 9,
                "tip_given": 0,
                "tip_amount": 0.0,
            },
            {
                "taxi_type": "yellow",
                "pickup_datetime": "2025-01-01 10:00:00",
                "PULocationID": 2,
                "DOLocationID": 1,
                "pickup_zone": "JFK Airport",
                "dropoff_zone": "Midtown Center",
                "pickup_hour": 10,
                "tip_given": 1,
                "tip_amount": 8.0,
            },
        ]
    )


def test_ablation_columns_drop_groups():
    cols = ["pickup_zone", "dropoff_zone", "pickup_hour", "fare_amount", "trip_distance"]

    out = ablation_columns(cols, "no_zones")

    assert out == ["pickup_hour", "fare_amount", "trip_distance"]


def test_calibration_table_bins_probabilities():
    out = calibration_table([0, 1, 1, 0], [0.1, 0.2, 0.8, 0.9], bins=2)

    assert list(out["rows"]) == [2, 2]
    assert out.iloc[0]["actual"] == 0.5
    assert out.iloc[1]["predicted"] == 0.85


def test_flow_features_counts_pickup_and_dropoff():
    out = flow_features(_trips())

    midtown = out[out["zone_id"] == 1].iloc[0]
    assert midtown["pickup_count"] == 2
    assert midtown["dropoff_count"] == 1
    assert midtown["out_degree"] == 2


def test_sequence_rows_make_next_hour_targets():
    out = sequence_rows(_trips())

    assert {"tip_rate_next", "avg_tip_next"}.issubset(out.columns)
    assert len(out) == 2
    assert out.iloc[0]["hour"] == 8


def test_score_answers_checks_recommendation_and_numbers():
    cases = pd.DataFrame(
        [
            {
                "question": "JFK or LaGuardia?",
                "expected_zone": "JFK Airport",
                "expected_tip": 6.15,
                "must_include": "cash tips",
            }
        ]
    )
    answers = ["Choose JFK Airport. Expected tip is $6.15. Cash tips are not observed."]

    out = score_answers(cases, answers)

    assert out.iloc[0]["has_zone"] == 1
    assert out.iloc[0]["has_number"] == 1
    assert out.iloc[0]["has_caveat"] == 1


def test_llm_examples_build_grounded_messages():
    zones = pd.DataFrame(
        [
            {"pickup_zone": "JFK Airport", "expected_tip": 6.15, "q10_tip": 6.38, "predicted_tip_probability": 0.51, "rows": 100},
            {"pickup_zone": "LaGuardia Airport", "expected_tip": 5.84, "q10_tip": 5.66, "predicted_tip_probability": 0.59, "rows": 100},
        ]
    )

    examples = llm_examples(zones, limit=2)

    assert len(examples) == 2
    assert "messages" in examples[0]
    assert "JFK Airport" in examples[0]["messages"][-1]["content"]

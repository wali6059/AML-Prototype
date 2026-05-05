import pandas as pd

from tip_or_skip.driver_copilot import (
    answer_driver_question,
    build_driver_context,
    compare_zone_options,
    extract_zone_mentions,
)


def _context():
    zones = pd.DataFrame(
        [
            {
                "taxi_type": "yellow",
                "pickup_borough": "Queens",
                "pickup_zone": "JFK Airport",
                "rows": 16481,
                "expected_tip": 6.15,
                "q10_tip": 6.38,
                "predicted_tip_probability": 0.51,
            },
            {
                "taxi_type": "yellow",
                "pickup_borough": "Queens",
                "pickup_zone": "LaGuardia Airport",
                "rows": 12815,
                "expected_tip": 5.84,
                "q10_tip": 5.66,
                "predicted_tip_probability": 0.59,
            },
            {
                "taxi_type": "yellow",
                "pickup_borough": "Manhattan",
                "pickup_zone": "Battery Park City",
                "rows": 2219,
                "expected_tip": 2.81,
                "q10_tip": 2.55,
                "predicted_tip_probability": 0.62,
            },
        ]
    )
    return build_driver_context(zones, ["JFK Airport", "LaGuardia Airport", "Battery Park City", "Midtown Center"])


def test_extract_zone_mentions_uses_option_clause_when_present():
    mentions = extract_zone_mentions(
        "I am at Midtown Center and got ride options in JFK Airport and LaGuardia Airport.",
        _context(),
    )

    assert mentions == ["JFK Airport", "LaGuardia Airport"]


def test_compare_zone_options_recommends_higher_expected_tip():
    ranked = compare_zone_options(["LaGuardia Airport", "JFK Airport"], _context(), taxi_type="yellow")

    assert ranked.iloc[0]["pickup_zone"] == "JFK Airport"
    assert ranked.iloc[0]["expected_tip"] == 6.15


def test_driver_question_compares_two_ride_options():
    answer = answer_driver_question(
        "I'm at Midtown Center. I got two ride options: JFK Airport or LaGuardia Airport. Which should I choose?",
        _context(),
    )

    assert "JFK Airport" in answer
    assert "LaGuardia Airport" in answer
    assert "$6.15" in answer
    assert "recommend" in answer.lower()


def test_driver_question_assesses_single_area_quality():
    answer = answer_driver_question("Is Battery Park City likely to earn me a good tip?", _context())

    assert "Battery Park City" in answer
    assert "$2.81" in answer
    assert "tip" in answer.lower()

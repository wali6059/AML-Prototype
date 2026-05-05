import pandas as pd

from tip_or_skip.fact_assistant import build_fact_context, answer_question, fact_cards_markdown


def _context():
    summary = {
        "rows": 1_008_000,
        "columns": 46,
        "tip_rate": 0.9289365,
        "split_counts": {"train": 378_000, "valid": 126_000, "test": 504_000},
        "taxi_type_counts": {"yellow": 720_000, "green": 288_000},
    }
    metrics = pd.DataFrame(
        [
            {"model": "tree_hurdle", "class_roc_auc": 0.77, "expected_tip_mae": 1.45},
            {
                "model": "transformer_mdn",
                "class_roc_auc": 0.73,
                "expected_tip_mae": 2.34,
                "interval80_coverage": 0.74,
            },
        ]
    )
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
                "pickup_borough": "Manhattan",
                "pickup_zone": "Battery Park City",
                "rows": 2219,
                "expected_tip": 2.81,
                "q10_tip": 2.55,
                "predicted_tip_probability": 0.62,
            },
        ]
    )
    subgroup = pd.DataFrame(
        [
            {
                "taxi_type": "yellow",
                "pickup_borough": "Manhattan",
                "rows": 319329,
                "roc_auc": 0.63,
                "f1": 0.96,
            }
        ]
    )
    return build_fact_context(summary, metrics, zones, subgroup)


def test_fact_cards_include_dataset_and_best_model():
    cards = fact_cards_markdown(_context())

    assert "1,008,000" in cards
    assert "tree_hurdle" in cards
    assert "credit-card" in cards


def test_answer_question_handles_dataset_questions():
    answer = answer_question("What dataset did you use?", _context())

    assert "1,008,000" in answer
    assert "credit-card" in answer
    assert "2025" in answer


def test_answer_question_handles_model_questions():
    answer = answer_question("Which model performed best?", _context())

    assert "tree_hurdle" in answer
    assert "$1.45" in answer
    assert "Transformer-MDN" in answer


def test_answer_question_handles_zone_questions():
    answer = answer_question("Where are the highest tipping zones in Queens?", _context())

    assert "JFK Airport" in answer
    assert "$6.15" in answer

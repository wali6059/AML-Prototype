import app
from pathlib import Path


def test_compare_ride_options_accepts_weekday_label_from_ui():
    summary, table = app.compare_ride_options(
        "yellow",
        "Midtown Center",
        "JFK Airport",
        "LaGuardia Airport",
        18,
        "Friday",
        5,
        17.0,
        72.0,
        45.0,
        11.0,
        54.0,
        35.0,
    )

    assert "Recommendation" in summary
    assert len(table) == 2
    assert set(table["Ride option"]) == {"Option A", "Option B"}


def test_map_table_can_show_multiple_boroughs():
    table = app.map_zone_table("yellow", "All boroughs", "Tip Rate")

    assert "Borough" in table.columns
    assert table["Borough"].nunique() > 1


def test_report_generator_uses_experiment_framing():
    text = Path("scripts/generate_latex_report.py").read_text(encoding="utf-8").lower()

    assert ("a" + "+ stretch") not in text
    assert ("stretch " + "artifact") not in text
    assert ("to push the project " + "further") not in text

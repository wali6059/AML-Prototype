import app


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

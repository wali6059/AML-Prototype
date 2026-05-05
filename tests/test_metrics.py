import numpy as np

from tip_or_skip.metrics import binary_calibration_error, interval_coverage, regression_summary


def test_binary_calibration_error_is_zero_for_perfect_bins():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    assert binary_calibration_error(y_true, y_prob, n_bins=2) == 0.0


def test_interval_coverage_counts_inside_bounds():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 1.5, 2.5, 4.5])
    upper = np.array([1.5, 2.5, 3.5, 5.0])

    result = interval_coverage(y_true, lower, upper)

    assert result["coverage"] == 0.75
    assert result["mean_width"] == 1.0


def test_regression_summary_returns_mae_and_rmse():
    result = regression_summary(np.array([1.0, 2.0]), np.array([2.0, 2.0]))

    assert result["mae"] == 0.5
    assert result["rmse"] == np.sqrt(0.5)


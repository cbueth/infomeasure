"""Test for the p-value calculation."""

import numpy as np
import pytest

from tests.conftest import discrete_random_variables
from infomeasure.estimators.base import PValueMixin


@pytest.mark.parametrize("n_tests", [2, 5, 300])
def test_mutual_information_p_value(mi_estimator, n_tests):
    """Test the p-value calculation for mutual information."""
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator, kwargs = mi_estimator
    estimator = estimator(data_x, data_y, **kwargs)
    p_value = estimator.p_value(n_tests)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert isinstance(estimator.t_score(n_tests), float)


@pytest.mark.parametrize("n_tests", [2, 5, 300])
def test_mi_t_score(mi_estimator, n_tests):
    """Test only accessing the t-score, without the p-value, for MI."""
    source = np.arange(10)
    dest = np.arange(10)
    estimator, kwargs = mi_estimator
    estimator = estimator(source, dest, **kwargs)
    t_score = estimator.t_score(n_tests)
    assert isinstance(t_score, float)


@pytest.mark.parametrize("n_tests", [2, 5, 50])
def test_transfer_entropy_p_value(te_estimator, n_tests):
    """Test the p-value calculation for transfer entropy."""
    source, dest = discrete_random_variables(0, prop_time=0, length=100)
    estimator, kwargs = te_estimator
    estimator = estimator(source, dest, **kwargs)
    p_value = estimator.p_value(n_tests)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert isinstance(estimator.t_score(n_tests), float)
    # calling the functions again without parameters should return the cached result
    assert np.isclose(
        estimator.p_value(n_tests), estimator.p_value(), rtol=0, atol=0, equal_nan=True
    )
    assert np.isclose(
        estimator.t_score(n_tests), estimator.t_score(), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("n_tests", [2, 5, 25])
def test_mi_t_score(te_estimator, n_tests):
    """Test only accessing the t-score, without the p-value, for TE."""
    source, dest = discrete_random_variables(0, prop_time=0, length=100)
    estimator, kwargs = te_estimator
    estimator = estimator(source, dest, **kwargs)
    t_score = estimator.t_score(n_tests)
    assert isinstance(t_score, float)
    # calling the functions again without parameters should return the cached result
    assert np.isclose(t_score, estimator.t_score(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "observed_value, test_values, p_value, t_score",
    [
        (0.5, [0.1, 0.2, 0.3, 0.4, 0.5], 0.0, 1.2649110640),
        (0.25, [0.1, 0.2, 0.3, 0.4, 0.5], 0.6, -0.316227766),
        (0.3, [0.1, 0.2, 0.3, 0.4, 0.5], 0.4, 0.0),
        (0.35, [0.1, 0.2, 0.3, 0.4, 0.5], 0.4, 0.3162277660),
        (0.1, [0.1, 0.2, 0.3, 0.4, 0.5], 0.8, -1.264911064),
        (0.0, [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, -1.897366596),
        (0.09, [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, -1.328156617),
        (1.0, [-1, 1], 0.0, 0.707106781),
        (0.0, [-2, 0], 0.0, 0.707106781),
        (1.0, [1, 1], 0, np.nan),  # Not enough variance in test_values
        (0.0, [2.0] * 10, 1.0, np.nan),  # Not enough variance in test_values
    ],
)
def test_p_value_t_score(observed_value, test_values, p_value, t_score):
    """Test the p-value and t-score calculation."""
    result = PValueMixin._p_value_t_score(observed_value, test_values)
    assert result[0] == pytest.approx(p_value, abs=1e-6)
    if not np.isnan(t_score):
        assert result[1] == pytest.approx(t_score, abs=1e-6)
    else:
        assert np.isnan(result[1])


@pytest.mark.parametrize("n_test_vals", [0, 1])
def test_p_value_t_score_not_enough_test_values(n_test_vals):
    """Test the p-value and t-score calculation with not enough test values."""
    with pytest.raises(ValueError, match="Not enough test values for statistical test"):
        PValueMixin._p_value_t_score(1.0, [1] * n_test_vals)

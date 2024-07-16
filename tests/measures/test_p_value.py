"""Test for the p-value calculation."""

import numpy as np
import pytest


@pytest.mark.parametrize("num_permutations", [2, 5, 300])
def test_entropy_p_value(entropy_estimator, num_permutations):
    """Test the p-value calculation for entropy."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator, kwargs = entropy_estimator
    estimator = estimator(data, **kwargs)
    p_value = estimator.p_value(num_permutations)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


@pytest.mark.parametrize("num_permutations", [2, 5, 300])
def test_mutual_information_p_value(mi_estimator, num_permutations):
    """Test the p-value calculation for mutual information."""
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator, kwargs = mi_estimator
    estimator = estimator(data_x, data_y, **kwargs)
    p_value = estimator.p_value(num_permutations)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


@pytest.mark.parametrize("num_permutations", [2, 5, 300])
def test_transfer_entropy_p_value(te_estimator, num_permutations):
    """Test the p-value calculation for transfer entropy."""
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator, kwargs = te_estimator
    estimator = estimator(source, dest, **kwargs)
    p_value = estimator.p_value(num_permutations)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1

"""Miller-Madow mutual information estimator tests."""

import pytest
from numpy import e, log, array

import infomeasure as im
from infomeasure.estimators.mutual_information import (
    MillerMadowMIEstimator,
    MillerMadowCMIEstimator,
)
from tests.conftest import (
    discrete_random_variables,
    discrete_random_variables_condition,
)


@pytest.mark.parametrize(
    "data_x, data_y, base, expected_correction",
    [
        (
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            2,
            0.0,
        ),  # K_x=1, K_y=1, K_xy=1, correction=0 (degenerate case)
        (
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            2,
            (2 + 2 - 2 - 2 + 1) / (2 * 4) / log(2),
        ),  # Perfect correlation: K_x=2, K_y=2, K_xy=2
        (
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            2,
            (2 + 2 - 2 - 2 + 1) / (2 * 4) / log(2),
        ),  # Anti-correlation: K_x=2, K_y=2, K_xy=2
        (
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            2,
            (4 + 4 - 2 - 4 + 1) / (2 * 4) / log(2),
        ),  # Perfect correlation: K_x=4, K_y=4, K_xy=4
        (
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            2,
            (4 + 4 - 2 - 4 + 1) / (2 * 4) / log(2),
        ),  # Perfect anti-correlation: K_x=4, K_y=4, K_xy=4
    ],
)
def test_miller_madow_mi_basic(data_x, data_y, base, expected_correction):
    """Test basic Miller-Madow mutual information estimation."""
    # Calculate MI without correction
    mi_regular = im.mutual_information(data_x, data_y, approach="discrete", base=base)

    # Calculate MI with Miller-Madow correction
    mi_mm = im.mutual_information(data_x, data_y, approach="miller_madow", base=base)

    # The difference should be approximately the expected correction
    assert mi_mm - mi_regular == pytest.approx(expected_correction, abs=1e-10)


@pytest.mark.parametrize(
    "data_x, data_y, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 2),
        ([1, 0, 1, 0], [0, 1, 0, 1], 10),
        ([1, 2, 3, 4], [1, 2, 3, 4], "e"),
        (["a", "b", "a", "b"], ["x", "y", "x", "y"], 2),
    ],
)
def test_miller_madow_mi_estimator(data_x, data_y, base):
    """Test MillerMadowMIEstimator class."""
    estimator = MillerMadowMIEstimator(data_x, data_y, base=base)
    result = estimator.result()

    # Should return a float
    assert isinstance(result, float)
    # Should be non-negative for MI
    assert result >= 0.0

    # Test local values
    local_vals = estimator.local_vals()
    assert len(local_vals) == len(data_x)


@pytest.mark.parametrize(
    "data_x, data_y, offset, base",
    [
        ([1, 0, 1, 0, 1], [1, 0, 1, 0, 1], 1, 2),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 2, 10),
    ],
)
def test_miller_madow_mi_offset(data_x, data_y, offset, base):
    """Test Miller-Madow MI with offset."""
    estimator = MillerMadowMIEstimator(data_x, data_y, offset=offset, base=base)
    result = estimator.result()

    assert isinstance(result, float)
    assert result >= 0.0


@pytest.mark.parametrize(
    "data, base",
    [
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), 2),
        (([1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]), 10),
        (([1, 2, 1, 2], [2, 1, 2, 1], [1, 1, 2, 2]), "e"),
    ],
)
def test_miller_madow_mi_3_vars(data, base):
    """Test Miller-Madow MI with 3 variables."""
    estimator = MillerMadowMIEstimator(*data, base=base)
    result = estimator.result()

    assert isinstance(result, float)
    # Interaction information can be negative

    # Test local values
    local_vals = estimator.local_vals()
    assert len(local_vals) == len(data[0])


@pytest.mark.parametrize(
    "data_x, data_y, cond, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], 2),
        ([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2], 10),
        (["a", "b", "a", "b"], ["x", "y", "x", "y"], [1, 1, 2, 2], "e"),
    ],
)
def test_miller_madow_cmi(data_x, data_y, cond, base):
    """Test Miller-Madow conditional mutual information."""
    estimator = MillerMadowCMIEstimator(data_x, data_y, cond=cond, base=base)
    result = estimator.result()

    assert isinstance(result, float)
    assert result >= 0.0

    # Test local values
    local_vals = estimator.local_vals()
    assert len(local_vals) == len(data_x)


@pytest.mark.parametrize(
    "data, cond, base",
    [
        (([1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]), [1, 1, 0, 0], 2),
        (([1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]), [1, 1, 2, 2], 10),
    ],
)
def test_miller_madow_cmi_multiple_vars(data, cond, base):
    """Test Miller-Madow CMI with multiple variables."""
    estimator = MillerMadowCMIEstimator(*data, cond=cond, base=base)
    result = estimator.result()

    assert isinstance(result, float)

    # Test local values
    local_vals = estimator.local_vals()
    assert len(local_vals) == len(data[0])


def test_miller_madow_mi_vs_discrete():
    """Test that Miller-Madow MI is close to discrete MI for large samples."""
    # Generate larger dataset where correction should be smaller
    data_x = [i % 10 for i in range(1000)]
    data_y = [(i + 1) % 10 for i in range(1000)]

    mi_discrete = im.mutual_information(data_x, data_y, approach="discrete", base=2)
    mi_mm = im.mutual_information(data_x, data_y, approach="miller_madow", base=2)

    # For large samples, the correction should be small
    assert abs(mi_mm - mi_discrete) < 0.1


def test_miller_madow_cmi_vs_discrete():
    """Test that Miller-Madow CMI is close to discrete CMI for large samples."""
    # Generate larger dataset where correction should be smaller
    data_x = [i % 5 for i in range(1000)]
    data_y = [(i + 1) % 5 for i in range(1000)]
    cond = [i % 3 for i in range(1000)]

    cmi_discrete = im.conditional_mutual_information(
        data_x, data_y, cond=cond, approach="discrete", base=2
    )
    cmi_mm = im.conditional_mutual_information(
        data_x, data_y, cond=cond, approach="miller_madow", base=2
    )

    # For large samples, the correction should be small
    assert abs(cmi_mm - cmi_discrete) < 0.1


@pytest.mark.parametrize(
    "rng_int,method,p_mi,p_cmi",
    [
        (1, "permutation_test", 0.74, 0.98),
        (1, "bootstrap", 0.82, 0.92),
        (2, "permutation_test", 0.88, 0.86),
        (2, "bootstrap", 0.86, 0.84),
        (3, "permutation_test", 0.98, 0.96),
        (3, "bootstrap", 1.0, 0.9),
        (4, "permutation_test", 0.86, 0.82),
        (4, "bootstrap", 0.78, 0.84),
    ],
)
def test_miller_madow_mi_statistical_test(rng_int, method, p_mi, p_cmi):
    """Test the Miller-Madow MI for p-values. Fix rng."""
    data_x, data_y, cond = discrete_random_variables_condition(rng_int)
    est_mi = MillerMadowMIEstimator(data_x, data_y, base=2, seed=8)
    est_cmi = MillerMadowCMIEstimator(data_x, data_y, cond=cond, base=2, seed=8)
    test = est_mi.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_mi)
    test = est_cmi.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cmi)

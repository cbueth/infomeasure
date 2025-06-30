"""Tests for discrete interaction information functions."""

import pytest
from numpy import e, log, array

from infomeasure.estimators.utils.discrete_interaction_information import (
    millermadow_mi_corr,
    mutual_information_global,
    mutual_information_local,
    conditional_mutual_information_global,
    conditional_mutual_information_local,
    _mutual_information_global_2d_int,
    _mutual_information_global_nd_int,
    _mutual_information_global_nd_other,
)


@pytest.mark.parametrize(
    "k_i, k_joint, n, base, expected",
    [
        (
            [2, 2],
            2,
            4,
            2,
            (2 + 2 - 2 - 2 + 1) / (2 * 4) / log(2),
        ),  # Simple case: sum(k_i) - len(k_i) - k_joint + 1
        ([3, 3], 4, 6, 10, (3 + 3 - 2 - 4 + 1) / (2 * 6) / log(10)),  # Different base
        ([2, 2], 2, 4, "e", (2 + 2 - 2 - 2 + 1) / (2 * 4)),  # Natural log
        ([4, 4], 4, 8, 2, (4 + 4 - 2 - 4 + 1) / (2 * 8) / log(2)),  # Larger case
    ],
)
def test_millermadow_mi_corr(k_i, k_joint, n, base, expected):
    """Test the Miller-Madow correction function."""
    result = millermadow_mi_corr(k_i, k_joint, n, base)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "k_i, k_joint, n, base, k_cond, expected",
    [
        (
            [2, 2],
            2,
            4,
            2,
            2,
            (2 + 2 - 2 - 2 + 1 - (2 - 1)) / (2 * 4) / log(2),
        ),  # With condition: sum(k_i) - len(k_i) - k_joint + 1 - (k_cond - 1)
        (
            [3, 3],
            4,
            6,
            10,
            3,
            (3 + 3 - 2 - 4 + 1 - (3 - 1)) / (2 * 6) / log(10),
        ),  # Different base
    ],
)
def test_millermadow_mi_corr_conditional(k_i, k_joint, n, base, k_cond, expected):
    """Test the Miller-Madow correction function with conditional variable."""
    result = millermadow_mi_corr(k_i, k_joint, n, base, k_cond=k_cond)
    assert result == pytest.approx(expected)


def test_millermadow_mi_corr_invalid_base():
    """Test that invalid base raises error."""
    # The millermadow_mi_corr function doesn't validate base, it just tries to use it
    # This will raise a TypeError when trying to compute log("invalid")
    with pytest.raises(TypeError):
        millermadow_mi_corr([2, 2], 2, 4, "invalid")


@pytest.mark.parametrize(
    "data_x, data_y, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 2),
        ([1, 2, 3, 4], [4, 3, 2, 1], 10),
        (["a", "b", "a", "b"], ["x", "y", "x", "y"], "e"),
    ],
)
def test_mutual_information_global_with_correction(data_x, data_y, base):
    """Test mutual_information_global function with Miller-Madow correction."""
    # Convert to numpy arrays
    data_x = array(data_x)
    data_y = array(data_y)

    # Without correction
    mi_no_corr = mutual_information_global(data_x, data_y)

    # With correction
    if base == "e":
        mi_with_corr = mutual_information_global(
            data_x, data_y, miller_madow_correction="e"
        )
    else:
        mi_with_corr = mutual_information_global(
            data_x, data_y, miller_madow_correction=base
        )

    # Correction should make a difference (unless data is degenerate)
    if len(set(data_x)) > 1 and len(set(data_y)) > 1:
        assert mi_with_corr != mi_no_corr


@pytest.mark.parametrize(
    "data_x, data_y, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 2),
        ([1, 2, 3, 4], [4, 3, 2, 1], 10),
    ],
)
def test_mutual_information_local_with_correction(data_x, data_y, base):
    """Test mutual_information_local function with Miller-Madow correction."""
    # Convert to numpy arrays
    data_x = array(data_x)
    data_y = array(data_y)

    # Without correction
    mi_local_no_corr = mutual_information_local(data_x, data_y)

    # With correction
    mi_local_with_corr = mutual_information_local(
        data_x, data_y, miller_madow_correction=base
    )

    assert len(mi_local_no_corr) == len(data_x)
    assert len(mi_local_with_corr) == len(data_x)

    # Check that correction affects the values
    if len(set(data_x)) > 1 and len(set(data_y)) > 1:
        assert not all(a == b for a, b in zip(mi_local_no_corr, mi_local_with_corr))


@pytest.mark.parametrize(
    "data_x, data_y, cond, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], 2),
        ([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2], 10),
    ],
)
def test_conditional_mutual_information_global_with_correction(
    data_x, data_y, cond, base
):
    """Test conditional_mutual_information_global function with Miller-Madow correction."""
    # Convert to numpy arrays
    data_x = array(data_x)
    data_y = array(data_y)
    cond = array(cond)

    # Without correction
    cmi_no_corr = conditional_mutual_information_global(data_x, data_y, cond=cond)

    # With correction
    cmi_with_corr = conditional_mutual_information_global(
        data_x, data_y, cond=cond, miller_madow_correction=base
    )

    assert isinstance(cmi_no_corr, float)
    assert isinstance(cmi_with_corr, float)

    # Correction should make a difference (unless data is degenerate)
    if len(set(data_x)) > 1 and len(set(data_y)) > 1:
        assert cmi_with_corr != cmi_no_corr


@pytest.mark.parametrize(
    "data_x, data_y, cond, base",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], 2),
        ([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2], 10),
    ],
)
def test_conditional_mutual_information_local_with_correction(
    data_x, data_y, cond, base
):
    """Test conditional_mutual_information_local function with Miller-Madow correction."""
    # Convert to numpy arrays
    data_x = array(data_x)
    data_y = array(data_y)
    cond = array(cond)

    # Without correction
    cmi_local_no_corr = conditional_mutual_information_local(data_x, data_y, cond=cond)

    # With correction
    cmi_local_with_corr = conditional_mutual_information_local(
        data_x, data_y, cond=cond, miller_madow_correction=base
    )

    assert len(cmi_local_no_corr) == len(data_x)
    assert len(cmi_local_with_corr) == len(data_x)

    # Check that correction affects the values
    if len(set(data_x)) > 1 and len(set(data_y)) > 1:
        assert not all(a == b for a, b in zip(cmi_local_no_corr, cmi_local_with_corr))


def test_mutual_information_global_error_conditions():
    """Test error conditions for mutual_information_global."""
    data_x = [1, 0, 1, 0]
    data_y = [1, 0, 1, 0]

    # Test invalid miller_madow_correction parameter
    with pytest.raises(ValueError):
        mutual_information_global(data_x, data_y, miller_madow_correction="invalid")


def test_conditional_mutual_information_global_error_conditions():
    """Test error conditions for conditional_mutual_information_global."""
    data_x = [1, 0, 1, 0]
    data_y = [1, 0, 1, 0]
    cond = [1, 1, 0, 0]

    # Test invalid miller_madow_correction parameter
    with pytest.raises(ValueError):
        conditional_mutual_information_global(
            data_x, data_y, cond=cond, miller_madow_correction="invalid"
        )


# Tests for equivalence between different MI global functions
@pytest.mark.parametrize(
    "data_x, data_y",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0]),  # 2D integer data
        ([1, 2, 3, 4], [4, 3, 2, 1]),  # 2D integer data
        ([0, 1, 2, 0, 1], [2, 1, 0, 2, 1]),  # 2D integer data
    ],
)
def test_mi_global_2d_vs_nd_int_equivalence(data_x, data_y):
    """Test that _mutual_information_global_2d_int and _mutual_information_global_nd_int give same results for 2D data."""
    data_x = array(data_x)
    data_y = array(data_y)

    # Test without correction
    mi_2d = _mutual_information_global_2d_int(data_x, data_y)
    mi_nd = _mutual_information_global_nd_int(data_x, data_y)

    assert mi_2d == pytest.approx(mi_nd, abs=1e-10)

    # Test with correction
    mi_2d_corr = _mutual_information_global_2d_int(
        data_x, data_y, miller_madow_correction=2
    )
    mi_nd_corr = _mutual_information_global_nd_int(
        data_x, data_y, miller_madow_correction=2
    )

    assert mi_2d_corr == pytest.approx(mi_nd_corr, abs=1e-10)


def test_mi_global_dispatch_logic():
    """Test that mutual_information_global correctly dispatches to the right implementation."""
    # Test 1D data with 2 variables -> should use _mutual_information_global_2d_int
    data_x_1d = array([1, 0, 1, 0])
    data_y_1d = array([1, 0, 1, 0])

    mi_global = mutual_information_global(data_x_1d, data_y_1d)
    mi_2d_direct = _mutual_information_global_2d_int(data_x_1d, data_y_1d)

    assert mi_global == pytest.approx(mi_2d_direct, abs=1e-10)

    # Test 1D data with 3 variables -> should use _mutual_information_global_nd_int
    data_z_1d = array([0, 1, 0, 1])

    mi_global_3d = mutual_information_global(data_x_1d, data_y_1d, data_z_1d)
    mi_nd_direct = _mutual_information_global_nd_int(data_x_1d, data_y_1d, data_z_1d)

    assert mi_global_3d == pytest.approx(mi_nd_direct, abs=1e-10)

    # Test multidimensional data -> should use _mutual_information_global_nd_other
    data_x_2d = array([[1, 2], [0, 1], [1, 2], [0, 1]])
    data_y_2d = array([[3, 4], [2, 3], [3, 4], [2, 3]])

    mi_global_2d = mutual_information_global(data_x_2d, data_y_2d)
    mi_nd_other_direct = _mutual_information_global_nd_other(data_x_2d, data_y_2d)

    assert mi_global_2d == pytest.approx(mi_nd_other_direct, abs=1e-10)


def test_mi_global_functions_with_multidimensional_data():
    """Test that _mutual_information_global_nd_int and _mutual_information_global_nd_other work with multidimensional data."""
    # Create 2D data (each sample is a vector)
    data_x = array([[1, 2], [0, 1], [1, 2], [0, 1]])
    data_y = array([[3, 4], [2, 3], [3, 4], [2, 3]])

    # _mutual_information_global_nd_int should handle this
    mi_nd_int = _mutual_information_global_nd_int(data_x, data_y)
    assert isinstance(mi_nd_int, float)

    # _mutual_information_global_nd_other should also handle this
    mi_nd_other = _mutual_information_global_nd_other(data_x, data_y)
    assert isinstance(mi_nd_other, float)

    # They should give the same result for multidimensional data
    assert mi_nd_int == pytest.approx(mi_nd_other, abs=1e-10)

    # Test with correction
    mi_nd_int_corr = _mutual_information_global_nd_int(
        data_x, data_y, miller_madow_correction=2
    )
    mi_nd_other_corr = _mutual_information_global_nd_other(
        data_x, data_y, miller_madow_correction=2
    )

    assert mi_nd_int_corr == pytest.approx(mi_nd_other_corr, abs=1e-10)


def test_mi_global_functions_with_string_data():
    """Test that _mutual_information_global_nd_other works with string data."""
    data_x = array(["a", "b", "a", "b"])
    data_y = array(["x", "y", "x", "y"])

    # _mutual_information_global_nd_other should work with string data
    mi_nd_other = _mutual_information_global_nd_other(data_x, data_y)
    assert isinstance(mi_nd_other, float)
    assert mi_nd_other >= 0.0

    # Test that the main function also works with string data (should dispatch to _nd_other)
    mi_global = mutual_information_global(data_x, data_y)
    assert mi_global == pytest.approx(mi_nd_other, abs=1e-10)


def test_mi_global_functions_consistency_across_implementations():
    """Test that different implementations give consistent results for their intended data types."""
    # Test 1D integer data - compare 2D and ND implementations
    data_x = array([1, 0, 1, 0, 2, 1])
    data_y = array([1, 0, 1, 0, 2, 1])

    mi_2d = _mutual_information_global_2d_int(data_x, data_y)
    mi_nd = _mutual_information_global_nd_int(data_x, data_y)

    assert mi_2d == pytest.approx(mi_nd, abs=1e-10)

    # Test with Miller-Madow correction
    mi_2d_corr = _mutual_information_global_2d_int(
        data_x, data_y, miller_madow_correction=2
    )
    mi_nd_corr = _mutual_information_global_nd_int(
        data_x, data_y, miller_madow_correction=2
    )

    assert mi_2d_corr == pytest.approx(mi_nd_corr, abs=1e-10)

    # Test 3D data with ND implementation
    data_z = array([0, 1, 0, 1, 1, 0])

    mi_3d = _mutual_information_global_nd_int(data_x, data_y, data_z)
    assert isinstance(mi_3d, float)

    # Test with correction
    mi_3d_corr = _mutual_information_global_nd_int(
        data_x, data_y, data_z, miller_madow_correction=2
    )
    assert isinstance(mi_3d_corr, float)

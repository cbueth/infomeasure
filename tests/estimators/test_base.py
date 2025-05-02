"""Tests for the Estimator base classes."""

import pytest
from numpy import array, ndarray

from infomeasure import estimator
from infomeasure.estimators.base import (
    Estimator,
    ConditionalMutualInformationEstimator,
    EntropyEstimator,
)


class ExampleTestEstimator(Estimator):
    """Test class for Estimator."""

    def __init__(self, calc_vals, local_values=None):
        super().__init__()
        self.calc_vals = calc_vals
        self.local_values = local_values

    def _calculate(self):
        """Calculate a value."""
        return self.calc_vals

    def _extract_local_values(self) -> ndarray[float]:
        return self.local_values


@pytest.mark.parametrize("calc_vals", [array([[1, 2], [3, 4]]), "string", None])
def test_faulty_estimator_local_values(calc_vals):
    """Test estimator with local values of ndim > 1."""
    # Create an instance of the faulty estimator
    faulty_estimator = ExampleTestEstimator(calc_vals)
    with pytest.raises(RuntimeError):
        faulty_estimator.result()


def test_faulty_local_vals():
    """Test estimator when mean(local) != global."""
    faulty_estimator = ExampleTestEstimator(calc_vals=5, local_values=array([5, 6]))
    faulty_estimator.data = (1,)
    with pytest.raises(RuntimeError, match="Mean of local values"):
        faulty_estimator.local_vals()
    faulty_estimator.local_vals()  # Only raises the error once

    faulty_estimator = ExampleTestEstimator(calc_vals=5, local_values=array([5, 6]))
    faulty_estimator.data = (1, 2, 3, 4, 5, 6)
    with pytest.raises(RuntimeError, match="As you are using 6 random variables"):
        faulty_estimator.local_vals()


def test_faulty_call_not_overwritten():
    """Test estimator when _calculate() is not overwritten."""

    class FaultyEstimator(Estimator):
        pass

    with pytest.raises(TypeError, match="abstract method '?_calculate"):
        FaultyEstimator()


@pytest.mark.parametrize(
    "base, expected",
    [
        (10, 0.30102999),
        (5, 0.430676558),
        (2, 1.0),
        ("e", 0.69314718),
        (0, 0),
    ],
)
def test_log_base_function_bases(base, expected):
    """Test log_base function."""
    test_estimator = ExampleTestEstimator(calc_vals=None)
    test_estimator.base = base
    assert test_estimator._log_base(2) == pytest.approx(expected)


def test_log_base_function_negative_base():
    """Test log_base function with negative base."""
    test_estimator = ExampleTestEstimator(calc_vals=None)
    test_estimator.base = -10
    with pytest.raises(ValueError, match="Logarithm base must be positive, not -10."):
        test_estimator._log_base(2)


@pytest.mark.parametrize(
    "faulty_data",
    [
        (([1, 2, 3], "abc"),),
        ((["a", "b"], "c"),),
        ((["a", "b"], [1, 2]), "c"),
        (([1, 2, 3], [4, 5], None),),
        ((None,),),
    ],
)
def test_entropy_unsupported_data(faulty_data):
    """Test entropy function with unsupported data."""
    with pytest.raises(
        ValueError,
        match="For normal entropy, data must be a single array-like object"
        if len(faulty_data) == 1
        else "For cross-entropy, data must be two array-like objects",
    ):
        estimator(*faulty_data, measure="entropy", approach="discrete")


@pytest.mark.parametrize(
    "faulty_data",
    [
        (([1, 2, 3], [1, 2]),),
        ((["a", "b"], [1]),),
        (([1, 2, 3], [4, 5], [1, 2]),),
    ],
)
def test_entropy_inhomogenous_data(faulty_data):
    """Test entropy function with inhomogenous data."""
    with pytest.raises(
        ValueError,
        match="All elements of a joint random variable must have the same length.",
    ):
        estimator(*faulty_data, measure="entropy", approach="discrete")


def test_cross_entropy_missing_method():
    """Test error for Entropy estimators without cross-entropy support"""

    class FaultyEntropy(EntropyEstimator):
        def _simple_entropy(self):
            pass

        def _joint_entropy(self):
            pass

    est = FaultyEntropy(range(10), range(10))
    with pytest.raises(
        NotImplementedError,
        match="Cross-entropy is not implemented for FaultyEntropy.",
    ):
        est.result()


def test_mi_faulty_offset_multiple_vars():
    """Test mutual information function with offset when multiple variables."""
    with pytest.raises(
        ValueError,
        match="Offset is only supported for two data arrays.",
    ):
        estimator([1, 2], [3, 4], [5, 6], measure="mi", approach="discrete", offset=1)


def test_mi_faulty_data_lengths():
    """Test mutual information function with faulty data lengths."""
    with pytest.raises(
        ValueError,
        match=r"Data arrays must have the same first dimension, not \[2, 3\]\.",
    ):
        estimator([1, 2], [3, 4, 5], measure="mi", approach="discrete")


class FaultyCMI(ConditionalMutualInformationEstimator):
    def _calculate(self) -> float | ndarray[float]:
        pass


def test_cmi_faulty_no_condition():
    """Test conditional mutual information function with missing condition."""
    with pytest.raises(
        ValueError,
        match="Conditional data must be provided for CMI estimation.",
    ):
        FaultyCMI([1, 2], [3, 4], cond=None)


def test_cmi_faulty_offset():
    """Test conditional mutual information function with unsupported offset."""
    with pytest.raises(
        ValueError,
        match="Offset is not supported for CMI estimation.",
    ):
        FaultyCMI([1, 2], [3, 4], cond=[5, 6], offset=1)


@pytest.mark.parametrize(
    "data, cond",
    [
        [(array([1, 2]), array([3, 4])), array([5, 6, 0])],
        [(array([1, 2]), array([3, 4, 0])), array([5, 6])],
        [(array([1, 2, 0]), array([3, 4])), array([5, 6])],
    ],
)
def test_cmi_faulty_data_lengths(data, cond):
    """Test conditional mutual information function with mismatched data lengths."""
    with pytest.raises(
        ValueError,
        match="Data arrays must be of the same length,",
    ):
        FaultyCMI(*data, cond=cond)


@pytest.mark.parametrize(
    "data, cond",
    [
        [(array([[1], [2]]), array([3, 4])), array([5, 6])],
        [(array([1, 2]), array([[3], [4]])), array([5, 6])],
        [(array([1, 2]), array([3, 4])), array([[5], [6]])],
        [(array([1, 2]), array([3, 4]), array([3, 4])), array([[5], [6]])],
    ],
)
def test_cmi_faulty_normalization(data, cond):
    """Test conditional mutual information function with normalization issues."""
    with pytest.raises(
        ValueError,
        match="Data arrays must be 1D for normalization.",
    ):
        FaultyCMI(*data, cond=cond, normalize=True)

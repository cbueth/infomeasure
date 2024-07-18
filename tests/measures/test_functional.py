"""Simple tests for the functional interface of the measures."""

from io import UnsupportedOperation

import numpy as np
import pytest

import infomeasure as im
from infomeasure.measures.base import (
    EntropyEstimator,
    MutualInformationEstimator,
    TransferEntropyEstimator,
)


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("kl", {}),
    ],
)
def test_entropy_functional_addressing(approach, kwargs):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    entropy = im.entropy(data, approach=approach, **kwargs)
    assert isinstance(entropy, float)


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("kl", {}),
    ],
)
def test_entropy_class_addressing(approach, kwargs):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(data=data, measure="entropy", approach=approach, **kwargs)
    assert isinstance(est, EntropyEstimator)
    assert isinstance(est.results(), float)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.results()
    with pytest.raises(UnsupportedOperation):
        est.local_val()
    with pytest.raises(UnsupportedOperation):
        est.std_val()
    assert 0 <= est.p_value(10) <= 1
    with pytest.raises(AttributeError):
        est.effective_val()


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_mutual_information_functional_addressing(approach, kwargs):
    """Test addressing the mutual information estimator classes."""
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mi = im.mutual_information(data_x, data_y, approach=approach, **kwargs)
    assert isinstance(mi, (float, tuple))
    if isinstance(mi, tuple):
        assert len(mi) == 3
        print(f"mi: {mi}")
        assert isinstance(mi[0], float)
        assert isinstance(mi[1], np.ndarray)
        assert isinstance(mi[2], float)


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_mutual_information_class_addressing(approach, kwargs):
    """Test addressing the mutual information estimator classes."""
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 5, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        data_x=data_x,
        data_y=data_y,
        measure="mutual_information",
        approach=approach,
        **kwargs,
    )
    assert isinstance(est, MutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    if approach == "discrete":
        assert isinstance(est.results(), float)
        with pytest.raises(UnsupportedOperation):
            est.local_val()
        with pytest.raises(UnsupportedOperation):
            est.std_val()
    else:
        assert isinstance(est.results(), tuple)
        assert isinstance(est.local_val(), np.ndarray)
        assert isinstance(est.std_val(), float)
    assert 0 <= est.p_value(10) <= 1
    with pytest.raises(AttributeError):
        est.effective_val()


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {"k": 4, "l": 4, "delay": 1}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_transfer_entropy_functional_addressing(approach, kwargs):
    """Test addressing the transfer entropy estimator classes."""
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    te = im.transfer_entropy(source, dest, approach=approach, **kwargs)
    assert isinstance(te, (float, tuple))
    if isinstance(te, tuple):
        assert len(te) == 3
        assert isinstance(te[0], float)
        assert isinstance(te[1], np.ndarray)
        assert isinstance(te[2], float)


@pytest.mark.parametrize(
    "approach,kwargs",
    [
        ("discrete", {"k": 4, "l": 4, "delay": 1}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_transfer_entropy_class_addressing(approach, kwargs):
    """Test addressing the transfer entropy estimator classes."""
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        source=source,
        dest=dest,
        measure="transfer_entropy",
        approach=approach,
        **kwargs,
    )
    assert isinstance(est, TransferEntropyEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    if approach == "discrete":
        assert isinstance(est.results(), float)
        with pytest.raises(UnsupportedOperation):
            est.local_val()
        with pytest.raises(UnsupportedOperation):
            est.std_val()
    else:
        assert isinstance(est.results(), tuple)
        assert isinstance(est.local_val(), np.ndarray)
        assert isinstance(est.std_val(), float)
    assert 0 <= est.p_value(10) <= 1
    assert isinstance(est.effective_val(), float)


@pytest.mark.parametrize(
    "func", [im.entropy, im.mutual_information, im.transfer_entropy]
)
def test_functional_addressing_unknown_approach(func):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        func(data, approach="unknown")


def test_class_addressing_unknown_measure():
    """Test addressing the entropy estimator classes."""
    with pytest.raises(ValueError):
        im.estimator(measure="unknown", approach="")

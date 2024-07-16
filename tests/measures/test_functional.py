"""Simple tests for the functional interface of the measures."""

import numpy as np
import pytest

import infomeasure as im


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("kl", {}),
    ],
)
def test_entropy_addressing(estimator, kwargs):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    entropy = im.entropy(data, estimator=estimator, **kwargs)
    assert isinstance(entropy, float)


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        ("discrete", {}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_mutual_information_addressing(estimator, kwargs):
    """Test addressing the mutual information estimator classes."""
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mi = im.mutual_information(data_x, data_y, estimator=estimator, **kwargs)
    assert isinstance(mi, (float, tuple))
    if isinstance(mi, tuple):
        assert len(mi) == 3
        print(f"mi: {mi}")
        assert isinstance(mi[0], float)
        assert isinstance(mi[1], np.ndarray)
        assert isinstance(mi[2], float)


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        ("discrete", {"k": 4, "l": 4, "delay": 1}),
        ("kernel", {"bandwidth": 0.3, "kernel": "box"}),
        ("metric", {}),
        ("ksg", {}),
    ],
)
def test_transfer_entropy_addressing(estimator, kwargs):
    """Test addressing the transfer entropy estimator classes."""
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    te = im.transfer_entropy(source, dest, estimator=estimator, **kwargs)
    assert isinstance(te, (float, tuple))
    if isinstance(te, tuple):
        assert len(te) == 3
        assert isinstance(te[0], float)
        assert isinstance(te[1], np.ndarray)
        assert isinstance(te[2], float)

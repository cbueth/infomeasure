"""Simple tests for the functional interface of the estimators."""

from io import UnsupportedOperation

import numpy as np
import pytest

import infomeasure as im
from infomeasure.estimators.base import (
    EntropyEstimator,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)


def test_entropy_functional_addressing(entropy_approach):
    """Test addressing the entropy estimator classes."""
    approach_str, needed_kwargs = entropy_approach
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    entropy = im.entropy(data, approach=approach_str, **needed_kwargs)
    assert isinstance(entropy, (float, tuple))


def test_entropy_class_addressing(entropy_approach):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    approach_str, needed_kwargs = entropy_approach
    est = im.estimator(
        data=data, measure="entropy", approach=approach_str, **needed_kwargs
    )
    assert isinstance(est, EntropyEstimator)
    assert isinstance(est.result(), (float, tuple))
    assert isinstance(est.global_val(), float)
    assert 0 <= est.p_value(10) <= 1
    with pytest.raises(AttributeError):
        est.effective_val()


@pytest.mark.parametrize("offset", [0, 1, 5])
@pytest.mark.parametrize("normalize", [True, False])
def test_mutual_information_functional_addressing(mi_approach, offset, normalize):
    """Test addressing the mutual information estimator classes."""
    approach_str, needed_kwargs = mi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mi = im.mutual_information(
        data_x,
        data_y,
        approach=approach_str,
        offset=offset,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(mi, (float, tuple))
    if isinstance(mi, tuple):
        assert len(mi) == 3
        assert isinstance(mi[0], float)
        assert isinstance(mi[1], np.ndarray)
        assert isinstance(mi[2], float)


@pytest.mark.parametrize("offset", [0, 1, 5])
@pytest.mark.parametrize("normalize", [True, False])
def test_mutual_information_class_addressing(mi_approach, offset, normalize):
    """Test addressing the mutual information estimator classes."""
    approach_str, needed_kwargs = mi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 5, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        data_x=data_x,
        data_y=data_y,
        measure="mutual_information",
        approach=approach_str,
        offset=offset,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(est, MutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    if approach_str in ["discrete", "renyi", "tsallis", "symbolic", "permutation"]:
        with pytest.raises(UnsupportedOperation):
            est.local_val()
    else:
        assert isinstance(est.local_val(), np.ndarray)
    assert 0 <= est.p_value(10) <= 1
    assert -1 <= est.effective_val()


@pytest.mark.parametrize("normalize", [True, False])
def test_cond_mutual_information_functional_addressing(cmi_approach, normalize):
    """Test addressing the conditional mutual information estimator classes."""
    approach_str, needed_kwargs = cmi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mi = im.mutual_information(
        data_x,
        data_y,
        data_z=data_z,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(mi, (float, tuple))
    if isinstance(mi, tuple):
        assert len(mi) == 3
        assert isinstance(mi[0], float)
        assert isinstance(mi[1], np.ndarray)
        assert isinstance(mi[2], float)
    # Query with data_z as positional argument
    im.mutual_information(
        data_x,
        data_y,
        data_z,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    # Use conditional_mutual_information function
    im.conditional_mutual_information(
        data_x,
        data_y,
        data_z,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    im.conditional_mutual_information(
        data_x,
        data_y,
        data_z=data_z,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )


def test_cmi_functional_addressing_faulty():
    """Test wrong usage of the conditional mutual information estimator."""
    with pytest.raises(ValueError):
        im.conditional_mutual_information(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            approach="metric",
        )


@pytest.mark.parametrize("normalize", [True, False])
def test_cond_mutual_information_class_addressing(cmi_approach, normalize):
    """Test addressing the conditional mutual information estimator classes."""
    approach_str, needed_kwargs = cmi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 5, 5, 6, 7, 8, 9, 10])
    data_z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        data_x=data_x,
        data_y=data_y,
        data_z=data_z,
        measure="mutual_information",
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(est, ConditionalMutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), (float, tuple))


@pytest.mark.parametrize("prop_time", [0, 1, 5])
@pytest.mark.parametrize("src_hist_len", [1, 2, 3])
@pytest.mark.parametrize("dest_hist_len", [1, 2, 3])
def test_transfer_entropy_functional_addressing(
    te_approach, prop_time, src_hist_len, dest_hist_len
):
    """Test addressing the transfer entropy estimator classes."""
    approach_str, needed_kwargs = te_approach
    source = np.arange(100)
    dest = np.arange(100)
    te = im.transfer_entropy(
        source,
        dest,
        approach=approach_str,
        prop_time=prop_time,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        **needed_kwargs,
    )
    assert isinstance(te, (float, tuple))
    if isinstance(te, tuple):
        assert len(te) == 3
        assert isinstance(te[0], float)
        assert isinstance(te[1], np.ndarray)
        assert isinstance(te[2], float)


@pytest.mark.parametrize("src_hist_len", [1, 2, 3])
@pytest.mark.parametrize("dest_hist_len", [1, 2, 3])
@pytest.mark.parametrize("cond_hist_len", [1, 2, 3])
def test_cond_transfer_entropy_functional_addressing(
    cte_approach, src_hist_len, dest_hist_len, cond_hist_len
):
    """Test addressing the conditional transfer entropy estimator classes."""
    approach_str, needed_kwargs = cte_approach
    source = np.arange(100)
    dest = np.arange(100)
    cond = np.arange(100)
    te = im.transfer_entropy(
        source,
        dest,
        cond,
        approach=approach_str,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        **needed_kwargs,
    )
    assert isinstance(te, (float, tuple))
    if isinstance(te, tuple):
        assert len(te) == 3
        assert isinstance(te[0], float)
        assert isinstance(te[1], np.ndarray)
        assert isinstance(te[2], float)
    # Query with cond as keyword argument
    im.transfer_entropy(
        source,
        dest,
        cond,
        approach=approach_str,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        **needed_kwargs,
    )
    # Use conditional_transfer_entropy function
    im.conditional_transfer_entropy(
        source,
        dest,
        cond=cond,
        approach=approach_str,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        **needed_kwargs,
    )


def test_cte_functional_addressing_faulty():
    """Test wrong usage of the conditional transfer entropy estimator."""
    with pytest.raises(ValueError):
        im.conditional_transfer_entropy(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            approach="metric",
        )


def test_transfer_entropy_class_addressing(te_approach):
    """Test addressing the transfer entropy estimator classes."""
    approach_str, needed_kwargs = te_approach
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        source=source,
        dest=dest,
        measure="transfer_entropy",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, TransferEntropyEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    if approach_str in ["discrete", "renyi", "tsallis", "symbolic", "permutation"]:
        assert isinstance(est.result(), float)
        with pytest.raises(UnsupportedOperation):
            est.local_val()
    else:
        assert isinstance(est.local_val(), np.ndarray)
    assert 0 <= est.p_value(10) <= 1
    assert isinstance(est.effective_val(), float)


def test_cond_transfer_entropy_class_addressing(cte_approach):
    """Test addressing the conditional transfer entropy estimator classes."""
    approach_str, needed_kwargs = cte_approach
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cond = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        source=source,
        dest=dest,
        cond=cond,
        measure="transfer_entropy",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, ConditionalTransferEntropyEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), (tuple, float))


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

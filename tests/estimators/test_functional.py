"""Tests for the functional interface of the estimators."""

from io import UnsupportedOperation

import numpy as np
import pytest

import infomeasure as im
from tests.conftest import discrete_random_variables
from infomeasure.estimators.base import (
    ConditionalMutualInformationEstimator,
    ConditionalTransferEntropyEstimator,
    EntropyEstimator,
    Estimator,
    MutualInformationEstimator,
    TransferEntropyEstimator,
)
from infomeasure.estimators.entropy import (
    OrdinalEntropyEstimator,
    RenyiEntropyEstimator,
)
from infomeasure.estimators.mutual_information import (
    DiscreteMIEstimator,
    KSGCMIEstimator,
    KSGMIEstimator,
)
from infomeasure.estimators.transfer_entropy import KSGTEEstimator, TsallisCTEEstimator


@pytest.mark.parametrize(
    "measure, approach, expected",
    [
        ("entropy", "permutation", OrdinalEntropyEstimator),
        ("h", "Renyi", RenyiEntropyEstimator),
        ("mutual_information", "discrete", DiscreteMIEstimator),
        ("MI", "ksg", KSGMIEstimator),
        ("conditional_mutual_information", "metric", KSGCMIEstimator),
        ("cmi", "metric", KSGCMIEstimator),
        ("transfer_entropy", "metric", KSGTEEstimator),
        ("cte", "tsallis", TsallisCTEEstimator),
    ],
)
def test_get_estimator_class(measure, approach, expected):
    """Test getting the correct estimator class for a given measure and approach."""
    estimator_cls = im.get_estimator_class(measure, approach)
    assert issubclass(estimator_cls, Estimator)
    assert expected == estimator_cls


def test_get_estimator_class_invalid_measure():
    """Test getting the correct estimator class for an invalid measure."""
    with pytest.raises(
        ValueError, match="Unknown measure: invalid_measure. Available measures:"
    ):
        im.get_estimator_class("invalid_measure", "permutation")


def test_get_estimator_class_no_measure():
    """Test getting the correct estimator class for no measure."""
    with pytest.raises(ValueError, match="The measure must be specified."):
        im.get_estimator_class(approach="permutation")


def test_entropy_functional_addressing(entropy_approach):
    """Test addressing the entropy estimator classes."""
    approach_str, needed_kwargs = entropy_approach
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    entropy = im.entropy(np.array(data), approach=approach_str, **needed_kwargs)
    assert isinstance(entropy, float)
    # test list input
    assert im.entropy(data, approach=approach_str, **needed_kwargs) == pytest.approx(
        entropy
    )


def test_entropy_class_addressing(entropy_approach):
    """Test addressing the entropy estimator classes."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    approach_str, needed_kwargs = entropy_approach
    est = im.estimator(data, measure="entropy", approach=approach_str, **needed_kwargs)
    assert isinstance(est, EntropyEstimator)
    assert isinstance(est.result(), float)
    assert isinstance(est.global_val(), float)
    with pytest.raises(AttributeError):
        est.effective_val()
    if approach_str in ["renyi", "tsallis"]:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    else:
        assert isinstance(est.local_vals(), np.ndarray)


def test_cross_entropy_functional_addressing(entropy_approach, default_rng):
    """Test addressing the cross-entropy estimator classes."""
    approach_str, needed_kwargs = entropy_approach
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 1, 0, -4, 10] * 100)
    entropy = im.entropy(data, data, approach=approach_str, **needed_kwargs)
    assert isinstance(entropy, float)
    # test entropy(data) == cross_entropy(data, data)
    if approach_str not in ["metric", "kl"]:
        assert im.entropy(
            data, approach=approach_str, **needed_kwargs
        ) == pytest.approx(entropy)
        assert im.cross_entropy(
            data, data, approach=approach_str, **needed_kwargs
        ) == pytest.approx(entropy)


def test_cross_entropy_class_addressing(entropy_approach, default_rng):
    """Test addressing the entropy estimator classes."""
    data, _ = discrete_random_variables(0)
    approach_str, needed_kwargs = entropy_approach
    if approach_str in ["metric", "kl"]:
        needed_kwargs["noise_level"] = 0.0
    if approach_str != "discrete":
        data = data + default_rng.normal(0, 0.1, size=len(data))
    est = im.estimator(
        data, data, measure="cross_entropy", approach=approach_str, **needed_kwargs
    )
    assert isinstance(est, EntropyEstimator)
    assert isinstance(est.result(), float)
    assert isinstance(est.global_val(), float)
    assert im.estimator(  # "cross_entropy" not needed, also works with "entropy"
        data, data, measure="entropy", approach=approach_str, **needed_kwargs
    ).result() == pytest.approx(est.result())
    if approach_str not in ["metric", "kl", "renyi", "tsallis"]:
        # cross_entropy(data, data) ~= entropy(data) :
        # as cross_entropy considers on k less, assuming the two RVs are independent
        # for entropy() we have +1, not counting the sample itself for the kth distances
        # Does to this the equality is not satisfied
        assert est.global_val() == pytest.approx(
            im.entropy(data, approach=approach_str, **needed_kwargs)
        )
    with pytest.raises(AttributeError):
        est.effective_val()
    with pytest.raises(UnsupportedOperation):
        est.local_vals()


def test_entropy_class_addressing_no_data():
    """Test addressing the entropy estimator classes without data."""
    with pytest.raises(ValueError, match="``data`` is required for entropy estimation"):
        im.estimator(measure="entropy", approach="renyi")


@pytest.mark.parametrize("n_rv", [3, 4])
def test_entropy_class_addressing_too_many_rv(n_rv):
    """Test addressing the entropy estimator classes with too many random variables."""
    with pytest.raises(
        ValueError,
        match="One or two data parameters are required for entropy estimation. "
        f"Got {n_rv}. To signal that you want",
    ):
        im.estimator(
            *([1, 2, 3] for _ in range(n_rv)), measure="entropy", approach="renyi"
        )


def test_entropy_class_addressing_condition():
    """Test addressing the entropy estimator with an unneeded condition."""
    with pytest.raises(
        ValueError,
        match="Do not pass ``cond`` for entropy estimation. "
        "Conditional entropy is not explicitly supported.",
    ):
        im.estimator([1, 2, 3], cond=[4, 5, 6], measure="entropy", approach="renyi")


@pytest.mark.parametrize("n_rv", [0, 1])
def test_cross_entropy_class_addressing_too_few_vars(n_rv):
    """Test addressing the functional cross-entropy with too few RVs"""
    with pytest.raises(
        ValueError,
        match="Cross-entropy requires at least two random variables "
        "passed as positional parameters:",
    ):
        im.cross_entropy(
            *([1, 2, 3] for _ in range(n_rv)), measure="cross_entropy", approach="renyi"
        )


@pytest.mark.parametrize("n_rv", [0, 1])
def test_cross_entropy_class_addressing_too_few_vars(n_rv):
    """Test addressing the class method for cross-entropy with too few RVs"""
    with pytest.raises(
        ValueError,
        match="Cross-entropy requires at least two random variables "
        "passed as positional parameters:",
    ):
        im.estimator(
            *([1, 2, 3] for _ in range(n_rv)), measure="cross_entropy", approach="renyi"
        )


def test_cross_entropy_functional_random_symmetry(entropy_approach, default_rng):
    """Test cross-entropy is not symmetric. Inputs can be of differing length."""
    approach_str, needed_kwargs = entropy_approach
    p, q = discrete_random_variables(0)
    if approach_str != "discrete":
        p = p + default_rng.normal(0, 0.1, size=len(p))
        q = q + default_rng.normal(0, 0.1, size=len(q))
    assert (
        abs(
            im.cross_entropy(p, q, approach=approach_str, **needed_kwargs)
            - im.cross_entropy(q, p, approach=approach_str, **needed_kwargs)
        )
        > 0.0001
    )


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
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(mi, float)


@pytest.mark.parametrize("n_vars", [0, 1])
def test_mutual_information_functional_too_few_vars(n_vars, default_rng, mi_approach):
    """Test that an error is raised when not enough variables are provided."""
    approach_str, needed_kwargs = mi_approach
    with pytest.raises(
        ValueError,
        match="Mutual Information requires at least two variables as arguments. "
        "If needed",
    ):
        im.mutual_information(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            approach=approach_str,
            **needed_kwargs,
        )


@pytest.mark.parametrize("offset", [0, 1, 5])
@pytest.mark.parametrize("normalize", [True, False])
def test_mutual_information_class_addressing(mi_approach, offset, normalize):
    """Test addressing the mutual information estimator classes."""
    approach_str, needed_kwargs = mi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 5, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        data_x,
        data_y,
        measure="mutual_information",
        approach=approach_str,
        offset=offset,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(est, MutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    if approach_str in ["renyi", "tsallis"]:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    else:
        assert isinstance(est.local_vals(), np.ndarray)
    assert 0 <= est.p_value(10) <= 1


@pytest.mark.parametrize("n_vars", [0, 1])
def test_mutual_information_class_addressing_too_few_vars(
    n_vars, default_rng, mi_approach, caplog
):
    """Test that an error is raised when too few variables are provided."""
    approach_str, needed_kwargs = mi_approach
    if n_vars == 0:
        with pytest.raises(
            ValueError,
            match="No data was provided for mutual information estimation.",
        ):
            im.estimator(
                measure="mutual_information", approach=approach_str, **needed_kwargs
            )
    if n_vars == 1:
        im.estimator(
            default_rng.integers(0, 2, size=10),
            measure="mutual_information",
            approach=approach_str,
            **needed_kwargs,
        )
        assert "WARNING" in caplog.text
        assert (
            "Only one data array provided for mutual information estimation."
            in caplog.text
        )


@pytest.mark.parametrize("n_vars", [2, 3, 4])
def test_mutual_information_class_addressing_n_vars(n_vars, mi_approach, default_rng):
    """Test the mutual information estimator classes with multiple variables."""
    approach_str, needed_kwargs = mi_approach
    data = (default_rng.integers(0, 5, 1000) for _ in range(n_vars))
    est = im.estimator(
        *data,
        measure="mutual_information",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, MutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    # Shannon-like measures have local values
    if approach_str not in ["renyi", "tsallis"]:
        assert isinstance(est.local_vals(), np.ndarray)
    else:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    # p-value is only supported for 2 variables
    if n_vars == 2:
        assert 0 <= est.p_value(10) <= 1
    else:
        with pytest.raises(UnsupportedOperation):
            est.p_value(10)


@pytest.mark.parametrize("normalize", [True, False])
def test_cond_mutual_information_functional_addressing(cmi_approach, normalize):
    """Test addressing the conditional mutual information estimator classes."""
    approach_str, needed_kwargs = cmi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cond = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mi = im.mutual_information(
        data_x,
        data_y,
        cond=cond,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(mi, float)
    # Use conditional_mutual_information function
    im.conditional_mutual_information(
        data_x,
        data_y,
        cond=cond,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    im.conditional_mutual_information(
        data_x,
        data_y,
        cond=cond,
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )


@pytest.mark.parametrize("n_vars", [0, 1])
def test_cmi_functional_too_few_vars(n_vars, default_rng, cmi_approach):
    """Test that an error is raised when not enough variables are provided."""
    approach_str, needed_kwargs = cmi_approach
    with pytest.raises(
        ValueError,
        match="CMI requires at least two variables as arguments",
    ):
        im.conditional_mutual_information(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            cond=default_rng.integers(0, 2, size=10),
            approach=approach_str,
            **needed_kwargs,
        )


def test_cmi_functional_no_condition(cmi_approach, default_rng):
    """Test that an error is raised when no condition variable is provided."""
    approach_str, needed_kwargs = cmi_approach
    with pytest.raises(
        ValueError,
        match="CMI requires a conditional variable. Pass a 'cond' keyword argument.",
    ):
        im.conditional_mutual_information(
            [0] * 10,
            [1] * 10,
            approach=approach_str,
            **needed_kwargs,
        )


@pytest.mark.parametrize("n_vars", [2, 3, 4])
def test_cond_mutual_information_class_addressing_n_vars(
    n_vars, cmi_approach, default_rng
):
    """Test the conditional mutual information estimator classes with multiple variables."""
    approach_str, needed_kwargs = cmi_approach
    data = (default_rng.integers(0, 5, 1000) for _ in range(n_vars))
    cond = default_rng.integers(0, 5, 1000)
    est = im.estimator(
        *data,
        cond=cond,
        measure="conditional_mutual_information",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, ConditionalMutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    # Shannon-like measures have local values
    if approach_str not in ["renyi", "tsallis"]:
        assert isinstance(est.local_vals(), np.ndarray)
    else:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    # p-value is not supported for conditional mutual information
    with pytest.raises(AttributeError):
        est.p_value(10)


@pytest.mark.parametrize("normalize", [True, False])
def test_cond_mutual_information_class_addressing(cmi_approach, normalize):
    """Test addressing the conditional mutual information estimator classes."""
    approach_str, needed_kwargs = cmi_approach
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_y = np.array([1, 2, 3, 5, 5, 6, 7, 8, 9, 10])
    cond = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        data_x,
        data_y,
        cond=cond,
        measure="mutual_information",
        approach=approach_str,
        **(
            {"normalize": normalize}
            if approach_str not in ["discrete", "ordinal", "symbolic", "permutation"]
            else {}
        ),
        **needed_kwargs,
    )
    assert isinstance(est, ConditionalMutualInformationEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    if approach_str in ["renyi", "tsallis"]:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    else:
        assert isinstance(est.local_vals(), np.ndarray)


@pytest.mark.parametrize("n_vars", [0, 1])
def test_cmi_class_addressing_too_few_vars(n_vars, default_rng, cmi_approach, caplog):
    """Test that an error is raised when too many variables are provided."""
    approach_str, needed_kwargs = cmi_approach
    if n_vars == 0:
        with pytest.raises(
            ValueError,
            match="No data was provided for mutual information estimation.",
        ):
            im.estimator(
                cond=default_rng.integers(0, 2, size=10),
                measure="cmi",
                approach=approach_str,
                **needed_kwargs,
            )
    if n_vars == 1:
        im.estimator(
            default_rng.integers(0, 2, size=10),
            cond=default_rng.integers(0, 2, size=10),
            measure="cmi",
            approach=approach_str,
            **needed_kwargs,
        )
        assert "WARNING" in caplog.text
        assert (
            "Only one data array provided for mutual information estimation."
            in caplog.text
        )


def test_cmi_class_addressing_no_condition(cmi_approach, default_rng):
    """Test that an error is raised when no condition variable is provided."""
    approach_str, needed_kwargs = cmi_approach
    with pytest.raises(
        ValueError,
        match="No conditional data was provided",
    ):
        im.estimator(
            [0] * 10,
            [1] * 10,
            measure="cmi",
            approach=approach_str,
            **needed_kwargs,
        )


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
    assert isinstance(te, float)


@pytest.mark.parametrize("n_vars", [3, 4, 5])
def test_transfer_entropy_functional_too_many_vars(n_vars, default_rng, te_approach):
    """Test that an error is raised when too many variables are provided."""
    approach_str, needed_kwargs = te_approach
    with pytest.raises(
        ValueError,
        match="Transfer Entropy requires two variables as arguments and if needed,",
    ):
        im.transfer_entropy(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            approach=approach_str,
            **needed_kwargs,
        )


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
    te = im.conditional_transfer_entropy(
        source,
        dest,
        cond=cond,
        approach=approach_str,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        **needed_kwargs,
    )
    assert isinstance(te, float)
    # Query with cond as keyword argument in the normal im.transfer_entropy() function
    im.transfer_entropy(
        source,
        dest,
        cond=cond,
        approach=approach_str,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        **needed_kwargs,
    )


@pytest.mark.parametrize("n_vars", [3, 4, 5])
def test_cte_functional_too_many_vars(n_vars, default_rng, cte_approach):
    """Test that an error is raised when too many variables are provided."""
    approach_str, needed_kwargs = cte_approach
    with pytest.raises(
        ValueError,
        match="CTE requires two variables as arguments and "
        "the conditional data as keyword argument:",
    ):
        im.conditional_transfer_entropy(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            cond=default_rng.integers(0, 2, size=10),
            approach=approach_str,
            **needed_kwargs,
        )


def test_cte_functional_addressing_faulty(cte_approach):
    """Test wrong usage of the conditional transfer entropy estimator."""
    approach_str, needed_kwargs = cte_approach
    with pytest.raises(ValueError):
        im.conditional_transfer_entropy(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            approach=approach_str,
            **needed_kwargs,
        )


def test_transfer_entropy_class_addressing(te_approach):
    """Test addressing the transfer entropy estimator classes."""
    approach_str, needed_kwargs = te_approach
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        source,
        dest,
        measure="transfer_entropy",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, TransferEntropyEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    if approach_str in ["renyi", "tsallis"]:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    else:
        assert isinstance(est.local_vals(), np.ndarray)
    assert 0 <= est.p_value(10) <= 1
    assert isinstance(est.effective_val(), float)


@pytest.mark.parametrize("n_vars", [3, 4, 5])
def test_transfer_entropy_class_addressing_too_many_vars(
    n_vars, default_rng, te_approach
):
    """Test that an error is raised when too many variables are provided."""
    approach_str, needed_kwargs = te_approach
    with pytest.raises(
        ValueError,
        match="Exactly two data arrays are required for transfer entropy estimation.",
    ):
        im.estimator(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            measure="transfer_entropy",
            approach=approach_str,
            **needed_kwargs,
        )


def test_cond_transfer_entropy_class_addressing(cte_approach):
    """Test addressing the conditional transfer entropy estimator classes."""
    approach_str, needed_kwargs = cte_approach
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cond = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    est = im.estimator(
        source,
        dest,
        cond=cond,
        measure="transfer_entropy",
        approach=approach_str,
        **needed_kwargs,
    )
    assert isinstance(est, ConditionalTransferEntropyEstimator)
    assert isinstance(est.global_val(), float)
    assert est.global_val() == est.res_global
    assert isinstance(est.result(), float)
    if approach_str in ["renyi", "tsallis"]:
        with pytest.raises(UnsupportedOperation):
            est.local_vals()
    else:
        assert isinstance(est.local_vals(), np.ndarray)


@pytest.mark.parametrize("n_vars", [3, 4, 5])
def test_cte_class_addressing_too_many_vars(n_vars, default_rng, cte_approach):
    """Test that an error is raised when too many variables are provided."""
    approach_str, needed_kwargs = cte_approach
    with pytest.raises(
        ValueError,
        match="Exactly two data arrays are required for transfer entropy estimation.",
    ):
        im.estimator(
            *(default_rng.integers(0, 2, size=10) for _ in range(n_vars)),
            cond=default_rng.integers(0, 2, size=10),
            measure="cte",
            approach=approach_str,
            **needed_kwargs,
        )


def test_cte_class_addressing_no_condition(cte_approach, default_rng):
    """Test that an error is raised when no condition variable is provided."""
    approach_str, needed_kwargs = cte_approach
    with pytest.raises(
        ValueError,
        match="No conditional data was provided",
    ):
        im.estimator(
            [0] * 10,
            [1] * 10,
            measure="cte",
            approach=approach_str,
            **needed_kwargs,
        )


@pytest.mark.parametrize("prop_time", [0, 1, 5])
def test_te_offset_prop_time(te_approach, caplog, prop_time):
    """Test offset parameter for the transfer entropy estimator.

    The prop time can also be passed as `offset` parameter, for user-friendliness.
    Test that results are the same for both parameters.
    """
    approach_str, needed_kwargs = te_approach
    source = np.random.rand(100)
    dest = np.random.rand(100)
    if approach_str in ["renyi", "tsallis", "ksg", "metric"]:
        needed_kwargs["noise_level"] = 0
    res_pt = im.te(
        source,
        dest,
        approach=approach_str,
        prop_time=prop_time,
        **needed_kwargs,
    )
    assert (
        "Using the `offset` parameter as `prop_time`. "
        "Please use `prop_time` for the propagation time."
    ) not in caplog.text
    res_offset = im.te(
        source,
        dest,
        approach=approach_str,
        offset=prop_time,
        **needed_kwargs,
    )
    assert res_pt == res_offset
    # check that warning was printed to the log
    if prop_time != 0:
        assert (
            "Using the `offset` parameter as `prop_time`. "
            "Please use `prop_time` for the propagation time."
        ) in caplog.text


def test_use_both_offset_prop_time(te_approach):
    """Test error when using both offset and prop_time parameters."""
    approach_str, needed_kwargs = te_approach
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="Both `offset` and `prop_time` are set."):
        im.te(
            source,
            dest,
            approach=approach_str,
            offset=1,
            prop_time=1,
            **needed_kwargs,
        )


@pytest.mark.parametrize(
    "func", [im.entropy, im.mutual_information, im.transfer_entropy]
)
def test_functional_addressing_unknown_approach(func):
    """Test addressing the functional wrappers with unknown approaches."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(
        ValueError, match="Unknown estimator: unknown. Available estimators: "
    ):
        func(data, approach="unknown")


@pytest.mark.parametrize(
    "func", [im.entropy, im.mutual_information, im.transfer_entropy]
)
def test_functional_addressing_no_approach(func):
    """Test addressing the functional wrappers without an approach."""
    with pytest.raises(ValueError, match="``approach`` must be provided"):
        func([1, 2, 3, 4, 5], approch="test")


def test_class_addressing_unknown_measure():
    """Test addressing the estimator wrapper with an unknown measure."""
    with pytest.raises(ValueError, match="Unknown measure: unknown"):
        im.estimator(measure="unknown", approach="")


def test_class_addressing_no_measure():
    """Test addressing the estimator wrapper without a measure."""
    with pytest.raises(ValueError, match="``measure`` is required."):
        im.estimator(approach="test")

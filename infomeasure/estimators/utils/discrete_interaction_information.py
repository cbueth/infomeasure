"""Functions for interaction information, a multivariate generalization of
mutual information."""

from collections import Counter

from numpy import (
    clip,
    uint64,
    log,
    ndarray,
    ones,
    prod,
    ravel,
    unique,
    zeros,
    count_nonzero,
)
from numpy import (
    sum as np_sum,
)
from scipy.sparse import find as sp_find
from scipy.stats.contingency import crosstab
from sparse import COO, asnumpy


def mutual_information_global(
    *data: tuple,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Estimate the global mutual information between multiple random variables.

    Parameters
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the global mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    log_func : callable, optional
        The logarithm function to use. Default is the natural logarithm.
    miller_madow_correction : str | float | int, optional
        If not None, apply the Miller-Madow correction to the global mutual
        information in the information unit of the passed value.
        ``log_func`` and ``miller_madow_correction`` should be the same base.

    Returns
    -------
    float
        The global mutual information between the random variables.
    """
    # check that miller_madow_correction is None, float, int or "e"
    if miller_madow_correction is not None and (
        not isinstance(miller_madow_correction, (str, float, int))
        or (isinstance(miller_madow_correction, str) and miller_madow_correction != "e")
    ):
        raise ValueError(
            f"miller_madow_correction must be None, float or 'e', "
            f"got {miller_madow_correction}."
        )

    if all(d.ndim == 1 for d in data):
        if len(data) == 2:
            return _mutual_information_global_2d_int(
                *data,
                log_func=log_func,
                miller_madow_correction=miller_madow_correction,
            )
        else:
            return _mutual_information_global_nd_int(
                *data,
                log_func=log_func,
                miller_madow_correction=miller_madow_correction,
            )
    else:
        return _mutual_information_global_nd_other(
            *data,
            log_func=log_func,
            miller_madow_correction=miller_madow_correction,
        )


def _mutual_information_global_nd_int(
    *data: tuple,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Estimate the global mutual information between an arbitrary number of
    random variables."""
    uniques, indices = zip(*[unique(var, return_inverse=True, axis=0) for var in data])
    contingency_coo = COO(
        coords=indices,
        data=ones(len(indices[0]), dtype=uint64),
        shape=tuple(len(uniq) for uniq in uniques),
        fill_value=0,
    )

    # Non-zero indices and values
    idxs = contingency_coo.nonzero()
    vals = contingency_coo.data

    # Marginal probabilities
    count_marginals = [
        asnumpy(
            contingency_coo.sum(  # all axes, except i
                axis=tuple(range(contingency_coo.ndim))[:i]
                + tuple(range(contingency_coo.ndim))[i + 1 :]
            )
        )
        for i in range(contingency_coo.ndim)
    ]

    # Early return if any of the marginal entropies is zero
    if any(count_m.size == 1 for count_m in count_marginals):
        return 0.0

    # Calculate the expected logarithm values for the outer product of marginal
    # probabilities, only for non-zero entries.
    outer = prod(
        [
            count.take(idx).astype(uint64, copy=False)
            for count, idx in zip(count_marginals, idxs)
        ],
        axis=0,
    )

    # Normalized contingency table (joint probability)
    contingency_sum = contingency_coo.sum()
    p_joint = vals / contingency_sum

    # Logarithm of the non-zero elements
    log_p_joint = log_func(vals)

    log_outer = -log_func(outer) + len(count_marginals) * log_func(contingency_sum)
    # Combine the terms to calculate the mutual information
    mi = p_joint * (log_p_joint - log_func(contingency_sum)) + p_joint * log_outer

    misum = mi.sum()  # interaction information can be negative, do not clip
    if miller_madow_correction is None:
        return misum
    else:
        corr = millermadow_mi_corr(
            contingency_coo.shape, len(vals), len(data[0]), miller_madow_correction
        )
        return misum + corr


def _mutual_information_global_2d_int(
    *data: tuple[ndarray[int]],
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Estimate the global mutual information between two random variables.

    The approach relies on the contingency table of the two variables.
    Instead of calculating the full outer product, only the non-zero elements are
    considered.
    Code adapted from
    the :func:`mutual_info_score() <sklearn.metrics.mutual_info_score>` function in
    scikit-learn.
    """
    # Contingency table - COOrdinate sparse matrix
    contingency_coo = crosstab(*data, sparse=True).count
    # Non-zero indices and values
    nzx, nzy, nzv = sp_find(contingency_coo)

    # Normalized contingency table (joint probability)
    contingency_sum = contingency_coo.sum()
    p_joint = nzv / contingency_sum
    # Marginal probabilities
    pi = ravel(contingency_coo.sum(axis=1))
    pj = ravel(contingency_coo.sum(axis=0))

    # Early return if any of the marginal entropies is zero
    if pi.size == 1 or pj.size == 1:
        return 0.0

    # Logarithm of the non-zero elements
    log_p_joint = log_func(nzv)

    # Calculate the expected logarithm values for the outer product of marginal
    # probabilities, only for non-zero entries.
    outer = pi.take(nzx).astype(uint64, copy=False) * pj.take(nzy).astype(
        uint64, copy=False
    )
    log_outer = -log_func(outer) + log_func(pi.sum()) + log_func(pj.sum())
    # Combine the terms to calculate the mutual information
    mi = p_joint * (log_p_joint - log_func(contingency_sum)) + p_joint * log_outer
    misum = clip(mi.sum(), 0.0, None)
    if miller_madow_correction is None:
        return misum
    else:
        corr = millermadow_mi_corr(
            contingency_coo.shape, len(nzv), len(data[0]), miller_madow_correction
        )
        return misum + corr


def _mutual_information_global_nd_other(
    *data: tuple[ndarray],
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Alternative method to estimate the global mutual information between an
    arbitrary number of random variables.

    Same as :func:`_mutual_information_global_nd_int`, but for non-integer data.
    """
    # data is a tuple of ndarrays, for joint data, concatenate these rows
    # joint_data = [tuple(row) for row in column_stack(data)]
    joint_data = [tuple(tuple(val) for val in row) for row in zip(*data)]

    # Count joint and marginal occurrences
    joint_counts = Counter(joint_data)
    joint_total = sum(joint_counts.values())
    marginal_counts = [Counter([tuple(val) for val in var]) for var in data]
    marginal_totals = [sum(counts.values()) for counts in marginal_counts]

    # Estimate probabilities
    joint_prob = {key: val / joint_total for key, val in joint_counts.items()}
    marginal_prob = [
        {key: val / total for key, val in counts.items()}
        for counts, total in zip(marginal_counts, marginal_totals)
    ]

    # Calculate the mutual information
    mi_sum = [
        joint_prob[key]
        * log_func(
            joint_prob[key]
            / prod([marginal_prob[i][key[i]] for i in range(len(data))], axis=0)
        )
        for key in joint_prob
    ]
    if len(mi_sum) == 0:
        return 0.0
    misum = np_sum(mi_sum)

    if miller_madow_correction is None:
        return misum
    else:
        corr = millermadow_mi_corr(
            list(len(counter.keys()) for counter in marginal_counts),
            len(joint_counts),
            len(data[0]),
            miller_madow_correction,
        )
        return misum + corr


def mutual_information_local(
    *data: tuple,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> ndarray:
    """Estimate the local mutual information between multiple random variables.

    The mean of the local mutual information is the global mutual information.
    Only calculating the global value is more efficient,
    so evaluating the local mutual information should only be done
    when explicitly needed.

    Parameters
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the local mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    log_func : callable, optional
        The logarithm function to use. Default is the natural logarithm.
    miller_madow_correction : str | float | int, optional
        If not None, apply the Miller-Madow correction to the global mutual
        information in the information unit of the passed value.
        ``log_func`` and ``miller_madow_correction`` should be the same base.

    Returns
    -------
    ndarray
        The local mutual information between the random variables.
    """
    # check that miller_madow_correction is None, float, int or "e"
    if miller_madow_correction is not None and (
        not isinstance(miller_madow_correction, (str, float, int))
        or (isinstance(miller_madow_correction, str) and miller_madow_correction != "e")
    ):
        raise ValueError(
            f"miller_madow_correction must be None, float or 'e', "
            f"got {miller_madow_correction}."
        )
    # Contingency table - COOrdinate sparse matrix
    uniques, indices = zip(*[unique(var, return_inverse=True, axis=0) for var in data])
    contingency_coo = COO(
        coords=indices,
        data=ones(len(indices[0]), dtype=uint64),
        shape=tuple(len(uniq) for uniq in uniques),
        fill_value=0,
    )

    # Normalized contingency table (joint probability)
    contingency_sum = contingency_coo.sum()
    # Marginal probabilities
    count_marginals = [
        asnumpy(
            contingency_coo.sum(  # all axes, except i
                axis=tuple(range(contingency_coo.ndim))[:i]
                + tuple(range(contingency_coo.ndim))[i + 1 :]
            )
        )
        for i in range(contingency_coo.ndim)
    ]

    # Early return if any of the marginal entropies is zero
    if any(count_m.size == 1 for count_m in count_marginals):
        return zeros(len(data[0]))

    # To get local values we iterate over *data
    # for each row in the input data: log( p(data) / p(data1) * p(data2) )
    p_joint = contingency_coo[indices].data / contingency_sum
    outer = prod(
        [count[indices[i]] for i, count in enumerate(count_marginals)]
        / contingency_sum,
        axis=0,
    )
    mi_local = -log_func(outer) + log_func(p_joint)
    if miller_madow_correction is None:
        return mi_local
    else:
        corr = millermadow_mi_corr(
            contingency_coo.shape,
            len(contingency_coo.data),
            len(data[0]),
            miller_madow_correction,
        )
        return mi_local + corr


def conditional_mutual_information_global(
    *data: tuple,
    cond: ndarray,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Estimate the global conditional mutual information
    between multiple random variables and a conditioning variable.

    Parameters
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the global mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like, shape (n_samples,)
        The conditioning variable.
    log_func : callable, optional
        The logarithm function to use. Default is the natural logarithm.
    miller_madow_correction : str | float | int, optional
        If not None, apply the Miller-Madow correction to the global mutual
        information in the information unit of the passed value.
        ``log_func`` and ``miller_madow_correction`` should be the same base.

    Returns
    -------
    float
        The global conditional mutual information between the random variables.

    Notes
    -----
    If wanting a condition of joint random variables, one must join them beforehand
    into one dimension. This is due to the complexity of the calculation, keeping it
    arbitrary enough.
    One can join discrete random variables through
    :py:func:`~infomeasure.estimators.utils.ordinal.reduce_joint_space`, but when using
    CMI and CTE, this will happen automatically when passing a tuple of RVs as the
    ``cond``.

    Raises
    ------
    ValueError
        If the conditioning variable is not one-dimensional.
    """
    # check that miller_madow_correction is None, float, int or "e"
    if miller_madow_correction is not None and (
        not isinstance(miller_madow_correction, (str, float, int))
        or (isinstance(miller_madow_correction, str) and miller_madow_correction != "e")
    ):
        raise ValueError(
            f"miller_madow_correction must be None, float or 'e', "
            f"got {miller_madow_correction}."
        )
    if cond.ndim != 1:
        raise ValueError("The conditioning variable must be one-dimensional.")
    return _conditional_mutual_information_global_nd_int(
        *data,
        cond=cond,
        log_func=log_func,
        miller_madow_correction=miller_madow_correction,
    )


def _conditional_mutual_information_global_nd_int(
    *data: tuple,
    cond: ndarray,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> float:
    """Estimate the global conditional mutual information between an arbitrary number of
    random variables and a conditioning variable."""
    uniques, indices = zip(
        *[unique(var, return_inverse=True, axis=0) for var in (data + (cond,))]
    )
    contingency_coo = COO(
        coords=indices,
        data=ones(len(indices[0]), dtype=uint64),
        shape=tuple(len(uniq) for uniq in uniques),
        fill_value=0,
    )

    # Non-zero indices and values
    idxs = contingency_coo.nonzero()
    vals = contingency_coo.data

    # Marginal-conditioned probabilities
    count_marginals_cond = [
        asnumpy(
            contingency_coo.sum(  # all axes, except i and cond
                axis=tuple(range(contingency_coo.ndim - 1))[:i]
                + tuple(range(contingency_coo.ndim - 1))[i + 1 :]
            )
        )
        for i in range(contingency_coo.ndim - 1)
    ]
    count_cond = asnumpy(
        contingency_coo.sum(axis=tuple(range(contingency_coo.ndim - 1)))
    )
    # all axes, except cond

    # Early return if any of the marginal entropies is zero
    if any(count_m.size == 1 for count_m in count_marginals_cond):
        return 0.0

    # Calculate the expected logarithm values for the outer product of marginal
    # probabilities, only for non-zero entries.
    outer = prod(
        [
            count[idx, idxs[-1]].astype(uint64, copy=False)
            for count, idx in zip(count_marginals_cond, idxs[:-1])
        ],
        axis=0,
    )

    # Normalized contingency table (joint probability)
    contingency_sum = contingency_coo.sum()
    p_joint = vals / contingency_sum

    # Logarithm of the non-zero elements
    p_cond = count_cond.take(idxs[-1]).astype(uint64, copy=False)
    log_p_joint = log_func(vals * p_cond) - log_func(contingency_sum)

    log_outer = -log_func(outer) + sum(
        log_func(count_m.sum()) for count_m in count_marginals_cond[:-1]
    )
    # Combine the terms to calculate the mutual information
    mi = p_joint * log_p_joint + p_joint * log_outer
    misum = mi.sum()  # interaction information can be negative, do not clip
    if miller_madow_correction is None:
        return misum
    else:
        corr = millermadow_mi_corr(
            list(count_nonzero(marg) for marg in count_marginals_cond),
            len(vals),
            len(data[0]),
            miller_madow_correction,
            k_cond=len(count_cond),
        )
        return misum + corr


def conditional_mutual_information_local(
    *data: tuple,
    cond: ndarray,
    log_func: callable = log,
    miller_madow_correction: str | float | int = None,
) -> ndarray:
    """Estimate the local conditional mutual information between multiple
    random variables and a conditioning variable.

    The mean of the local conditional mutual information is the
    global conditional mutual information.
    Only calculating the global value is more efficient,
    so evaluating the local conditional mutual information should only be done
    when explicitly needed.

    Parameters
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the local mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like, shape (n_samples,)
        The conditioning variable.
    log_func : callable, optional
        The logarithm function to use. Default is the natural logarithm.
    miller_madow_correction : str | float | int, optional
        If not None, apply the Miller-Madow correction to the global mutual
        information in the information unit of the passed value.
        ``log_func`` and ``miller_madow_correction`` should be the same base.

    Returns
    -------
    ndarray
        The local conditional mutual information between the random variables.
    """
    # check that miller_madow_correction is None, float, int or "e"
    if miller_madow_correction is not None and (
        not isinstance(miller_madow_correction, (str, float, int))
        or (isinstance(miller_madow_correction, str) and miller_madow_correction != "e")
    ):
        raise ValueError(
            f"miller_madow_correction must be None, float or 'e', "
            f"got {miller_madow_correction}."
        )
    # Contingency table - COOrdinate sparse matrix
    uniques, indices = zip(
        *[unique(var, return_inverse=True, axis=0) for var in (data + (cond,))]
    )
    contingency_coo = COO(
        coords=indices,
        data=ones(len(indices[0]), dtype=uint64),
        shape=tuple(len(uniq) for uniq in uniques),
        fill_value=0,
    )

    # Normalized contingency table (joint probability)
    contingency_sum = contingency_coo.sum()
    # Marginal-conditioned probabilities
    count_marginals_cond = [
        asnumpy(
            contingency_coo.sum(  # all axes, except i and cond
                axis=tuple(range(contingency_coo.ndim - 1))[:i]
                + tuple(range(contingency_coo.ndim - 1))[i + 1 :]
            )
        )
        for i in range(contingency_coo.ndim - 1)
    ]
    count_cond = asnumpy(
        contingency_coo.sum(axis=tuple(range(contingency_coo.ndim - 1)))
    )

    # Early return if any of the marginal entropies is zero
    if any(count_m.size == 1 for count_m in count_marginals_cond):
        return zeros(len(data[0])).astype(float)

    # To get local values we iterate over *data
    # for each row in the input data: log( p(data) / p(data1) * p(data2) )
    p_joint = contingency_coo[indices].data / contingency_sum
    p_cond = count_cond[indices[-1]] / contingency_sum
    outer = prod(
        [count[indices[i], indices[-1]] for i, count in enumerate(count_marginals_cond)]
        / contingency_sum,
        axis=0,
    )
    misum = log_func(p_joint * p_cond) - log_func(outer)
    if miller_madow_correction is None:
        return misum
    else:
        corr = millermadow_mi_corr(
            list(count_nonzero(marg) for marg in count_marginals_cond),
            len(contingency_coo.data),
            len(data[0]),
            miller_madow_correction,
            k_cond=len(count_cond),
        )
        return misum + corr


def millermadow_mi_corr(k_i, k_joint, n, base, k_cond=None):
    """
    Computes the Miller-Madow mutual information correction term.

    This function calculates a correction term used to adjust the bias in mutual
    information estimates, which arise due to finite sample size issues.
    The correction is based on the marginal counts and joint count in the observed
    distributions.

    Parameters
    ----------
    k_i : list[int]
        A list containing the marginal cardinalities of individual variables in the
        dataset.
        Each element represents the number of unique values for the respective
        variable.
    k_joint : int
        The cardinality of the joint distribution.
        Represents the number of unique
        observations across all combined dimensions.
    n : int
        The sample size, representing the total number of observations in the data.
    base : str | float | int
        The logarithmic base used for the mutual information computation.
        If set to
        "e", natural logarithm is used.
        Otherwise, log of the specified base is used.
    k_cond : int, optional
        The cardinality of the conditional variable.
        When this is used, k_i can be used as k_iZ, to calculate the correction for
        conditional MI.

    Returns
    -------
    float
        The calculated Miller-Madow correction term to adjust the mutual information
        value.
    """
    corr = (
        sum(k_i)  # sum K_i
        - len(k_i)  # sum -1
        - k_joint
        + 1  # -(K_{1,...,i} +1)
        - ((k_cond - 1) if k_cond is not None else 0)
    ) / (2 * n)  # / 2N
    if base != "e":
        corr /= log(base)
    return corr

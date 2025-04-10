"""Jensen-Shannon Divergence (JSD)."""

from numpy import sum as np_sum, concatenate, ndarray

from ..estimators.base import DistributionMixin
from ..estimators.entropy import (
    RenyiEntropyEstimator,
    TsallisEntropyEstimator,
    KozachenkoLeonenkoEntropyEstimator,
    KernelEntropyEstimator,
)
from ..estimators.functional import get_estimator_class


def jensen_shannon_divergence(*data, approach: str | None = None, **kwargs):
    r"""Calculate the Jensen-Shannon Divergence between two or more distributions.

    The Jensen-Shannon Divergence is a symmetrized and smoothed version of the
    Kullback-Leibler Divergence. It is calculated as the average of the
    Kullback-Leibler Divergence between each distribution and the average
    distribution.

    .. math::

        JSD(P \| Q) = \frac{1}{2} KL(P \| M) + \frac{1}{2} KL(Q \| M)

    where :math:`M = \frac{1}{2} (P + Q)`.

    Parameters
    ----------
    p : array-like
        The first data.
    q : array-like
        The second data.
    ... : array-like
        Further data to compare.
    approach : str
        The name of the entropy estimator to use.
    **kwargs : dict
        Additional keyword arguments to pass to the entropy estimator.

    Returns
    -------
    float
        The Jensen-Shannon Divergence.

    Raises
    ------
    ValueError
        If the approach is not supported or the entropy estimator is not
        compatible with the Jensen-Shannon Divergence.
    ValueError
        If any of the given data is not an array-like object.
    """
    if approach is None:
        raise ValueError("The approach must be specified.")
    if not all(isinstance(var, (list, ndarray)) for var in data):
        raise ValueError("All data must be array-like objects.")
    estimator_class = get_estimator_class(measure="entropy", approach=approach)
    if issubclass(
        estimator_class,
        (
            RenyiEntropyEstimator,
            TsallisEntropyEstimator,
            KozachenkoLeonenkoEntropyEstimator,
        ),
    ):
        raise ValueError(
            "The Jensen-Shannon Divergence is not supported for the "
            f"{estimator_class.__name__} estimator."
        )
    # if estimator_class has mixin DistributionMixin
    # then we can use the distribution method
    if issubclass(estimator_class, DistributionMixin):
        estimators = tuple(estimator_class(var, **kwargs) for var in data)
        marginal = sum(estimator.global_val() for estimator in estimators) / len(data)
        # the distributions have some matching and some unique keys, create a new dict
        # with the sum of the values of union of keys
        dists = [estimator.distribution() for estimator in estimators]
        # dict(
        #   m_i: (p(x_i) + q(x_i) + ... + r(x_i)) / n
        # )
        dists = {
            key: sum(dist.get(key, 0) for dist in dists) / len(dists)
            for key in set().union(*dists)
        }
        mixture = list(dists.values())
        mixture = -np_sum(mixture * estimators[0]._log_base(mixture))
        return mixture - marginal
    if issubclass(estimator_class, KernelEntropyEstimator):
        # The mixture distribution is the union of the data, as the kernel density
        # estimation is applied afterward.
        mix_est = estimator_class(concatenate(data, axis=0), **kwargs)
        return mix_est.global_val() - sum(
            estimator_class(var, **kwargs).global_val() for var in data
        ) / len(data)
    else:
        raise ValueError(  # pragma: no cover
            f"The approach {approach} is not supported."
        )

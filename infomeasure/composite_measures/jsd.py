"""Jensen Shannon Divergence (JSD) between two probability distributions."""

from numpy import sum as np_sum, concatenate

from ..estimators.base import DistributionMixin
from ..estimators.functional import entropy_estimators, _get_estimator


def jensen_shannon_divergence(*data, approach="discrete", **kwargs):
    r"""Calculate the Jensen-Shannon Divergence between two distributions.

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
    """
    if approach in ["renyi", "tsallis", "kl"]:
        raise ValueError(
            "The Jensen-Shannon Divergence is not supported for the "
            f"{approach.capitalize()} entropy."
        )
    EstimatorClass = _get_estimator(entropy_estimators, approach)
    # if EstimatorClass has mixin DistributionMixin
    # then we can use the distribution method
    if issubclass(EstimatorClass, DistributionMixin):
        estimators = tuple(EstimatorClass(var, **kwargs) for var in data)
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
    if approach in ["kernel"]:
        # The mixture distribution is the union of the data, as the kernel density
        # estimation is applied afterwards.
        mix_est = EstimatorClass(concatenate(data, axis=0), **kwargs)
        return mix_est.global_val() - sum(
            EstimatorClass(var, **kwargs).global_val() for var in data
        ) / len(data)
    else:
        raise ValueError(f"The approach {approach} is not supported.")

    # # distinguish between
    # # distribution: dict (discrete, symbolic,
    # # distribution: using KDTree (kernel density estimation)
    #
    # ## DICT (discrete, permutation)
    # # Ensure the distributions are numpy arrays
    # estimators = (
    #     estimator.distribution(
    #         var,
    #     )
    #     for var in data
    # )
    #
    # # dict(
    # #   m_i: (p(x_i) + q(x_i) + ... + r(x_i)) / n
    # # )
    #
    # # Mixture distribution
    # mixture = sum(distributions) / len(distributions)
    #
    # # Calculate the Jensen-Shannon Divergence
    # return shannon(mixture) + sum(
    #     estimator.global_val(dist) for dist in distributions
    # ) / len(distributions)
    # ##  kernel
    # return h(stack(data), **kwargs) - sum(entropy(d, **kwargs) for d in data) / len(
    #     data
    # )
    # ## unsupported KL, Renyi, Tsallis (could be same as above)

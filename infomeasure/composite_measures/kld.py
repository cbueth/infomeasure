"""Kullback-Leibler divergence."""

from ..estimators.functional import get_estimator_class
from ..utils.config import logger


def kullback_leiber_divergence(data_p, data_q, approach: str = "", **kwargs):
    r"""Calculate the Kullback-Leibler Divergence between two distributions.

    The Kullback-Leibler Divergence is a measure of the difference between two
    probability distributions. It is calculated as the expectation of the
    logarithm of the ratio of the probability of two events.
    To calculate, we use the identity of combining the joint and marginal
    entropies:

    .. math::

        KL(P \| Q) = \sum_{x \in X} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
                = H_Q(P) - H(P)

    Parameters
    ----------
    data_p : array-like
        The first data.
    data_q : array-like
        The second data.
    approach : str
        The name of the entropy estimator to use.
    **kwargs : dict
        Additional keyword arguments to pass to the entropy estimator.

    Returns
    -------
    float
        The Kullback-Leibler Divergence.

    Raises
    ------
    ValueError
        If the approach is not supported or the entropy estimator is not
        compatible with the Kullback-Leibler Divergence.
    """
    if approach is None:
        raise ValueError("The approach must be specified.")
    estimator_class = get_estimator_class(measure="entropy", approach=approach)
    h_qp = estimator_class(data_p, data_q, **kwargs).global_val()
    h_p = estimator_class(data_p, **kwargs).global_val()
    logger.debug(f"KLD: H_Q(P)= {h_qp}, H(P) = {h_p}")
    return h_qp - h_p

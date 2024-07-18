"""Module for the Renyi mutual information estimator."""

from ..base import LogBaseMixin, PValueMixin, MutualInformationEstimator


class RenyiMIEstimator(LogBaseMixin, PValueMixin, MutualInformationEstimator):
    r"""Estimator for the Renyi mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    alpha : float
        The Renyi parameter.
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass

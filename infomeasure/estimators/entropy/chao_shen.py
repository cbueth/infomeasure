"""Module for the Chao-Shen entropy estimator."""

from numpy import sum as np_sum

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.exceptions import TheoreticalInconsistencyError


class ChaoShenEntropyEstimator(DiscreteHEstimator):
    r"""Chao-Shen entropy estimator.

    .. math::

        \hat{H}_{CS} = - \sum_{i=1}^{K} \frac{\hat{p}_i^{CS} \log \hat{p}_i^{CS}}{1 - (1 - \hat{p}_i^{ML} C)^N}

    where

    .. math::

        \hat{p}_i^{CS} = C \cdot \hat{p}_i^{ML}

    and :math:`C = 1 - \frac{f_1}{N}` is the estimated coverage,
    :math:`f_1` is the number of singletons (species observed exactly once),
    :math:`\hat{p}_i^{ML}` is the maximum likelihood probability estimate,
    :math:`N` is the sample size, and :math:`K` is the number of observed species.

    The Chao-Shen estimator provides a bias-corrected estimate of Shannon entropy
    that accounts for unobserved species through coverage estimation.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def _simple_entropy(self):
        """Calculate the Chao-Shen entropy of the data.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        N = self.data[0].N
        # Number of singletons
        f1 = np_sum(self.data[0].counts == 1)
        if f1 == N:
            f1 -= 1  # Avoid C=0

        # Estimated coverage
        C = 1 - f1 / N
        pa = (  # Coverage adjusted empirical frequencies
            C * self.data[0].probabilities
        )
        la = 1 - (1 - pa) ** N  # Probability to see a bin (species) in the sample

        # Chao-Shen (2003) entropy estimator
        return -np_sum(pa * self._log_base(pa) / la)

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for Chao-Shen estimator
            due to fundamental issues with mixing bias corrections from different distributions.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Chao-Shen estimator. "
            "The Chao-Shen correction creates theoretical inconsistencies when applied to cross-entropy: "
            "(1) Asymmetric nature problem - unclear which distribution should use Chao-Shen correction; "
            "(2) Coverage estimation issues - different coverage estimates for each distribution lack "
            "theoretical foundation when mixed; "
            "(3) Denominator complexity - the correction involves sample-specific denominators "
            "that are tied to individual distributions, making cross-distribution application arbitrary."
        )

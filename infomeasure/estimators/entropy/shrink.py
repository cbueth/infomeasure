"""Module for the shrink (James-Stein) entropy estimator."""

from numpy import asarray
from numpy import sum as np_sum

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.exceptions import TheoreticalInconsistencyError


class ShrinkEntropyEstimator(DiscreteHEstimator):
    r"""Shrinkage (James-Stein) entropy estimator.

    This estimator applies James-Stein shrinkage to the probability estimates
    before computing entropy, which can reduce bias in small sample scenarios.
    The shrinkage probabilities are calculated as:

    .. math::

        \hat{p}_x^{\text{SHR}} = \lambda t_x + (1 - \lambda) \hat{p}_x^{\text{ML}}

    where :math:`\hat{p}_x^{\text{ML}}` are the maximum likelihood probability estimates,
    :math:`t_x = 1/K` is the uniform target distribution, and the shrinkage parameter
    :math:`\lambda` is given by:

    .. math::

        \lambda = \frac{ 1 - \sum_{x=1}^{K} (\hat{p}_x^{\text{SHR}})^2}{(n-1) \sum_{x=1}^K (t_x - \hat{p}_x^{\text{ML}})^2}

    The entropy is then computed using these shrinkage-corrected probabilities.

    Based on the implementation in the R package entropy :cite:p:`hausserEntropyInferenceJamesStein2009`.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def _simple_entropy(self):
        """Calculate the shrinkage entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        p_shrink = self._shrink_probs()

        # Calculate entropy
        entropy = -np_sum(p_shrink * self._log_base(p_shrink))

        return entropy

    def _calculate_lambda_shrink(self, N, u, t):
        """Calculate the shrinkage parameter lambda.

        Parameters
        ----------
        N : int
            Total number of observations
        u : array-like
            Maximum likelihood probabilities
        t : float
            Target probability (1/K)

        Returns
        -------
        float
            The shrinkage parameter lambda
        """
        # Variance of u
        varu = u * (1.0 - u) / (N - 1)

        # Mean squared difference
        msp = np_sum((u - t) ** 2)

        if msp == 0:
            return 1.0
        else:
            lambda_val = np_sum(varu) / msp
            # Clamp lambda to [0, 1]
            if lambda_val > 1:
                return 1.0
            elif lambda_val < 0:
                return 0.0
            else:
                return lambda_val

    def _extract_local_values(self):
        """Separately, calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        shrink_dict = self.dist_dict

        # Calculate local values for each data point
        local_values = asarray(
            [-self._log_base(shrink_dict[val]) for val in self.data[0].data]
        )

        return local_values

    @property
    def dist_dict(self):
        """Dictionary of shrinkage probabilities for each unique value. Used by JSD."""
        p_shrink = self._shrink_probs()
        # Create a mapping from unique values to their shrinkage probabilities
        return dict(zip(self.data[0].uniq, p_shrink))

    def _shrink_probs(self):
        N = self.data[0].N  # total number of observations
        K = self.data[0].K
        # Maximum likelihood probabilities
        # p_ml = counts / N
        p_ml = self.data[0].probabilities
        # Target probabilities (uniform distribution)
        t = 1.0 / K
        # Calculate lambda (shrinkage parameter)
        if N == 0 or N == 1:
            lambda_shrink = 1.0
        else:
            lambda_shrink = self._calculate_lambda_shrink(N, p_ml, t)
        # Calculate shrinkage probabilities
        p_shrink = lambda_shrink * t + (1 - lambda_shrink) * p_ml
        return p_shrink

    def _cross_entropy(self) -> float:
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for shrinkage estimator
            due to a conceptual mismatch between shrinkage correction and cross-entropy.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for shrinkage estimator. "
            "The shrinkage correction is designed for bias correction in entropy "
            "estimation using a specific shrinkage target, but cross-entropy mixes "
            "probabilities from one distribution with corrections from another, "
            "creating a theoretical inconsistency."
        )

"""Module for the Asymptotic NSB entropy estimator."""

from numpy import euler_gamma, log
from scipy.special import digamma

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.config import logger
from ...utils.exceptions import TheoreticalInconsistencyError
from ... import Config
from ...utils.types import LogBaseType


class AnsbEntropyEstimator(DiscreteHEstimator):
    r"""Asymptotic NSB entropy estimator.

    The Asymptotic NSB (ANSB) estimator provides entropy estimation for extremely
    undersampled discrete data where the number of unique values K is comparable
    to the sample size N.

    .. math::

        \hat{H}_{\text{ANSB}} = (C_\gamma - \log(2)) + 2 \log(N) - \psi(\Delta)

    where :math:`C_\gamma \approx 0.5772156649\dots` is Euler's constant, :math:`\psi` is the
    digamma function, and :math:`\Delta = N - K` is the number of coincidences
    (repeated observations) in the data.

    This estimator is specifically designed for the extremely undersampled regime
    where :math:`K \sim N` and diverges with N when the data is well-sampled.
    The ANSB estimator requires that :math:`N/K \to 0`, which is checked by default
    using the ``undersampled`` parameter :cite:p:`nemenmanEntropyInformationNeural2004`.

    If there are no coincidences in the data (:math:`\Delta = 0`), ANSB returns NaN
    as the estimator is undefined in this case.

    Parameters
    ----------
    *data : array-like
        The data used to estimate the entropy.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    undersampled : float, default=0.1
        Maximum allowed ratio N/K to consider data sufficiently undersampled.
        A warning is issued if this threshold is exceeded.
    base : LogBaseType, default=Config.get("base")
        The logarithm base for entropy calculation.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.

    Notes
    -----
    The ANSB estimator is based on the asymptotic expansion of the NSB estimator
    for the case of extreme undersampling. It provides a computationally efficient
    alternative to the full NSB estimator when :math:`K \sim N`.

    Examples
    --------
    >>> import infomeasure as im
    >>> data = [1, 2, 3, 4, 5, 1, 2]  # Some repeated values
    >>> im.entropy(data, approach='ansb')
    np.float64(3.353104447353747)
    """

    def __init__(
        self,
        *data,
        K: int = None,
        undersampled: float = 0.1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the ANSB entropy estimator."""
        super().__init__(*data, base=base)
        self.k_given = K
        self.undersampled = undersampled

    def _simple_entropy(self):
        """Calculate the ANSB entropy of the data.

        Returns
        -------
        float or tuple
            The calculated ANSB entropy. If std_dev=True, returns (entropy, std_dev).
        """
        N = self.data[0].N
        K = self.data[0].K if self.k_given is None else self.k_given

        # Check if data is sufficiently undersampled
        ratio = N / K if K > 0 else float("inf")
        if ratio > self.undersampled:
            logger.warning(
                f"Data is not sufficiently undersampled (N/K = {ratio:.3f} > {self.undersampled}), "
                "so calculation may diverge..."
            )

        # Calculate coincidences (number of repeated observations)
        coincidences = N - K

        if coincidences == 0:
            logger.warning("No coincidences in data - ANSB estimator is undefined")
            return float("nan")

        # ANSB formula: (γ - log(2)) + 2 * log(N) - ψ(Δ)
        entropy_nats = (euler_gamma - log(2)) + 2 * log(N) - digamma(coincidences)

        # Convert to the desired base if needed
        if self.base != "e":
            entropy_nats /= log(self.base)

        return entropy_nats

    def _extract_local_values(self):
        """Extract local values for ANSB estimator.

        Raises
        ------
        TheoreticalInconsistencyError
            Local values cannot be meaningfully extracted for ANSB estimator.
        """
        raise TheoreticalInconsistencyError(
            "Local values extraction is not implemented for ANSB estimator. "
            "The ANSB estimator is based on global statistics (coincidences) "
            "and does not provide meaningful local entropy values for individual data points."
        )

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for ANSB estimator.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for ANSB estimator. "
            "The ANSB estimator is designed for single distribution entropy estimation "
            "in the extremely undersampled regime and does not extend to cross-entropy "
            "calculations between different distributions."
        )

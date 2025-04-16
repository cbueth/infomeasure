"""Module for the Renyi mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..entropy.renyi import RenyiEntropyEstimator
from ... import Config
from ...utils.types import LogBaseType


class BaseRenyiMIEstimator(ABC):
    r"""Base class for Renyi mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    normalize : bool, optional
        If True, normalize the data before analysis.
    """

    def __init__(
        self,
        *data,
        cond=None,
        k: int = 4,
        alpha: float | int = None,
        noise_level=1e-8,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the estimator with specific parameters.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        k : int
            The number of nearest neighbors to consider.
        alpha : float | int
            The Renyi parameter, order or exponent.
            Sometimes denoted as :math:`\alpha` or :math:`q`.
        noise_level : float
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        normalize
            If True, normalize the data before analysis.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.

        Raises
        ------
        ValueError
            If the Renyi parameter is not a positive number.
        ValueError
            If the number of nearest neighbors is not a positive integer.
        """
        if cond is None:
            super().__init__(*data, offset=offset, normalize=normalize, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=normalize, base=base
            )
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.alpha = alpha
        self.noise_level = noise_level


class RenyiMIEstimator(BaseRenyiMIEstimator, PValueMixin, MutualInformationEstimator):
    r"""Estimator for the Renyi mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    normalize : bool, optional
        If True, normalize the data before analysis.

    Notes
    -----
    The Rényi entropy is a generalization of Shannon entropy,
    where the small values of probabilities are emphasized for :math:`\alpha < 1`,
    and higher probabilities are emphasized for :math:`\alpha > 1`.
    For :math:`\alpha = 1`, it reduces to Shannon entropy.
    The Rényi-Entropy class can be particularly interesting for systems where additivity
    (in Shannon sense) is not always preserved, especially in nonlinear complex systems,
    such as when dealing with long-range forces.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Renyi mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=RenyiEntropyEstimator,
            noise_level=self.noise_level,
            kwargs={"alpha": self.alpha, "k": self.k, "base": self.base},
        )


class RenyiCMIEstimator(BaseRenyiMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for the conditional Renyi mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    normalize : bool, optional
        If True, normalize the data before analysis.

    Notes
    -----
    The Rényi entropy is a generalization of Shannon entropy,
    where the small values of probabilities are emphasized for :math:`\alpha < 1`,
    and higher probabilities are emphasized for :math:`\alpha > 1`.
    For :math:`\alpha = 1`, it reduces to Shannon entropy.
    The Rényi-Entropy class can be particularly interesting for systems where additivity
    (in Shannon sense) is not always preserved, especially in nonlinear complex systems,
    such as when dealing with long-range forces.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Conditional Renyi mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=RenyiEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, alpha=self.alpha, base=self.base),
        )

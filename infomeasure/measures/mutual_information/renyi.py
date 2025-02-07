"""Module for the Renyi mutual information estimator."""

from abc import ABC
from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..entropy.renyi import RenyiEntropyEstimator


class BaseRenyiMIEstimator(ABC):
    r"""Base class for Renyi mutual information estimators.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
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
        Not compatible with the ``data_z`` parameter / conditional MI.
    normalize : bool, optional
        If True, normalize the data before analysis.
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z=None,
        *,  # all following parameters are keyword-only
        k: int = 4,
        alpha: float | int = None,
        noise_level=1e-8,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with specific parameters.

        Parameters
        ----------
        data_x, data_y : array-like
            The data used to estimate the (conditional) mutual information.
        data_z : array-like, optional
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
            Not compatible with the ``data_z`` parameter / conditional MI.
        base : int | float | "e", optional
            The logarithm base for the transfer entropy calculation.
            The default can be set
            with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

        Raises
        ------
        ValueError
            If the Renyi parameter is not a positive number.
        ValueError
            If the number of nearest neighbors is not a positive integer.
        """
        if data_z is None:
            super().__init__(
                data_x, data_y, offset=offset, normalize=normalize, base=base
            )
        else:
            super().__init__(
                data_x, data_y, data_z, offset=offset, normalize=normalize, base=base
            )
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.alpha = alpha
        self.noise_level = noise_level


class RenyiMIEstimator(
    BaseRenyiMIEstimator, EffectiveValueMixin, MutualInformationEstimator
):
    """Estimator for the Renyi mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
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
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
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
    """Estimator for the conditional Renyi mutual information.

    Attributes
    ----------
    data_x, data_y, data_z : array-like
        The data used to estimate the conditional mutual information.
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
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
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

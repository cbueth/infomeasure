"""Module for the Tsallis mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..entropy.tsallis import TsallisEntropyEstimator
from ... import Config
from ...utils.types import LogBaseType


class BaseTsallisMIEstimator(ABC):
    r"""Base class for Tsallis mutual information estimators.

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
    q : float
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.
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
        q: float | int = None,
        noise_level: float = 1e-8,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the BaseTsallisMIEstimator.

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
        q : float | int
            The Tsallis parameter, order or exponent.
            Sometimes denoted as :math:`q`,
            analogous to the Rényi parameter :math:`\alpha`.
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
            If the Tsallis parameter is not a positive number.
        ValueError
            If the number of nearest neighbors is not a positive integer.
        """
        if cond is None:
            super().__init__(*data, offset=offset, normalize=normalize, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=normalize, base=base
            )

        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError("The Tsallis parameter ``q`` must be a positive number.")
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )

        self.k = k
        self.q = q
        self.noise_level = noise_level


class TsallisMIEstimator(
    BaseTsallisMIEstimator, PValueMixin, MutualInformationEstimator
):
    r"""Estimator for the Tsallis mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.
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
    In the :math:`q \to 1` limit, the Jackson sum (q-additivity) reduces to
    ordinary summation,
    and the Tallis entropy reduces to Shannon Entropy.
    This class of entropy measure is in particularly useful in the study in connection
    with long-range correlated systems and with non-equilibrium phenomena.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Tsallis mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=TsallisEntropyEstimator,
            noise_level=self.noise_level,
            kwargs={"q": self.q, "k": self.k, "base": self.base},
        )


class TsallisCMIEstimator(
    BaseTsallisMIEstimator, ConditionalMutualInformationEstimator
):
    r"""Estimator for the conditional Tsallis mutual information.

    Parameters
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    k : int
        The number of nearest neighbors to consider.
    q : float | int
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`,
        analogous to the Rényi parameter :math:`\alpha`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    normalize
        If True, normalize the data before analysis.

    Notes
    -----
    In the :math:`q \to 1` limit, the Jackson sum (q-additivity) reduces to
    ordinary summation,
    and the Tallis entropy reduces to Shannon Entropy.
    This class of entropy measure is in particularly useful in the study in connection
    with long-range correlated systems and with non-equilibrium phenomena.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Tsallis conditional mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=TsallisEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(q=self.q, k=self.k, base=self.base),
        )

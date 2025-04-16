"""Module for the Ordinal / Permutation mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer, ndarray

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..utils.discrete_interaction_information import (
    mutual_information_global,
    mutual_information_local,
    conditional_mutual_information_global,
    conditional_mutual_information_local,
)
from ..utils.ordinal import symbolize_series, reduce_joint_space
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class BaseOrdinalMIEstimator(ABC):
    r"""Base class for the Ordinal mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    embedding_dim : int
        The size of the permutation patterns.
    stable : bool, optional
        If True, when sorting the data, the order of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.

    Notes
    -----
    - The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
      There is no ``normalize`` option, as this would not influence the order of the data.
    - If ``embedding_dim`` is set to 1, the mutual information is always 0.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If ``offset`` and ``embedding_dim`` are such that the data is too small.
    TypeError
        If the data are not 1d array-like(s).
    """

    def __init__(
        self,
        *data,
        cond=None,
        embedding_dim: int = None,
        stable: bool = False,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the OrdinalMIEstimator.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        embedding_dim : int
            The embedding dimension of the Ordinal entropy.
        stable : bool, optional
            If True, when sorting the data, the order of equal elements is preserved.
            This can be useful for reproducibility and testing, but might be slower.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        if cond is None:
            super().__init__(*data, offset=offset, normalize=False, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=False, base=base
            )
            if self.cond.ndim > 1:
                raise TypeError(
                    "The conditional variable must be an 1d array, "
                    "so that ordinal patterns can be computed from it."
                )
        if not issubdtype(type(embedding_dim), integer) or embedding_dim < 0:
            raise ValueError("The embedding_dim must be a non-negative integer.")
        if embedding_dim == 1:
            logger.warning(
                "The Ordinal mutual information is always 0 for embedding_dim=1. "
                "Consider using a higher embedding_dim for more meaningful results."
            )
        if any(var.ndim > 1 for var in self.data):
            raise TypeError(
                "The data must be tuples of 1D arrays. "
                "Ordinal patterns can only be computed from 1D arrays."
            )
        self.embedding_dim = embedding_dim
        if len(self.data[0]) < (embedding_dim - 1) + 1:
            raise ValueError("The data is too small for the given embedding_dim.")
        self.stable = stable

        self.symbols = [
            reduce_joint_space(
                symbolize_series(
                    var, self.embedding_dim, stable=self.stable, to_int=False
                )
            )
            for var in self.data
        ]  # Convert permutation tuples to integers for efficiency (reduce_joint_space),
        # so mutual_information_global can use crosstab method internally


class OrdinalMIEstimator(
    BaseOrdinalMIEstimator, PValueMixin, MutualInformationEstimator
):
    r"""Estimator for the Ordinal mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    embedding_dim : int
        The size of the permutation patterns.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.

    Notes
    -----
    - The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
      There is no ``normalize`` option, as this would not influence the order of the data.
    - If ``embedding_dim`` is set to 1, the mutual information is always 0.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If ``offset`` and ``embedding_dim`` are such that the data is too small.
    """

    def _calculate(self) -> float:
        """Calculate the mutual information of the data."""

        if self.embedding_dim == 1:
            return 0.0

        return mutual_information_global(*self.symbols, log_func=self._log_base)

    def _extract_local_values(self) -> ndarray[float]:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of mi.
        """
        return mutual_information_local(*self.symbols, log_func=self._log_base)


class OrdinalCMIEstimator(
    BaseOrdinalMIEstimator, ConditionalMutualInformationEstimator
):
    """Estimator for the Ordinal conditional mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    embedding_dim : int
        The size of the permutation patterns.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.
    symbols_cond : array-like, shape (n_samples,)
        The symbolized conditional data used to estimate the
        conditional mutual information.

    Notes
    -----
    - The order will be determined via :func:`numpy.argsort() <numpy.argsort>`.
      There is no ``normalize`` option, as this would not influence the order of the data.
    - If ``embedding_dim`` is set to 1, the mutual information is always 0.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If ``offset`` and ``embedding_dim`` are such that the data is too small.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbols_cond = reduce_joint_space(
            symbolize_series(
                self.cond, self.embedding_dim, stable=self.stable, to_int=False
            )
        )

    def _calculate(self) -> float:
        """Calculate the conditional mutual information of the data."""

        if self.embedding_dim == 1:
            return 0.0

        return conditional_mutual_information_global(
            *self.symbols,
            cond=self.symbols_cond,
            log_func=self._log_base,
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return conditional_mutual_information_local(
            *self.symbols, cond=self.symbols_cond, log_func=self._log_base
        )

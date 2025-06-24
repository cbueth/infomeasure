"""Module for the discrete Miller-Madow mutual information estimator."""

from abc import ABC

from numpy import ndarray

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
    DiscreteMixin,
)
from ..utils.discrete_interaction_information import (
    mutual_information_global,
    mutual_information_local,
    conditional_mutual_information_global,
    conditional_mutual_information_local,
)
from ... import Config
from ...utils.types import LogBaseType


class BaseMillerMadowMIEstimator(DiscreteMixin, ABC):
    """Base class for discrete Miller-Madow mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    """

    def __init__(
        self,
        *data,
        cond=None,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseMillerMadowMIEstimator.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        self.data: tuple[ndarray] = None
        if cond is None:
            super().__init__(*data, offset=offset, normalize=False, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=False, base=base
            )
        self._check_data_mi()


class MillerMadowMIEstimator(
    BaseMillerMadowMIEstimator, PValueMixin, MutualInformationEstimator
):
    r"""Estimator for the discrete Miller-Madow mutual information.

    .. math::

        \begin{align}\hat{I}_{\tiny{MM}}(X; Y)
        &= \hat{H}_{\tiny{MM}}(X) + \hat{H}_{\tiny{MM}}(Y) - \hat{H}_{\tiny{MM}}(X, Y)\\
        &= \hat{H}_{\tiny{MLE}}(X) + \hat{H}_{\tiny{MLE}}(Y) - \hat{H}_{\tiny{MLE}}(X, Y)
          + (K_X + K_Y - K_{XY} - 1)/(2N \cdot \log(\texttt{base}))\\
        &= \hat{I}_{\tiny{MLE}}(X; Y) + (K_X + K_Y - K_{XY} - 1)/(2N \cdot \log(\texttt{base}))\\
        \end{align}

    For an arbitrary number of random variables this is equivalent to:

    .. math::

        \begin{align}\hat{I}_{\tiny{MM}}(X_1; \dots; X_n)
        &= \hat{I}_{\tiny{MLE}}(X_1; \dots; X_n)
        + \frac{\left(\sum_{i=1}^{n}K_i-1\right) - \left(K_{1,\dots,i}-1\right)}{2N \cdot \log(\texttt{base})}
        \end{align}

    :math:`\hat{I}_{\tiny{MLE}}(X_1; \dots; X_n)` is the
    initial :class:`~infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator` estimate,
    :math:`K_i` is the number of unique values in the i-th variable,
    :math:`K_{1,\dots,i}` is the number of unique joint values,
    and :math:`N` is the number of samples.


    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    """

    def _calculate(self):
        """Calculate the Miller-Madow mutual information of the data.

        Returns
        -------
        float
            The calculated mutual information.
        """
        return mutual_information_global(
            *self.data, log_func=self._log_base, miller_madow_correction=self.base
        )

    def _extract_local_values(self) -> ndarray[float]:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of mi.
        """
        return mutual_information_local(
            *self.data, log_func=self._log_base, miller_madow_correction=self.base
        )


class MillerMadowCMIEstimator(
    BaseMillerMadowMIEstimator, ConditionalMutualInformationEstimator
):
    r"""Estimator for the discrete Miller-Madow conditional mutual information.

    .. math::

        \begin{align}
        \hat{I}_{\tiny{MM}}(X_1; X_2; \ldots; X_n \mid Z)&=
        \hat{I}_{\tiny{MLE}}(\dots)
        + \frac{\left(\sum_{i=1}^{n}K_{iZ}-1\right) - \left(K_{1,\dots,i,Z}-1\right)
         - \left(K_{Z}-1\right)}{2N \cdot \log(\texttt{base})}
        \end{align}

    :math:`\hat{I}_{\tiny{MLE}}(X_1; \dots; X_n \mid Z)` is the
    initial :class:`~infomeasure.estimators.mutual_information.discrete.DiscreteCMIEstimator` estimate,
    :math:`K_{iZ}` is the number of unique values in the i-th variable joint with Z,
    :math:`K_{1,\dots,i,Z}` is the number of unique joint values,
    :math:`K_Z` is the number of unique values in the Z variable,
    and :math:`N` is the number of samples.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            The calculated conditional mutual information.
        """
        return conditional_mutual_information_global(
            *self.data,
            cond=self.cond,
            log_func=self._log_base,
            miller_madow_correction=self.base,
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return conditional_mutual_information_local(
            *self.data,
            cond=self.cond,
            log_func=self._log_base,
            miller_madow_correction=self.base,
        )

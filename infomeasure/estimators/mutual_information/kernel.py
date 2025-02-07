"""Module for the kernel-based mutual information estimator."""

from abc import ABC
from numpy import finfo, column_stack, ndarray
from numpy import newaxis

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..utils.kde import kde_probability_density_function


class BaseKernelMIEstimator(ABC):
    """Base class for mutual information using Kernel Density Estimation (KDE).

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``data_z`` parameter / conditional MI.
    normalize : bool, optional
        If True, normalize the data before analysis.
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z=None,
        *,  # all following parameters are keyword-only
        bandwidth: float | int = None,
        kernel: str = None,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with specific bandwidth and kernel.

        Parameters
        ----------
        data_x, data_y : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
        data_z : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``data_z`` parameter / conditional MI.
        normalize
            If True, normalize the data before analysis.
        """
        self.data_y = None
        self.data_x = None
        if data_z is None:
            super().__init__(
                data_x, data_y, offset=offset, normalize=normalize, base=base
            )
        else:
            super().__init__(
                data_x, data_y, data_z, offset=offset, normalize=normalize, base=base
            )
            # Ensure self.data_z is a 2D array
            if self.data_z.ndim == 1:
                self.data_z = self.data_z[:, newaxis]
        self.bandwidth = bandwidth
        self.kernel = kernel
        # Ensure self.data_x and self.data_y are 2D arrays
        if self.data_x.ndim == 1:
            self.data_x = self.data_x[:, newaxis]
        if self.data_y.ndim == 1:
            self.data_y = self.data_y[:, newaxis]


class KernelMIEstimator(
    BaseKernelMIEstimator, EffectiveValueMixin, MutualInformationEstimator
):
    """Estimator for mutual information using Kernel Density Estimation (KDE).

    .. math::

        I(X;Y) = \\sum_{i=1}^{n} p(x_i, y_i) \\log
                 \\left( \frac{p(x_i, y_i)}{p(x_i)p(y_i)} \right)

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    normalize : bool, optional
        If True, normalize the data before analysis.
    """

    def _calculate(self) -> ndarray:
        """Calculate the mutual information of the data.

        Returns
        -------
        local_mi_values : array
            The mutual information between the two datasets.
        average_mi : float
            The average mutual information between the two datasets.
        """
        # Combine data into a joint dataset
        joint_data = column_stack([self.data_x, self.data_y])

        # Compute joint density using KDE for each point in the joint data
        # densities.shape=(n_points, 3)
        # densities[i] = [p(x_i, y_i), p(x_i), p(y_i)]
        densities = column_stack(
            [
                kde_probability_density_function(
                    joint_data, self.bandwidth, kernel=self.kernel
                ),
                kde_probability_density_function(
                    self.data_x, self.bandwidth, kernel=self.kernel
                ),
                kde_probability_density_function(
                    self.data_y, self.bandwidth, kernel=self.kernel
                ),
            ]
        )

        # Avoid division by zero or log of zero by replacing zeros with a small positive value
        # TODO: Make optional
        densities[densities == 0] = finfo(float).eps

        # Compute local mutual information values
        local_mi_values = self._log_base(
            densities[:, 0] / (densities[:, 1] * densities[:, 2])
        )
        return local_mi_values


class KernelCMIEstimator(BaseKernelMIEstimator, ConditionalMutualInformationEstimator):
    """Estimator for conditional mutual information using
    Kernel Density Estimation (KDE).

    .. math::

        I(X;Y|Z) = \\sum_{i=1}^{n} p(x_i, y_i, z_i) \\log
                   \\left( \frac{p(z_i)p(x_i, y_i, z_i)}{p(x_i, z_i)p(y_i, z_i)} \right)

    Attributes
    ----------
    data_x, data_y, data_z : array-like
        The data used to estimate the conditional mutual information.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    normalize : bool, optional
        If True, normalize the data before analysis.
    """

    def _calculate(self) -> ndarray:
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        local_mi_values : array
            The mutual information between the two datasets.
        """
        # Combine data into a joint dataset
        joint_all = column_stack([self.data_x, self.data_y, self.data_z])

        # Compute densities for all points in the joint dataset
        # densities.shape=(n_points, 4)
        # densities[i] = [p(x_i, y_i, z_i), p(x_i, z_i), p(y_i, z_i), p(z_i)]
        densities = column_stack(
            [
                kde_probability_density_function(
                    joint_all, self.bandwidth, kernel=self.kernel
                ),
                kde_probability_density_function(
                    joint_all[:, [0, 2]], self.bandwidth, kernel=self.kernel
                ),
                kde_probability_density_function(
                    joint_all[:, [1, 2]], self.bandwidth, kernel=self.kernel
                ),
                kde_probability_density_function(
                    joint_all[:, 2, newaxis], self.bandwidth, kernel=self.kernel
                ),
            ]
        )

        # Avoid division by zero or log of zero by replacing zeros with a small positive value
        # TODO: Make optional
        densities[densities == 0] = finfo(float).eps

        # Compute local mutual information values
        local_mi_values = self._log_base(
            (densities[:, 3] * densities[:, 0]) / (densities[:, 1] * densities[:, 2])
        )
        return local_mi_values

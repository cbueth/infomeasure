"""Module for the kernel-based mutual information estimator."""

from abc import ABC

from numpy import column_stack, ndarray, prod
from numpy import newaxis

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..utils.array import assure_2d_data
from ..utils.kde import kde_probability_density_function
from ... import Config
from ...utils.types import LogBaseType


class BaseKernelMIEstimator(ABC):
    """Base class for mutual information using Kernel Density Estimation (KDE).

    Attributes
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
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
        bandwidth: float | int = None,
        kernel: str = None,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with specific bandwidth and kernel.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        normalize
            If True, normalize the data before analysis.
        """
        self.data: tuple[ndarray] = None
        self.cond = None
        if cond is None:
            super().__init__(*data, offset=offset, normalize=normalize, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=normalize, base=base
            )
            # Ensure self.cond is a 2D array
            self.cond = assure_2d_data(self.cond)
        self.bandwidth = bandwidth
        self.kernel = kernel
        # Ensure self.data are 2D arrays
        self.data = tuple(assure_2d_data(var) for var in self.data)


class KernelMIEstimator(BaseKernelMIEstimator, PValueMixin, MutualInformationEstimator):
    """Estimator for mutual information using Kernel Density Estimation (KDE).

    .. math::

        I(X;Y) = \\sum_{i=1}^{n} p(x_i, y_i) \\log
                 \\left( \frac{p(x_i, y_i)}{p(x_i)p(y_i)} \right)

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
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
        """
        # Combine data into a joint dataset
        joint_data = column_stack([*self.data])

        # Compute joint density using KDE for each point in the joint data
        # densities.shape=(n_points, len(data) + 1)
        # densities[i] = [p(x_i, y_i, ...), p(x_i), p(y_i), ...]
        densities = column_stack(
            [
                kde_probability_density_function(
                    joint_data, self.bandwidth, kernel=self.kernel
                ),
                *(
                    kde_probability_density_function(
                        var, self.bandwidth, kernel=self.kernel
                    )
                    for var in self.data
                ),
            ]
        )

        # Compute local mutual information values
        local_mi_values = self._log_base(  # p(x_i, y_i, ...) / (p(x_i) * p(y_i) * ...)
            densities[:, 0] / prod(densities[:, 1:], axis=1)
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
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
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
        joint_all = column_stack([*self.data, self.cond])

        # Compute densities for all points in the joint dataset
        # densities.shape=(n_points, len(data) + 2)
        # densities[i] = [p(x_i, y_i, ..., cond), p(x_i, cond), p(y_i, z_i), .., p(z_i)]
        densities = column_stack(
            [
                kde_probability_density_function(
                    joint_all, self.bandwidth, kernel=self.kernel
                ),
                *(
                    kde_probability_density_function(
                        joint_all[:, [i, -1]], self.bandwidth, kernel=self.kernel
                    )
                    for i in range(len(self.data))
                ),
                kde_probability_density_function(
                    joint_all[:, -1, newaxis], self.bandwidth, kernel=self.kernel
                ),
            ]
        )

        # Compute local mutual information values
        local_mi_values = self._log_base(
            (densities[:, 0] * densities[:, -1]) / prod(densities[:, 1:-1], axis=1)
        )  # p(x_i, y_i, ..., cond) * p(z_i) / (p(x_i, cond) * p(y_i, cond) * ...)
        return local_mi_values

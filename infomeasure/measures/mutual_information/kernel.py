"""Module for the kernel-based mutual information estimator."""

from numpy import array, finfo, hstack
from numpy import mean as np_mean
from numpy import newaxis

from ... import Config
from ...utils.types import LogBaseType
from ..base import EffectiveValueMixin, MutualInformationEstimator
from ..utils.kde import kde_probability_density_function


class KernelMIEstimator(EffectiveValueMixin, MutualInformationEstimator):
    """Estimator for mutual information using Kernel Density Estimation (KDE).

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
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

    def __init__(
        self,
        data_x,
        data_y,
        bandwidth: float | int,
        kernel: str,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with specific bandwidth and kernel.

        Parameters
        ----------
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
        offset : int, optional
            Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        normalize
            If True, normalize the data before analysis.
        """
        super().__init__(data_x, data_y, offset=offset, normalize=normalize, base=base)
        self.bandwidth = bandwidth
        self.kernel = kernel

        # Ensure self.data_x and self.data_y are 2D arrays
        if self.data_x.ndim == 1:
            self.data_x = self.data_x[:, newaxis]
        if self.data_y.ndim == 1:
            self.data_y = self.data_y[:, newaxis]

    def _calculate(self) -> tuple:
        """Calculate the mutual information of the data.

        Returns
        -------
        local_mi_values : array
            The mutual information between the two datasets.
        average_mi : float
            The average mutual information between the two datasets.

        """

        # Combine source and dest data for joint density estimation
        # joint_data = np.vstack([self.data_x.ravel(), self.data_y.ravel()]).T

        # Combine source and dest data for joint density estimation
        joint_data = hstack([self.data_x, self.data_y])

        # Compute joint density using KDE for each point in the joint data
        joint_density = array(
            [
                kde_probability_density_function(
                    joint_data, joint_data[i], self.bandwidth, self.kernel
                )
                for i in range(joint_data.shape[0])
            ]
        )

        # Compute individual densities for source and dest data
        source_density = array(
            [
                kde_probability_density_function(
                    self.data_x, self.data_x[i], self.bandwidth, self.kernel
                )
                for i in range(self.data_x.shape[0])
            ]
        )
        dest_density = array(
            [
                kde_probability_density_function(
                    self.data_y, self.data_y[i], self.bandwidth, self.kernel
                )
                for i in range(self.data_y.shape[0])
            ]
        )

        # Avoid division by zero or log of zero by replacing zeros with a small positive value
        # TODO: Make optional
        joint_density[joint_density == 0] = finfo(float).eps
        source_density[source_density == 0] = finfo(float).eps
        dest_density[dest_density == 0] = finfo(float).eps

        # Compute local mutual information values
        local_mi_values = self._log_base(
            joint_density / (source_density * dest_density)
        )
        average_mi = np_mean(local_mi_values)  # Global mutual information

        return average_mi, local_mi_values

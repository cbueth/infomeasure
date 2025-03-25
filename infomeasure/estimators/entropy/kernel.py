"""Module for the kernel entropy estimator."""

from numpy import column_stack

from ..base import EntropyEstimator
from ..utils.array import assure_2d_data
from ..utils.kde import kde_probability_density_function
from ... import Config
from ...utils.types import LogBaseType


class KernelEntropyEstimator(EntropyEstimator):
    """Estimator for entropy (Shannon) using Kernel Density Estimation (KDE).

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    """

    def __init__(
        self,
        data,
        *,  # all following parameters are keyword-only
        bandwidth: float | int,
        kernel: str,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the KernelEntropyEstimator.

        Parameters
        ----------
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
        """
        super().__init__(data, base=base)
        self.data = assure_2d_data(data)
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        array-like
            The local form of the entropy.
        """
        # Compute the KDE densities
        densities = kde_probability_density_function(
            self.data, self.bandwidth, kernel=self.kernel
        )

        # Compute the log of the densities
        return -self._log_base(densities)

    def _joint_entropy(self):
        """Calculate the joint entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        array-like
            The local form of the joint entropy.
        """
        self.data = column_stack(self.data)
        return self._simple_entropy()

"""Module for the kernel entropy estimator."""

from numpy import column_stack, sum as np_sum, isnan, nan

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, WorkersMixin
from ..utils.array import assure_2d_data
from ..utils.kde import kde_probability_density_function


class KernelEntropyEstimator(WorkersMixin, EntropyEstimator):
    """Estimator for entropy (Shannon) using Kernel Density Estimation (KDE).

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    workers : int, optional
       Number of workers to use for parallel processing.
       Default is 1, meaning no parallel processing.
       If set to -1, all available CPU cores will be used.

    Notes
    -----
    A small ``bandwidth`` can lead to under-sampling,
    while a large ``bandwidth`` may over-smooth the data, obscuring details.
    """

    def __init__(
        self,
        *data,
        bandwidth: float | int,
        kernel: str,
        workers: int = 1,
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
        workers : int, optional
           Number of workers to use for parallel processing.
           Default is 1, meaning no parallel processing.
           If set to -1, all available CPU cores will be used.
        """
        super().__init__(*data, workers=workers, base=base)
        self.data = tuple(assure_2d_data(var) for var in self.data)
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
            self.data[0], self.bandwidth, kernel=self.kernel, workers=self.n_workers
        )
        densities[densities == 0] = nan
        # Compute the log of the densities
        log_densities = -self._log_base(densities)
        log_densities[isnan(log_densities)] = 0
        return log_densities

    def _joint_entropy(self):
        """Calculate the joint entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        array-like
            The local form of the joint entropy.
        """
        self.data = (column_stack(self.data[0]),)
        return self._simple_entropy()

    def _cross_entropy(self) -> float:
        """Calculate the cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        # Compute the KDE densities
        densities = kde_probability_density_function(
            self.data[1],
            self.bandwidth,
            at=self.data[0],
            kernel=self.kernel,
            workers=self.n_workers,
        )
        # Compute the log of the densities
        return -np_sum(self._log_base(densities[densities > 0])) / len(densities)

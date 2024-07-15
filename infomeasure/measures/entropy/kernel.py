"""Module for the kernel entropy estimator."""

from numpy import array, finfo, newaxis
from numpy import mean as np_mean

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, LogBaseMixin
from ..utils.kde import kde_probability_density_function


class KernelEntropyEstimator(LogBaseMixin, EntropyEstimator):
    """Estimator for entropy (Shannon) using Kernel Density Estimation (KDE).

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Methods
    -------
    calculate()
        Calculate the entropy.
    """

    def __init__(
        self,
        data,
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
            implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
        """
        super().__init__(data, base=base)
        if self.data.ndim == 1:
            self.data = self.data[:, newaxis]
        self.bandwidth = bandwidth
        self.kernel = kernel

    def calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """

        densities = array(
            [
                kde_probability_density_function(
                    self.data, self.data[i], self.bandwidth, kernel=self.kernel
                )
                for i in range(self.data.shape[0])
            ]
        )

        # Replace densities of 0 with a small number to avoid log(0)
        # TODO: Make optional
        densities[densities == 0] = finfo(float).eps

        # Compute the log of the densities
        log_densities = self._log_base(densities)

        # Compute the entropy
        entropy = -np_mean(log_densities)

        return entropy

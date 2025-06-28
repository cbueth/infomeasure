"""Module for the kernel entropy estimator."""

from numpy import column_stack, sum as np_sum, isnan, nan

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator
from ..mixins import WorkersMixin
from ..utils.array import assure_2d_data
from ..utils.kde import kde_probability_density_function


class KernelEntropyEstimator(WorkersMixin, EntropyEstimator):
    r"""Kernel entropy estimator for continuous data using Kernel Density Estimation (KDE).

    The kernel entropy estimator computes the differential Shannon entropy by estimating
    the probability density function using kernel density estimation:

    .. math::

        \hat{H}(X) = -\int \hat{f}(x) \log \hat{f}(x) \, dx \approx -\frac{1}{N} \sum_{i=1}^{N} \log \hat{f}(x_i)

    where :math:`\hat{f}(x)` is the kernel density estimate:

    .. math::

        \hat{f}(x) = \frac{1}{N h^d} \sum_{i=1}^{N} K\left(\frac{x - x_i}{h}\right)

    with :math:`K(\cdot)` being the kernel function, :math:`h` the bandwidth parameter,
    :math:`d` the dimensionality, and :math:`N` the number of data points.

    For joint entropy of multiple variables, the estimator concatenates the variables
    into a single multivariate space and applies the same KDE approach.

    The estimator supports both Gaussian and box (uniform) kernels. The choice of
    bandwidth is critical: small values can lead to under-smoothing and overfitting,
    while large values may over-smooth the data and obscure important features
    :cite:p:`silverman1986density,garcia-portuguesChapter2Kernel2025`.

    Parameters
    ----------
    *data : array-like
        The continuous data used to estimate the entropy. For univariate entropy,
        pass a single array. For joint entropy, pass multiple arrays.
    bandwidth : float | int
        The bandwidth parameter for the kernel. Controls the smoothness of the
        density estimate.
    kernel : str
        Type of kernel to use. Supported options are:

        - ``'gaussian'``: Gaussian (normal) kernel
        - ``'box'``: Box (uniform) kernel

        Compatible with the KDE implementation
        :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    workers : int, optional
        Number of workers to use for parallel processing. Default is 1 (no parallelization).
        If set to -1, all available CPU cores will be used.
    base : float | str, optional
        Logarithm base for entropy calculation. Default is from global configuration.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use.
    workers : int
        Number of workers to use for parallel processing.

    Returns
    -------
    array-like
        Local entropy values for each data point when calling entropy calculation methods.
        The mean of these values gives the overall entropy estimate.

    Notes
    -----
    **Bandwidth Selection**: The bandwidth parameter critically affects the quality of
    the entropy estimate. A small bandwidth can lead to under-sampling and high variance,
    while a large bandwidth may over-smooth the data, obscuring important details and
    introducing bias.

    **Kernel Choice**:

    - Gaussian kernels provide smooth density estimates and are theoretically well-founded
    - Box kernels are computationally efficient and provide non-parametric estimates

    **Computational Complexity**: The algorithm has O(N²) complexity for box kernels
    using KDTree queries, and varies for Gaussian kernels depending on the implementation.

    **Cross-entropy**: Supported between two distributions by evaluating the density
    of the second distribution at points from the first distribution.

    Examples
    --------
    >>> import infomeasure as im
    >>> from numpy.random import default_rng
    >>> rng = default_rng(281769)
    >>> # Generate sample data
    >>> data = rng.normal(0, 1, 1000)
    >>>
    >>> # Create estimator
    >>> estimator = im.estimator(data, measure="h", approach="kernel", bandwidth=0.5, kernel='gaussian')
    >>>
    >>> # Calculate entropy
    >>> estimator.result()
    np.float64(1.366015332652949)
    >>> # Local values
    >>> estimator.local_vals()
    array([1.54017083, 1.35855839, 0.97949819, 0.97333173, 2.62084886,
       ...
       1.08174049, 0.97418054, 1.88055967, 0.99614516, 0.98548583])


    See Also
    --------
    infomeasure.estimators.utils.kde.kde_probability_density_function :
        Underlying KDE implementation
    infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator :
        For discrete data entropy estimation
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

"""Kernel Density Estimation (KDE) utilities."""

from numpy import abs as np_abs, ndarray, dot
from numpy import all as np_all
from numpy import argsort, asarray, cov, inf, issubdtype, newaxis, number
from numpy import sum as np_sum
from numpy.linalg import eig
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde


def kde_probability_density_function(data, bandwidth, kernel="box"):
    """
    Estimate the probability density function for a given data set using
    Kernel Density Estimation (KDE).

    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    bandwidth : float
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use ('gaussian' or 'box').

    Returns
    -------
    ndarray[float]
        KDE at the given point(s).

    Raises
    ------
    ValueError
        If the kernel type is not supported
    ValueError
        If the bandwidth is not a positive number.
    """
    if not issubdtype(type(bandwidth), number) or bandwidth <= 0:
        raise ValueError("The bandwidth must be a positive number.")

    if kernel == "gaussian":
        return gaussian_kernel_densities(data.T, bandwidth)
    elif kernel == "box":
        # Get the number of data points (N) and the number of dimensions (d)
        N, d = data.shape

        # Calculate the volume of the box kernel
        volume = bandwidth**d

        tree = KDTree(data)
        counts = tree.query_ball_point(data, bandwidth / 2, p=inf, return_length=True)
        densities = counts / (N * volume)

        # Squeeze the densities array to remove any single-dimensional entries
        return densities.squeeze()
    # Another approach to box kernel density estimation
    else:  # TODO: Add more kernel types as needed
        raise ValueError(f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'.")


def gaussian_kernel_densities(data, bandwidth, eigen_threshold: float = 1e-10):
    """Calculate kde for gaussian kernel.

    In case of multivariate data, checks rank of data and reduces dimensions
    if eigenvalues are below threshold.
    If already full rank, does no reprojection.

    Parameters
    ----------
    data : ndarray, shape (d, N)
        Data points to estimate density for.
    bandwidth : float
        Bandwidth parameter for kernel density estimation.
    eigen_threshold : float, optional
        Threshold for eigenvalues to determine rank of data. Default is 1e-10.

    Returns
    -------
    densities : ndarray, shape (n,)
        Estimated density values at data points.
    """
    if data.shape[0] > 1:  # Multivariate case
        # Calculate covariance matrix
        covariance_matrix = cov(data)
        # Get eigenvalues and eigenvectors
        values, vectors = eig(covariance_matrix)
        sorted_indices = argsort(values)[::-1]
        values_sorted = values[sorted_indices]
        vectors_sorted = vectors[:, sorted_indices]
        # Get the number of eigenvalues greater than the threshold
        num_non_zero_eigenvalues = np_sum(values_sorted > eigen_threshold)
        # Check projection necessary
        if num_non_zero_eigenvalues < data.shape[0]:
            # Project the data onto the reduced space
            pca_components = vectors_sorted[:, :num_non_zero_eigenvalues]
            data_projected = dot(data.T, pca_components).T
            kde = gaussian_kde(data_projected, bw_method=bandwidth)
            return kde.evaluate(data_projected).squeeze()

    kde = gaussian_kde(data, bw_method=bandwidth)
    return kde.evaluate(data).squeeze()

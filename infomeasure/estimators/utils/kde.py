"""Kernel Density Estimation (KDE) utilities."""

from multiprocessing import Pool, cpu_count

from numpy import (
    argsort,
    array_split,
    concatenate,
    cov,
    dot,
    inf,
    issubdtype,
    number,
)
from numpy import sum as np_sum
from numpy.linalg import eig
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from ...utils.config import logger


def kde_probability_density_function(
    data, bandwidth, at=None, kernel="box", workers=-1
):
    """
    Estimate the probability density function for a given data set using
    Kernel Density Estimation (KDE).

    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    bandwidth : float
        The bandwidth for the kernel.
    at : array, optional
        A numpy array of points at which to evaluate the KDE.
        If None, the KDE is evaluated at the data points.
    kernel : str
        Type of kernel to use ('gaussian' or 'box').
    workers : int
        Number of parallel processes to use.
        -1: Use all available CPU cores.
        Default is 1.

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
    logger.debug(
        f"Called kde_probability_density_function with "
        f"kernel: {kernel}, workers: {workers}"
    )
    if not issubdtype(type(bandwidth), number) or bandwidth <= 0:
        raise ValueError("The bandwidth must be a positive number.")

    if at is None:
        at = data
    elif data.shape[1] != at.shape[1]:
        # make sure ``data`` and ``at`` share the same dimensionality
        raise ValueError("The data and at must have the same number of dimensions.")

    if kernel == "gaussian":
        if workers == -1:
            workers = cpu_count()
        return gaussian_kernel_densities(data.T, bandwidth, at=at.T, workers=workers)
    elif kernel == "box":
        # Get the number of data points (N) and the number of dimensions (d)
        N, d = data.shape

        # Calculate the volume of the box kernel
        volume = bandwidth**d

        logger.debug("Creating KDTree from data points for box KDE... ")
        tree = KDTree(data)
        logger.debug(
            f"KDTree created with {N} data points and {d} dimensions. "
            f"Querying KDTree for ball points with {workers} workers."
        )
        counts = tree.query_ball_point(
            at,
            bandwidth / 2,
            p=inf,
            return_length=True,
            workers=workers,
        )
        densities = counts / (N * volume)

        # Squeeze the densities array to remove any single-dimensional entries
        return densities.squeeze()
    # Another approach to box kernel density estimation
    else:  # TODO: Add more kernel types as needed
        raise ValueError(f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'.")


def gaussian_kernel_densities(
    data, bandwidth, at=None, workers=1, eigen_threshold: float = 1e-10
):
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
    at : array, optional
        A numpy array of points at which to evaluate the KDE.
        If None, the KDE is evaluated at the data points.
    workers : int, optional
        Number of workers to use for parallel processing. Default is 1.
    eigen_threshold : float, optional
        Threshold for eigenvalues to determine rank of data. Default is 1e-10.

    Returns
    -------
    densities : ndarray, shape (n,)
        Estimated density values at data points.
    """
    logger.debug(f"Calculating Gaussian KDE with bandwidth {bandwidth}...")
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
            logger.debug(
                f"Reducing dimensionality from {data.shape[1]} to "
                f"{num_non_zero_eigenvalues} dimensions."
            )
            # Project the data onto the reduced space
            pca_components = vectors_sorted[:, :num_non_zero_eigenvalues]
            data_projected = dot(data.T, pca_components).T
            logger.debug("Reprojected data, evaluate kde...")
            # each worker gets a chunk of data and evaluates KDE on it
            return parallel_kde_evaluate(
                data_projected,
                data_projected if at is None else dot(at.T, pca_components).T,
                bandwidth,
                workers,
            )

    at = data if at is None else at
    return parallel_kde_evaluate(data, at, bandwidth, workers)


def query_chunk(params):
    """Evaluate KDE on a chunk of data."""
    full_data, query_data, bandwidth = params
    kde = gaussian_kde(full_data, bw_method=bandwidth)
    return kde.evaluate(query_data).squeeze()


def parallel_kde_evaluate(data, at, bandwidth, workers):
    """Evaluate KDE on a set of data in parallel.

    Parameters
    ----------
    data : array-like
        The data to evaluate the KDE on.
    at : array-like
        The points at which to evaluate the KDE.
    bandwidth : float or str
       The bandwidth to use for the KDE.
    workers : int
        The number of worker processes to use for evaluation.

    Notes
    -----
    If the data is < 100000 samples or the number of workers is 1,
    evaluate the KDE on a single worker.
    """
    # parallelization just gets really effective, if chunk size is not too small
    # if data is not too large, use less workers than possible
    if workers == 1 or data.shape[1] < 20000:
        logger.debug(f"Evaluating kde on a single worker with query size {at.shape}.")
        kde = gaussian_kde(data, bw_method=bandwidth)
        return kde.evaluate(at).squeeze()

    workers = min(workers, max(1, at.shape[1] // 8000))
    query_chunks = array_split(at, workers, axis=1)
    logger.debug(
        f"Evaluating kde on {workers} workers with data size {at.shape} and "
        f"query chunks shape: {query_chunks[0].shape}."
    )
    pool = Pool(processes=workers)
    results = pool.map(
        query_chunk,
        zip([data] * len(query_chunks), query_chunks, [bandwidth] * len(query_chunks)),
    )
    pool.close()
    pool.join()
    return concatenate(results, axis=0)

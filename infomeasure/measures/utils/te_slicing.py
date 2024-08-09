"""Generalized data slicing method for transfer entropy estimators."""

from numpy import arange, ndarray, column_stack
from numpy.random import Generator, default_rng

from ...utils.config import logger


def te_observations(
    source,
    destination,
    src_hist_len=1,
    dest_hist_len=1,
    step_size=1,
    permute_dest=False,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    r"""
    Slice the data arrays to prepare for TE calculation.

    For TE there are four observations that are required to calculate the
    transfer entropy.

    .. math::

            \hat{T}(X_{t+1}|X^{(k)}, Y^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)}) g(\hat x_i^{(k)})}{g(x_i^{(k)}, y_i^{(l)}) g(\hat{x}_{i+1}, x_i^{(k)})}

    Parameters
    ----------
    source : array
        A numpy array of data points for the source variable.
    destination : array
        A numpy array of data points for the destination variable.
    src_hist_len : int, optional
        Number of past observations to consider for the source data.
        Default is 1, no history.
    dest_hist_len : int, optional
        Number of past observations to consider for the destination data.
        Default is 1, no history.
    step_size : int, optional
        Step size for the time delay in the embedding. Default is 1, no delay.
    permute_dest : bool | Generator, optional
        Whether to shuffle the sliced dest data. Default is False.
        This is used for the permutation TE. Rows are permuted, keeping the
        history intact.
        If a random number generator is provided, it will be used for shuffling.
        If True, a new random number generator will be created.

    Returns
    -------
    joint_space_data : array, shape (max_len, src_hist_len + dest_hist_len + 1)
        :math:`g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)})`: Joint space data.
    dest_past_embedded : array, shape (max_len, dest_hist_len)
        :math:`g(\hat x_i^{(k)})` : Destination data.
    marginal_1_space_data : array, shape (max_len, dest_hist_len + src_hist_len)
        :math:`g(x_i^{(k)}, y_i^{(l)})` : Marginal space data for source and
        destination.
    marginal_2_space_data : array, shape (max_len, dest_hist_len + 1)
        :math:`g(\hat{x}_{i+1}, x_i^{(k)})` : Marginal space data for destination.


    With ``max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size``.

    Raises
    ------
    ValueError
        If the history (``src_hist_len`` or ``dest_hist_len`` times ``step_size``) is
        greater than the length of the data.
    """
    # log warning if step_size is >1 while src_hist_len or dest_hist_len are both 1
    if step_size > 1 and src_hist_len == 1 and dest_hist_len == 1:
        logger.warning(
            "If both ``src_hist_len`` and ``dest_hist_len`` are 1, "
            "having ``step_size`` > 1 does not impact the TE calculation."
        )
    # error if vars are not positive integers
    if not all(
        isinstance(var, int) and var > 0
        for var in (src_hist_len, dest_hist_len, step_size)
    ):
        raise ValueError(
            "src_hist_len, dest_hist_len, and step_size must be positive integers."
        )

    max_delay = max(dest_hist_len, src_hist_len) * step_size
    # max delay must be less than the length of the data, otherwise raise an error
    if max_delay >= len(source):
        raise ValueError(
            "The history demanded by the source and destination data "
            "is greater than the length of the data and results in empty arrays."
        )

    base_indices = arange(max_delay, len(source))
    # Construct src_future
    src_future = source[base_indices]

    # Construct src_history
    offset_indices = arange(step_size, (src_hist_len + 1) * step_size, step_size)
    src_history_indices = base_indices[:, None] - offset_indices[::-1]
    src_history = source[src_history_indices]

    # Construct dest_history
    offset_indices = arange(step_size, (dest_hist_len + 1) * step_size, step_size)
    y_history_indices = base_indices[:, None] - offset_indices[::-1]
    if isinstance(permute_dest, Generator):
        permute_dest.shuffle(y_history_indices, axis=0)
    elif permute_dest:
        rng = default_rng()
        rng.shuffle(y_history_indices, axis=0)
    dest_history = destination[y_history_indices]

    # g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)})
    joint_space_data = column_stack(
        (
            src_future,  # \hat{x}_{i+1}
            src_history,  # x_i^{(k)}
            dest_history,  # y_i^{(l)}
        )
    )
    # g(\hat{x}_i^{(k)})
    # src_past_embedded = src_history
    # g(x_i^{(k)}, y_i^{(l)})
    marginal_1_space_data = column_stack((src_history, dest_history))
    # g(\hat{x}_{i+1}, x_i^{(k)})
    marginal_2_space_data = column_stack((src_future, src_history))

    return (
        joint_space_data,
        src_history,
        marginal_1_space_data,
        marginal_2_space_data,
    )

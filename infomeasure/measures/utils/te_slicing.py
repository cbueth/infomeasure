r"""Generalized data slicing method for transfer entropy estimators.

This module provides a method to slice the data arrays to prepare for transfer
entropy (TE) calculation.
The TE measures the information flow from a source variable (X) to
a target/destination variable (Y).
In this context, the future state is always associated with
the target/destination variable.


Conventions:
- X: Source variable
- Y: Destination/target variable
- dest_future: Future state of the destination variable (Y)
- dest_history: Past states of the destination variable (Y)
- src_history: Past states of the source variable (X)

The TE is calculated as:

.. math::

    \hat{T}(Y_{t+1}|Y^{(k)}, X^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{y}_{i+1}, y_i^{(k)}, x_i^{(l)}) g(\hat y_i^{(k)})}{g(y_i^{(k)}, x_i^{(l)}) g(\hat{y}_{i+1}, y_i^{(k)})}

"""

from numpy import arange, ndarray, concatenate, expand_dims
from numpy.random import Generator, default_rng

from ...utils.config import logger


def te_observations(
    source,
    destination,
    src_hist_len=1,
    dest_hist_len=1,
    step_size=1,
    permute_src=False,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    r"""
    Slice the data arrays to prepare for TE calculation.

    For TE there are four observations that are required to calculate the
    transfer entropy.

    .. math::

            \hat{T}(Y_{t+1}|Y^{(k)}, X^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{y}_{i+1}, y_i^{(k)}, x_i^{(l)}) g(\hat y_i^{(k)})}{g(y_i^{(k)}, x_i^{(l)}) g(\hat{y}_{i+1}, y_i^{(k)})}

    Parameters
    ----------
    source : array, shape (n,)
        A numpy array of data points for the source variable (X).
    destination : array, shape (n,)
        A numpy array of data points for the destination variable (Y).
    src_hist_len : int, optional
        Number of past observations (l) to consider for the source data (X).
        Default is 1, only one current observation, no further history.
        One future observation is always considered for the source data.
    dest_hist_len : int, optional
        Number of past observations (k) to consider for the destination data (Y).
        Default is 1, only one current observation, no further history.
    step_size : int, optional
        Step size for the time delay in the embedding.
        Default is None, which equals to 1, every observation is considered.
        If step_size is greater than 1, the history is subsampled.
        This applies to both the source and destination data.
    permute_src : bool | Generator, optional
        Whether to shuffle the sliced source history data. Default is False.
        This is used for the permutation TE. Rows are permuted, keeping the
        history intact.
        If a random number generator is provided, it will be used for shuffling.
        If True, a new random number generator will be created.

    Returns
    -------
    joint_space_data : array, shape (max_len, src_hist_len + dest_hist_len + 1)
        :math:`g(x_i^{(l)}, y_i^{(k)}, \hat{y}_{i+1})`: Joint space data.
    dest_past_embedded : array, shape (max_len, dest_hist_len)
        :math:`g(\hat y_i^{(k)})` : Embedded past destination data.
    marginal_1_space_data : array, shape (max_len, dest_hist_len + src_hist_len)
        :math:`g(x_i^{(l)}, y_i^{(k)})` : Marginal space data for destination and
        source.
    marginal_2_space_data : array, shape (max_len, dest_hist_len + 1)
        :math:`g(y_i^{(k)}, \hat{y}_{i+1})` : Marginal space data for destination.


    With ``max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size``.

    Raises
    ------
    ValueError
        If the history (``src_hist_len`` or ``dest_hist_len`` times ``step_size``) is
        greater than the length of the data.
    ValueError
        If ``src_hist_len``, ``dest_hist_len``, or ``step_size`` are
        not positive integers.
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

    base_indices = arange(max_delay, len(destination), step_size)

    # Construct src_history
    offset_indices = arange(step_size, (src_hist_len + 1) * step_size, step_size)
    src_history_indices = base_indices[:, None] - offset_indices[::-1]
    if isinstance(permute_src, Generator):
        permute_src.shuffle(src_history_indices, axis=0)
    elif permute_src:
        rng = default_rng()
        rng.shuffle(src_history_indices, axis=0)
    src_history = source[src_history_indices]

    # Construct dest_history
    offset_indices = arange(step_size, (dest_hist_len + 1) * step_size, step_size)
    dest_history_indices = base_indices[:, None] - offset_indices[::-1]
    dest_history = destination[dest_history_indices]

    # Construct dest_future
    dest_future = destination[base_indices]
    # src_future: (data_len,) -> (data_len, 1); or (data_len, m) -> (data_len, 1, m)
    dest_future = expand_dims(dest_future, axis=1)

    # g(x_i^{(l)}, y_i^{(k)}, \hat{y}_{i+1})
    joint_space_data = concatenate(
        (
            src_history,  # x_i^{(l)}
            dest_history,  # y_i^{(k)}
            dest_future,  # \hat{y}_{i+1}
        ),
        axis=1,
    )
    # g(\hat{y}_i^{(k)})
    # dest_past_embedded = dest_history
    # g(x_i^{(l)}, y_i^{(k)})
    marginal_1_space_data = concatenate((src_history, dest_history), axis=1)
    # g(y_i^{(k)}, \hat{y}_{i+1})
    marginal_2_space_data = concatenate((dest_history, dest_future), axis=1)

    return (
        joint_space_data,
        dest_history,
        marginal_1_space_data,
        marginal_2_space_data,
    )

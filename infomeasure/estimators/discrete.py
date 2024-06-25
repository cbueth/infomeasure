"""Discrete estimators for information measures.

This module provides discrete data estimators for information measures.

1. :func:`Entropy <entropy>`: Shannon entropy of discrete data.
2. :func:`Mutual Information <mutual_information>`: Compute the average mutual
   information between two variables.
3. :func:`Transfer Entropy <transfer_entropy>`: Compute Transfer Entropy from
   source to destination.
"""

from typing import List

import numpy as np


def entropy(data: List[int], base: int) -> float:
    r"""
    Calculate the entropy of discrete data.

    Using the formula:

    .. math::

        H(X) = - \sum_{i=1}^{n} p(x_i) \log_b p(x_i)

    where :math:`p(x_i)` is the probability of the :math:`i`-th
    state :cite:p:`shannonMathematicalTheoryCommunication1948`.

    Parameters
    ----------
    data : array-like of shape (n_samples,)
        The discrete observations as a list of integers.
    base : int
        The base of the logarithm for entropy calculation, also indicates the number of
        unique states.

    Returns
    -------
    float
        The calculated entropy.

    """

    # Step 1: Initialization
    # Initialize a count array to store the frequency of each unique state.
    # The length of the array is determined by the base of the discrete values.
    count = np.zeros(base, dtype=int)

    # Step 2: Counting States  # TODO: Overhaul counting
    # Iterate over the observations (states) and increment the corresponding index in
    # the count array.
    for state in data:
        count[state] += 1

    # Step 3: Probability Calculation
    # Calculate the probability of each state by dividing its count by the total number
    # of observations.
    total_observations = len(data)
    probabilities = count / total_observations

    # Step 4: Entropy Calculation
    # Compute entropy based on the calculated probabilities.
    # Note: We only consider probabilities > 0 to avoid log(0).
    entropy = -np.sum(
        probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])
    )

    return entropy


def mutual_information(var1, var2, base1, base2, time_diff=0):
    """
    Compute the average mutual information between two variables.

    Parameters
    ----------
    var1, var2 : array-like
        Arrays containing the states of the two variables.
    base1, base2 : int
        Number of states for each variable.
    time_diff : int
        Time difference between the variables. The default is 0.

    Returns
    -------
    mi : float
        Average mutual information.
    """
    observations = len(var1) - time_diff  # Adjust for time difference
    joint_count = np.zeros((base1, base2), dtype=int)
    i_count = np.zeros(base1, dtype=int)
    j_count = np.zeros(base2, dtype=int)

    # Count occurrences with time difference
    for t in range(time_diff, len(var1)):
        i = var1[t - time_diff]
        j = var2[t]
        joint_count[i][j] += 1
        i_count[i] += 1
        j_count[j] += 1

    # Compute MI
    mi = 0.0
    for i in range(base1):
        prob_i = i_count[i] / observations
        for j in range(base2):
            prob_j = j_count[j] / observations
            joint_prob = joint_count[i][j] / observations

            if joint_prob * prob_i * prob_j > 0:
                local_value = np.log2(joint_prob / (prob_i * prob_j))
                mi += joint_prob * local_value

    return mi


def count_tuples(source, dest, l, k, delay):
    """
    Count tuples for Transfer Entropy computation.

    Parameters
    ----------
    source, dest : array-like
        Source and destination time-series data.
    l, k : int
        Embedding lengths for the source and destination variables.
    delay : int
        Time delay between the source and destination variables.

    Returns
    -------
    source_next_past_count : dict
        Count for source, next state of destination, and past state of destination.
    source_past_count : dict
        Count for source and past state of destination.
    next_past_count : dict
        Count for next state and past state of destination.
    past_count : dict
        Count for past state of destination.
    observations : int
        Total number of observations.
    """
    source_next_past_count = {}
    source_past_count = {}
    next_past_count = {}
    past_count = {}
    observations = 0

    for t in range(max(k, l + delay), len(dest)):
        # Next state for the destination variable
        next_state_dest = dest[t]

        # Update past states
        past_state_dest = dest[t - k : t]
        past_state_source = source[t - delay - l + 1 : t - delay + 1]

        # Convert arrays to tuple to use as dictionary keys
        past_state_dest_t = tuple(past_state_dest)
        past_state_source_t = tuple(past_state_source)

        # Update counts
        if (
            past_state_source_t,
            next_state_dest,
            past_state_dest_t,
        ) in source_next_past_count:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] += 1
        else:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] = 1

        if (past_state_source_t, past_state_dest_t) in source_past_count:
            source_past_count[past_state_source_t, past_state_dest_t] += 1
        else:
            source_past_count[past_state_source_t, past_state_dest_t] = 1

        if (next_state_dest, past_state_dest_t) in next_past_count:
            next_past_count[next_state_dest, past_state_dest_t] += 1
        else:
            next_past_count[next_state_dest, past_state_dest_t] = 1

        if past_state_dest_t in past_count:
            past_count[past_state_dest_t] += 1
        else:
            past_count[past_state_dest_t] = 1

        observations += 1

    return (
        source_next_past_count,
        source_past_count,
        next_past_count,
        past_count,
        observations,
    )


def transfer_entropy(source, dest, l, k, delay):
    r"""
    Compute Transfer Entropy from source to destination.

    Transfer Entropy formula:

    .. math::
        \text{TE}_{s \to d} =
        \sum p(s_{t-l}, d_{t}, d_{t-k})
        \log \left( \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})} \right)

    Parameters
    ----------
    source, dest : array-like
        Source and destination time-series data.
    l, k : int
        Embedding lengths for the source and destination variables.
    delay : int
        Time delay between the source and destination variables.

    Returns
    -------
    te : float
        Transfer Entropy from source to destination.
    """
    (
        source_next_past_count,
        source_past_count,
        next_past_count,
        past_count,
        observations,
    ) = count_tuples(source, dest, l, k, delay)

    te = 0
    for (s_t, d_t, d_t_k), p_s_t_d_t_d_t_k in source_next_past_count.items():
        p_s_t_d_t_d_t_k /= observations

        p_s_t_d_t_k = source_past_count[s_t, d_t_k] / observations
        p_d_t_d_t_k = next_past_count[d_t, d_t_k] / past_count[d_t_k]
        p_d_t_k = past_count[d_t_k] / observations

        log_term = (p_d_t_d_t_k / p_d_t_k) / (p_s_t_d_t_k / p_d_t_k)
        local_value = np.log(log_term)

        te += p_s_t_d_t_d_t_k * local_value

    # Convert to base 2 logarithm
    te /= np.log(2)

    return te

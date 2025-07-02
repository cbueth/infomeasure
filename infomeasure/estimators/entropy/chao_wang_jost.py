"""Module for the Chao Wang Jost entropy estimator."""

from numpy import log
from numpy import sum as np_sum
from scipy.special import digamma

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.config import logger
from ...utils.exceptions import TheoreticalInconsistencyError


class ChaoWangJostEntropyEstimator(DiscreteHEstimator):
    r"""Advanced bias-corrected Shannon entropy estimator using coverage estimation.

    The Chao-Wang-Jost estimator provides improved entropy estimates for incomplete sampling
    scenarios by accounting for unobserved species through sophisticated statistical corrections.
    This estimator is particularly valuable when dealing with ecological data, text analysis,
    or any discrete distribution where the sample may not capture all possible outcomes.

    The Chao-Wang-Jost estimator addresses the systematic underestimation of entropy in
    finite samples by applying sophisticated statistical corrections.
    Through coverage estimation using singleton and doubleton counts,
    it provides reliable entropy estimates even with small or incomplete samples.
    Based on species accumulation theory and Good-Turing estimation principles,
    this approach is particularly valuable when the sample doesn't capture all
    possible outcomes, such as in ecological diversity studies with incomplete species
    sampling or text analysis where vocabulary may be incompletely observed.
    The estimator is especially useful when standard entropy estimators show systematic
    bias due to sample size limitations.

    Standard entropy estimators often underestimate diversity in finite samples,
    especially when the sampling is incomplete. This estimator overcomes this limitation
    by leveraging information from rare species (singletons and doubletons) to estimate
    sample coverage and correct for unobserved species. The theoretical foundation in
    species accumulation curves and Good-Turing frequency estimation provide a robust
    statistical framework for addressing sampling bias issues.

    **Mathematical Foundation:**

    The estimator combines observed entropy with a correction term based on coverage estimation:

    .. math::

        \hat{H}_{\text{CWJ}} = \sum_{1 \leq n_i \leq N-1} \frac{n_i}{N} \left(\sum_{k=n_i}^{N-1} \frac{1}{k} \right) +
        \frac{f_1}{N} (1 - A)^{-N + 1} \left\{ - \log(A) - \sum_{r=1}^{N-1} \frac{1}{r} (1 - A)^r \right\}

    where the coverage parameter :math:`A` is estimated as:

    .. math::

        A = \begin{cases}
        \frac{2 f_2}{(N-1) f_1 + 2 f_2} \, & \text{if} \, f_2 > 0 \\
        \frac{2}{(N-1)(f_1 - 1) + 2} \, & \text{if} \, f_2 = 0, \; f_1 \neq 0 \\
        1, & \text{if} \, f_1 = f_2 = 0
        \end{cases}

    Here, :math:`f_1` represents the number of singletons (species observed exactly once) and
    :math:`f_2` the number of doubletons (species observed exactly twice) in the sample
    :cite:p:`chaoEntropySpeciesAccumulation2013`.

    Notes
    -----
    - The algorithm is adapted from the `entropart <https://ericmarcon.github.io/entropart/index.html>`_ R library :cite:p:`marconEntropartPackageMeasure2015`
    - The correction becomes negligible when samples are complete (:math:`f_1 = f_2 = 0`)

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.

    Examples
    --------
    >>> import infomeasure as im
    >>>
    >>> # Basic usage with incomplete sampling scenario
    >>> data = [1, 1, 2, 3, 4, 5]  # Many singletons suggest incomplete sampling
    >>> h_cwj = im.entropy(data, approach="chao_wang_jost", base=2)
    >>> h_standard = im.entropy(data, approach="discrete", base=2)
    >>> print(f"Chao-Wang-Jost: {h_cwj:.3f} bits")
    Chao-Wang-Jost: 3.635 bits
    >>> print(f"Standard: {h_standard:.3f} bits")
    Standard: 2.252 bits
    >>>
    >>> # Ecological diversity example
    >>> species_counts = [1, 1, 1, 2, 2, 3, 5, 8]  # Species abundance data
    >>> diversity = im.entropy(species_counts, approach="cwj", base="e")
    >>> print(f"Species diversity: {diversity:.3f} nats")
    Species diversity: 2.054 nats

    See Also
    --------
    infomeasure.estimators.functional.entropy : Functional interface for entropy calculation
    infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator : Standard maximum likelihood entropy estimator
    """

    def _simple_entropy(self):
        """Calculate the Chao Wang Jost entropy of the data.

        Returns
        -------
        float
            The calculated entropy value.
        """
        N = self.data[0].N
        counts = self.data[0].counts

        # Calculate singletons (f1) and doubletons (f2)
        f1 = np_sum(counts == 1)
        f2 = np_sum(counts == 2)

        #
        if f1 == 0 and f2 == 0:
            logger.warning(
                "There are no singletons and doubletons in the data, "
                "the corrections becomes negible."
            )

        # Calculate parameter A
        if f2 > 0:
            A = 2 * f2 / ((N - 1) * f1 + 2 * f2)
        elif f1 > 0:
            A = 2 / ((N - 1) * (f1 - 1) + 2)
        else:
            A = 1

        # First part of the formula: sum over observed counts
        # Using digamma(N) - digamma(n_i) = sum_{k=n_i}^{N-1} 1/k
        cwj = (
            counts[1 <= counts] * (digamma(N) - digamma(counts[1 <= counts]))
        ).sum() / N
        # Second part: correction term when A != 1
        if A != 1 and f1 > 0:
            # Calculate sum_{r=1}^{N-1} (1/r) * (1-A)^r
            p2 = sum(1 / r * (1 - A) ** r for r in range(1, N))
            correction = f1 / N * (1 - A) ** (1 - N) * (-log(A) - p2)
            cwj += correction

        # Convert to the desired base
        if self.base != "e":
            cwj /= log(self.base)

        return cwj

    def _extract_local_values(self):
        """Calculate local entropy values for each data point.

        Raises
        ------
        TheoreticalInconsistencyError
            Local values are not theoretically well-defined for Chao Wang Jost estimator
            due to the complex bias correction involving global sample statistics.
        """
        raise TheoreticalInconsistencyError(
            "Local values are not implemented for Chao Wang Jost estimator. "
            "The Chao Wang Jost correction involves global sample statistics (singletons, doubletons) "
            "and complex bias corrections that cannot be meaningfully decomposed into local contributions. "
            "The correction term depends on the entire sample structure and cannot be attributed to "
            "individual observations in a theoretically consistent manner."
        )

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for Chao Wang Jost estimator
            due to fundamental issues with mixing bias corrections from different distributions.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Chao Wang Jost estimator. "
            "The Chao Wang Jost correction creates theoretical inconsistencies when applied to cross-entropy: "
            "(1) The bias correction depends on sample-specific statistics (singletons, doubletons) "
            "that are tied to individual distributions; "
            "(2) Mixing corrections from different distributions lacks theoretical foundation; "
            "(3) The complex correction terms involving coverage estimation cannot be meaningfully "
            "applied across different probability distributions."
        )

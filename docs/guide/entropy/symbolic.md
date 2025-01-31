---
file_format: mystnb
kernelspec:
  name: python3
---

(symbolic_entropy)=
# Symbolic / Permutation Entropy Estimation
The Shannon {cite:p}`shannonMathematicalTheoryCommunication1948` differential entropy formula is given as:

$$
H(X) = -\int_{X} p(x) \log_b p(x) \, dx,
$$

where $x$ denotes the realizations of the random variable $X$ with probability $p(x)$, and $b$ is the base of the logarithm. Further details can be read in the section {ref}`Entropy / Uncertainty`.  

For a given time series dataset $\{x_t\}_{t=1, \ldots, T}$, where $T$ is the number of time points, each value can be replaced by a symbolic sequence $s(t)$ of length $s$, defined by the order parameter $s$. Using a sliding window of size $s$, subsets of the data are generated sequentially, and each subset is symbolized using a specific symbolization technique.  

The symbol assigned to each subset carries attributes of the data, as determined by the symbolization method. In this package, the **ordinal pattern approach** is used for symbolization. For details, refer to {cite:p}`PermutationEntropy2002`. 

```{Note}
**Example of Ordinal Pattern symbolization:**

For order $( s = 2 )$, each subsequence $( \{x(t), x(t+1)\} )$ of the time series will be analyzed as:

- If $( x(t) < x(t+1) )$, the pattern type is $(0, 1)$.
- Conversely, if $( x(t) > x(t+1) )$, the pattern type is $(1, 0)$.
```

Once the time series is mapped into the symbolic space, the probability distribution is estimated by computing the relative frequencies of symbols. Specifically, all $n!$ permutations $\pi$ of order $n$ are considered as possible order types of $n$ consecutive data points. For each permutation $\pi$, the relative frequency is:

$$
p(\pi) = \frac{\#\{t \mid t \leq T - n, \{x_t, \ldots, x_{t+n}\} \text{ has type } \pi\}}{T - n + 1}.
$$

The **permutation entropy** of order $n$ is then defined as:

$$
H(n) = -\sum p(\pi) \log p(\pi),
$$

where the sum runs over all $n!$ permutations $\pi$ of order $n$. This measures the information contained in comparing $n$ consecutive values of the time series. 

```{note}
- **Example**:
  - Time series: $[4, 7, 9, 10, 6, 11, 3]$
  - Ordinal Symbols of order 2: $(0, 1), (0, 1), (0, 1), (1, 0), (0, 1), (1, 0)$
  - Probabilities of symbols: $ p((0, 1)) = \frac{4}{6} = \frac{2}{3}, p((1, 0)) = \frac{2}{6} = \frac{1}{3} $
  - Symbolic Entropy: $ H(2) = -\left(\frac{2}{3} \log_2 \frac{2}{3} + \frac{1}{3} \log_2 \frac{1}{3}\right) $
  $\approx -\left(0.6667 \times -0.5849 + 0.3333 \times -1.5849\right) \approx 0.918 \text{ bits} $
 ```

Finally, the **permutation entropy per symbol** can be computed as an optional step by normalizing $H(n)$ with $(n - 1)$, since comparisons begin with the second value:

$$
h_n = \frac{H(n)}{n - 1}.
$$

This step provides a more granular understanding of the entropy distribution within the time series data.  
> Note:
> The package allows user to obtain both the local and global (average) values to the Entropy computation.
> Further one can also compute the entropy per symbol as optional choice.


The estimator is implemented in the {py:class}`SymbolicEntropyEstimator <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

## Implementation

```{eval-rst} 
.. autoclass:: infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

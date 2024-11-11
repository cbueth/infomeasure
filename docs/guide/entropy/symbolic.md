---
file_format: mystnb
kernelspec:
  name: python3
---

(symbolic_entropy)=
# Symbolic / Permutation Entropy Estimation

For a provided (let's say) time series data set $ \{x_t\}_{t=1, \ldots, T}$ with finite values, one can replace each of those values with a symbol sequence $s(t)$ of a length defined the order parameter $s$.
Imagine a dynamic window of size $s$ (order) which runs across all the datapoints generating the subset for each datapoint.
This subset is further symbolize by certain specific symbolization technique.
It is important to remember, that the symbol assigned to every datapoints will carry certain attributes of the data in according to the symbolization technique.
There are several ways to symbolize the data, in this package we will take the approach of ordinal pattern, for detail refer to artile {cite:p}`PermutationEntropy2002`.
```{Note}
**Example of Ordinal Pattern symbolization:**

For order $( s = 2 )$, each subsequence $( \{x(t), x(t+1)\} )$ of the time series will be analyzed as:

- If $( x(t) < x(t+1) )$, the pattern type is $(0, 1)$.
- Conversely, if $( x(t) > x(t+1) )$, the pattern type is $(1, 0)$.
```


After mapping the time series data into symbolic space, it is straightforward to estimate the probability distribution computing the relative frequency yof symbols,
i.e we will study all  $n!$  permutations  $\pi$ of order  $n$ which are considered here as possible order types of  $n$  different numbers.
Hence, for each $\pi$ we determine the relative frequency (\# means number) as follows:

$$
p(\pi) = \frac{\#\{t | t \leq T - n, \{x_t, \ldots, x_{t+n}\} \text{ has type } \pi\}}{T - n + 1}
$$

The permutation entropy of order $ n $  is defined as:

$$
H(n) = -\sum p(\pi) \log p(\pi),
$$
where the sum runs over all $( n! )$ permutations $( \pi $ of order $( n )$. This is the information contained in comparing $( n )$ consecutive values of the time series.

Finally, the permutation entropy per symbol is computed as an optional step,
dividing by $( n - 1 )$ since comparisons start with the second value:

$$
h_n = \frac{H(n)}{n - 1}.
$$

This additional step provides a more granular understanding of the entropy distribution within the time series data.

```{note}
- **Example**:
  - Time series: $[4, 7, 9, 10, 6, 11, 3]$
  - Ordinal Symbols of order 2: $(0, 1), (0, 1), (0, 1), (1, 0), (0, 1), (1, 0)$
  - Probabilities of symbols: $ p((0, 1)) = \frac{4}{6} = \frac{2}{3}, p((1, 0)) = \frac{2}{6} = \frac{1}{3} $
  - Symbolic Entropy: $ H(2) = -\left(\frac{2}{3} \log_2 \frac{2}{3} + \frac{1}{3} \log_2 \frac{1}{3}\right) $
  $\approx -\left(0.6667 \times -0.5849 + 0.3333 \times -1.5849\right) \approx 0.918 \text{ bits} $
 ```

The estimator is implemented in the {py:class}`SymbolicEntropyEstimator <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

---
file_format: mystnb
kernelspec:
  name: python3
---

(ordinal_entropy)=
# Ordinal / Symbolic / Permutation Entropy Estimation
The Shannon discrete {ref}`entropy <entropy_overview>` formula is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\sum_{x \in X} p(x) \log p(x),
$$

where $p(x)$ is the probability mass function (pmf).

Ordinal entropy is builds on Shannon entropy, but accepts continuous variables.
Instead of focusing on single realisations of a variable, ordinal entropy focuses on the order or sequence of values in a dataset.
From the time series $\{x_t\}_{t=1, \ldots, T}$, a sliding window of size $s$, called embedding dimension or order, is used to generate subsets of the data.
These ordinal sequences are the permutation patterns of the subsets,
describing the relative change.
Like this, the entropy of repeating patterns can be calculated.
The symbol assigned to each subset carries attributes of the data, as determined by the symbolization method {cite:p}`PermutationEntropy2002`.

```{admonition} Example for $s = 2$

For order $( s = 2 )$, each subsequence $( \{x(t), x(t+1)\} )$ of the time series will be analysed as:

- If $( x(t) < x(t+1) )$, the pattern type is $(0, 1)$.
- Conversely, if $( x(t) > x(t+1) )$, the pattern type is $(1, 0)$.
```

Once the time series is mapped into the ordinal space, the probability mass function (pmf) is estimated by computing the relative frequencies of symbols. Specifically, all $n!$ permutations $\pi$ of order $n$ are considered as possible order types of $n$ consecutive data points. For each permutation $\pi$, the relative frequency is:

$$
p(\pi) = \frac{\#\{t \mid t \leq T - n, \{x_t, \ldots, x_{t+n}\} \text{ has type } \pi\}}{T - n + 1}.
$$

The **permutation entropy** of order $n$ is then defined as:

$$
H(n) = -\sum p(\pi) \log p(\pi),
$$

where the sum runs over all $n!$ permutations $\pi$ of order $n$. This measures the information contained in comparing $n$ consecutive values of the time series.

```{admonition} Example for $s=3$
- Time series: $[4, 7, 9, 10, 6, 11, 3]$
- Ordinal Symbols of order 3: $(0, 1, 2), (0, 1, 2), (2, 0, 1), (1, 0, 2), (2, 0, 1)$
- Probabilities: $ p((0, 1, 2)) = \frac{2}{5}, p((2, 0, 1)) = \frac{2}{5}, p((1, 0, 2)) = \frac{1}{5} $
- Ordinal Entropy: $ H(3) = -\left(\frac{2}{5} \log_2 \frac{2}{5} + \frac{2}{5} \log_2 \frac{2}{5} + \frac{1}{5} \log_2 \frac{1}{5}\right) \approx 1.52\,\text{bit}$
```

```{code-cell}
import infomeasure as im
im.entropy([4, 7, 9, 10, 6, 11, 3], approach='ordinal', embedding_dim=3, base=2)
```

```{note}
The package allows user to obtain both the local and global (average) values to the Entropy computation.
The ordinal entropy is bounded between 0 and $\log(n!)$.
```


For demonstration, we generate a dataset of normally distributed values with mean $0$ and standard deviation $1$.
The analytical equation of the other approaches does not hold; as for ordinal entropy, the pmf of the ordinal patterns is analysed.

```{code-cell}
import numpy as np
rng = np.random.default_rng(692475)

std = 1.0
data = rng.normal(loc=0, scale=std, size=2000)

h = im.entropy(data, approach="ordinal", embedding_dim=3)
h_expected = (1 / 2) * np.log(2 * np.pi * np.e * std ** 2)
h, h_expected
```

To access the local values, an estimator instance is needed.

```{code-cell}
est = im.estimator(data, measure="h", approach="ordinal", embedding_dim=3)
est.result(), est.local_vals()
```

For this estimator, access to the distribution dictionary is also available.
```{code-cell}
est = im.estimator(data, measure="h", approach="ordinal", embedding_dim=3)
print(f"Entropy: {est.result():.4f} bits")
print(f"Distribution: {est.dist_dict}")
print(f"Probabilities sum to: {sum(est.dist_dict.values()):.1f}")
```

The estimator is implemented in the {py:class}`OrdinalEntropyEstimator <infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

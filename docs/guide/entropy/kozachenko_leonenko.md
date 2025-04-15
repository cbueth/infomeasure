---
file_format: mystnb
kernelspec:
  name: python3
---

(entropy_kozachenko_leonenko)=
# Kozachenko-Leonenko (KL) / Metric / kNN Entropy Estimation
The Shannon differential {ref}`entropy <entropy_overview>` formula for a continuous random variable $ X $ with density $ p(x)$ is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\int_{X} p(x) \log p(x) \, dx,
$$

where $p(x)$ is the probability density function (pdf).



The Kozachenko-Leonenko (KL) entropy estimator leverages a **nearest-neighbor approach** to estimate the Shannon entropy of a continuous random variable from a finite sample. The estimator approximates entropy as the expectation of the logarithm of the density. Given a sample $ \{x_1, x_2, \dots, x_N\} $, the density at each point $ x_i $ is estimated using the distance $ \epsilon(i) $ to its $ k $-th nearest neighbor. Assuming **local uniformity**, the estimated density suffices $ \widehat{p}(x_i) \approx c_d \epsilon(i)^d $, where $ c_d $ is the volume of a unit $ d $-dimensional ball. By leveraging **order statistics**, the expectation $ E(\log p_i) = \psi(k) - \psi(N) $ is obtained, where $ \psi(x) $ is the **digamma function**. Substituting this into the entropy definition leads to the final KL estimator {cite:p}`kozachenko1987sample,RevieEstimators,miKSG2004`:

$$
\hat{H}(X) = - \psi(k) + \psi(N) + \log c_d + \frac{d}{N} \sum_{i=1}^{N} \log \epsilon(i),
$$

where:
- $\psi$ is the _digamma function_, the derivative of the logarithm of the gamma function $\Gamma(x)$,
- $k$ is the number of nearest neighbors,
- $\epsilon_i$ is twice the distance from $x_i$ to its $k^{th}$ nearest neighbor, representing the diameter of the hypersphere encompassing the $k$ neighbors,
- $c_d$ is the volume of the unit ball in $d$-dimensional space, where $\log c_d = 0$ for the maximum norm and $c_d = \pi^{d/2} / (\Gamma(1 + d/2) \cdot 2^d)$ for Euclidean spaces.

For demonstration, we generate a dataset of normally distributed values with mean $0$ and standard deviation $1$.
We then calculate the entropy using the box kernel with a bandwidth of $0.5$.
The analytical expected values can be calculated with

$$
H(X) = \frac{1}{2} \log(2\pi e \sigma^2),
$$

where $\sigma^2$ is the variance of the data.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(692475)

std = 1.0
data = rng.normal(loc=0, scale=std, size=2000)

h = im.entropy(data, approach="metric", k=4)
h_expected = (1 / 2) * np.log(2 * np.pi * np.e * std ** 2)
h, h_expected
```

To access the local values, an estimator instance is needed.

```{code-cell}
est = im.estimator(data, measure="h", approach="metric", k=4)
est.result(), est.local_vals()
```

Higher-dimensional data also is easily processed.

```{code-cell}
im.entropy(rng.normal(loc=0, scale=1, size=(2000, 3)), approach="metric", k=4)
```

The estimator is implemented in the {py:class}`KozachenkoLeonenkoEntropyEstimator <infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

---
file_format: mystnb
kernelspec:
  name: python3
---

(tsallis_entropy)=
# Tsallis Entropy Estimation
The {ref}`Tsallis entropy <tsallis-q-entropy>` generalizes the {ref}`Shannon entropy` by modifying the additivity law {cite:p}`articleTsallis,Tsallis1999,Tsallis1998`.

```{admonition} Tsallis Entropy
:class: tip

The Tsallis entropy, also known as Havrda-Charvát entropy, of a random vector $(X \in \mathbb{R}^m)$ with probability density $f$, for $N$ independent and identically distributed data points, is given by:

$$
H_q = \frac{1}{q - 1} \left(1 - \int_{\mathbb{R}^m} f^q(x) \, dx \right), \quad q \neq 1.
$$

When $q \to 1$, Tsallis entropy converges to Shannon entropy:

$$
H_1 = - \int_{\mathbb{R}^m} f(x) \log f(x) \, dx.
$$
```

{cite:t}`leonenkoEstimationEntropiesDivergences2006` introduced a class of estimators for Tsallis entropy in the same article as the Rényi entropy estimator by extending the {ref}`K-L entropy estimation<entropy_kozachenko_leonenko>` technique, which is based on the $K^{th}$-Nearest Neighbour (KNN) search approach {cite:p}`leonenkoClassRenyiInformation2008,leonenkoEstimationEntropiesDivergences2006`.

Let us suppose $X$ has $N$ data points.
First, for each point $X_i$, compute the distances $\rho(X_i, X_j)$ to all other points $X_j$ (where $j \neq i$) and record $\rho_{k,N-1}^{(i)}$ as the distance from $X_i$ to its $K^{th}$-Nearest Neighbour.

The Tsallis entropy $\hat{H}_{N,k,q}$ is estimated as follows:

---
**For $q \neq 1$:**

$$
\hat{H}_{N,k,q} = \frac{1 - \hat{I}_{N,k,q}}{q - 1},
$$

where:

$$
\hat{I}_{N,k,q} = \frac{1}{N} \sum_{i=1}^N \left(\zeta_{N,i,k}\right)^{1-q},
$$

$$
\zeta_{N,i,k} = (N-1) \, C_k \, V_m \, \left(\rho_{k,N-1}^{(i)}\right)^m,
$$

- $V_m = \frac{\pi^{m/2}}{\Gamma(m/2 + 1)}$ is the volume of the unit ball in $\mathbb{R}^m$,
- $C_k = \left[ \frac{\Gamma(k)}{\Gamma(k+1-q)} \right]^{1/(1-q)}$,
- $\rho_{k,N-1}^{(i)}$ is the distance from the point $X_i$ to its $k^{th}$ nearest neighbor.

---

**For $q = 1$:**

$$
\hat{H}_{N,k,1} = \frac{1}{N} \sum_{i=1}^N \log \xi_{N,i,k},
$$

where:

$$
\xi_{N,i,k} = (N-1) \exp[-\Psi(k)] \, V_m \, \left(\rho_{k,N-1}^{(i)}\right)^m,
$$

- $\Psi(z) = \frac{\Gamma'(z)}{\Gamma(z)}$ is the digamma function.
- For $k \geq 1$:

$$
\Psi(k) = -\gamma + A_{k-1},
$$

where:
- $\gamma$ is the Euler-Mascheroni constant,
- $A_j = \sum_{j=1}^j \frac{1}{j}$ is the sum of the harmonic series up to $j$.
---



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

h = im.entropy(data, approach="tsallis", q=1)
h_expected = (1 / 2) * np.log(2 * np.pi * np.e * std ** 2)
h, h_expected
```

When $q \neq 1$, we can reweight the data points.

```{code-cell}
(im.entropy(data, approach="tsallis", q=0.8),
 im.entropy(data, approach="tsallis", q=1.2))
```

For a 2D point cloud, it is as easy to calculate the entropy:

```{code-cell}
im.entropy(
    data=rng.normal(loc=0, scale=1, size=(2000, 2)),
    approach="tsallis", q=2
)
```

Local values are not supported.

The estimator is implemented in the {py:class}`TsallisEntropyEstimator <infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

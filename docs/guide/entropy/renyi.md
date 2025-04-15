---
file_format: mystnb
kernelspec:
  name: python3
---

(renyi_entropy)=
# Rényi Entropy Estimation
The {ref}`Rényi entropy <renyi-alpha-entropy>` is a generalization of the {ref}`Shannon entropy`, incorporating a parameter $\alpha$ that controls the sensitivity of the entropy to the probabilities of different states of the random variable. For $\alpha < 1$, smaller probabilities $(p_i)$ are emphasized, highlighting rare events. For $\alpha > 1$, higher probabilities are emphasized, focusing on common events. When $\alpha = 1$, Rényi entropy reduces to Shannon entropy {cite:p}`renyi1976selected,jizbaInformationTheoryGeneralized2004`.

```{admonition} Rényi Entropy
:class: tip

The Rényi entropy of order $\alpha$ for a probability distribution with density function $f$ is given by:

$$
H^*_{\alpha} = \frac{1}{1-\alpha} \log \int_{\mathbb{R}^m} f^{\alpha}(x) \, dx, \quad \alpha \neq 1,
$$

As $\alpha \to 1$, Rényi entropy converges to Shannon entropy:

$$
H_1 = - \int_{\mathbb{R}^m} f(x) \log f(x) \, dx.
$$
```

{cite:t}`leonenkoEstimationEntropiesDivergences2006` introduce a class of estimators for Rényi entropy by extending the {ref}`K-L entropy estimation<entropy_kozachenko_leonenko>` technique, which is based on the $k^{th}$-Nearest Neighbour (kNN) search approach {cite:p}`leonenkoClassRenyiInformation2008,leonenkoEstimationEntropiesDivergences2006`.

Let us suppose $X$ has $N$ data points.
First, for each point $X_i$, compute the distances $\rho(X_i, X_j)$ to all other points $X_j$ (where $j \neq i$) and record $\rho_{k,N-1}^{(i)}$, the distance from $X_i$ to its $K^{th}$-Nearest Neighbour.

Rényi entropy $\hat{H}_{N,k,q}^*$ is estimated as follows:

---
**For $\alpha \neq 1$:**

$$
\hat{H}_{N,k,\alpha}^* = \frac{\log \hat{I}_{N,k,\alpha}}{1 - \alpha},
$$

where:

$$
\hat{I}_{N,k,\alpha} = \frac{1}{N} \sum_{i=1}^N \left(\zeta_{N,i,k}\right)^{1-\alpha},
$$

$$
\zeta_{N,i,k} = (N-1) \, C_k \, V_m \, \left(\rho_{k,N-1}^{(i)}\right)^m,
$$

- $V_m = \frac{\pi^{m/2}}{\Gamma(m/2 + 1)}$ is the volume of the unit ball in $\mathbb{R}^m$,
- $C_k = \left[\frac{\Gamma(k)}{\Gamma(k+1-\alpha)}\right]^{1/(1-\alpha)}$,
- $\rho_{k,N-1}^{(i)}$ is the distance from the point $X_i$ to its $k^{th}$ nearest neighbor.
---

**For $\alpha = 1$:**

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

h = im.entropy(data, approach="renyi", alpha=1)
h_expected = (1 / 2) * np.log(2 * np.pi * np.e * std ** 2)
h, h_expected
```

When $\alpha \neq 1$, we can reweight the data points.

```{code-cell}
(im.entropy(data, approach="renyi", alpha=0.8),
 im.entropy(data, approach="renyi", alpha=1.2))
```

For a 2D point cloud, it is as easy to calculate the entropy:

```{code-cell}
im.entropy(
    data=rng.normal(loc=0, scale=1, size=(2000, 2)),
    approach="renyi", alpha=2
)
```

Local values are not supported.


The estimator is implemented in the {py:class}`RenyiEntropyEstimator <infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

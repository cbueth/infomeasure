---
file_format: mystnb
kernelspec:
  name: python3
---

(renyi_entropy)=
# Rényi Entropy Estimation

The [**Rényi entropy**](index.md#renyi-alpha-entropy) is a generalization of the [Shannon entropy](index.md#shannon-entropy) which includes a parameter $\alpha$ that controls the sensitivity of the entropy to the probabilities of the different states of the random variable.
For $\alpha < 1$, the small values of probabilities $p_i$ are emphasized and for $\alpha > 1$, on the contrary, higher probabilities are emphasized.
For $\alpha = 1$ Rényi entropy reduces to Shannon Entropy.
```{admonition} Rényi Entropy
:class: tip

The Rényi entropy of order $\alpha$ for a probability distribution with density function \(f\) is given by:

$$
H^*_{\alpha} = \frac{1}{1-\alpha} \log \int_{\mathbb{R}^m} f^{\alpha}(x) \, dx, \quad \alpha \neq 1,
$$
When $ \alpha \rightarrow 1$, Rényi entropy converges to the Shannon entropy:

$$
H_1 = - \int_{\mathbb{R}^m} f(x) \log f(x) \, dx
$$
```
## Estimation Technique:
**Leonenko et. al.** introduced a class of estimators for Rényi entropy by extending the [K-L entropy estimation](kozachenko_leonenko.md) technique which is based on the $K^{th}$-Nearest Neighbour (KNN) search approach, for detail refer to {cite:p}`RenyiTsallisEstimator2008` {cite:p}`LeonenkoRenyiEstimator`.
Let us suppose the $X$ with $N$ datapoints.
First for each points $ X_i $, compute the distances $ \rho(X_i, X_j) $ to all other points $ X_j $ (where $ j \neq i $) and record the $ \rho_{k,N-1}^{(i)} $ as the distance from $ X_i $ to its $K^{th}$-Nearest Neighbour.
Renyi entropy $\hat{H}_{N,k,q}^*$ is estimated by:

**For $\alpha \neq 1$:**

$$
\hat{H}_{N,k,q}^* = \frac{\log \hat{I}_{N,k,q}}{1 - q}
$$
Where:

$$
\hat{I}_{N,k,q} = \frac{1}{N} \sum_{i=1}^N \left(\zeta_{N,i,k}\right)^{1-q}
$$

$$
\zeta_{N,i,k} = (N-1) \, C_k \, V_m \, \left(\rho_{k,N-1}^{(i)}\right)^m
$$

- $V_m = \frac{\pi^{m/2}}{\Gamma(m/2 + 1)}$ is the volume of the unit ball in $\mathbb{R}^m$.
- $C_k = \left[\frac{\Gamma(k)}{\Gamma(k+1-q)}\right]^{1/(1-q)}$
- $\rho_{k,N-1}^{(i)}$ is the distance from the point $X_i$ to its $k^{th}$ nearest neighbor.

**For $\alpha = 1$:**

$$
\hat{H}_{N,k,1} = \frac{1}{N} \sum_{i=1}^N \log \xi_{N,i,k}
$$
Where:

$$
\xi_{N,i,k} = (N-1) \exp[-\Psi(k)] \, V_m \, \left(\rho_{k,N-1}^{(i)}\right)^m
$$

- $\Psi(z) = \frac{\Gamma'(z)}{\Gamma(z)}$ is the digamma function.
- For $k \geq 1$:

$$
\Psi(k) = -\gamma + A_{k-1}
$$
Where:

- $\gamma$ is the Euler-Mascheroni constant.
- $A_j = \sum_{j=1}^j \frac{1}{j}$ is the sum of the harmonic series up to $j$.


```{admonition} Alfréd Rényi (1921–1970)
:class: tip
Alfréd Rényi was a Hungarian mathematician who made significant contributions to probability theory, information theory, and combinatorics. He is known for the Rényi entropy, Rényi divergence, and Rényi's parking constants.

    A mathematician is a device for turning coffee into theorems
    -- Alfréd Rényi

{cite:p}`suzuki2002history`
```
## Test: Renyi Entropy Estimation

```{code-cell}
import infomeasure as im
im.entropy([1, 2, 3, 4, 5], approach="renyi", alpha=2)
```


The estimator is implemented in the {py:class}`RenyiEntropyEstimator <infomeasure.measures.entropy.renyi.RenyiEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.renyi.RenyiEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

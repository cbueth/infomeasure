---
file_format: mystnb
kernelspec:
  name: python3
---

(tsallis_entropy)=
# Tsallis Entropy Estimation

The [**Tsallis entropy**](index.md#renyi-alpha-entropy) is a generalization of the [Shannon entropy](index.md#shannon-entropy) by modifying the additivity law, for detail read {cite:p}`articleTsallis,Tsallis1999,Tsallis1998`.

```{admonition} Tsallis Entropy
:class: tip

The Tsallis entropy or Havrda and Charvát entropy of a random vector $(X \in \mathbb{R}^m)$ with probability density $(f)$, from $N$ independent and identically distributedis given by:

$$
H_q = \frac{1}{q - 1} \left(1 - \int_{\mathbb{R}^m} f^q(x) \, dx \right), \quad q \neq 1,
$$
When $ q \rightarrow 1$, Tsallis entropy tends to the Shannon entropy:

$$
H_1 = - \int_{\mathbb{R}^m} f(x) \log f(x) \, dx.
$$
```
## Estimation Technique:
**Leonenko et. al.** in the same article to that of Rényi entropy estimator, introduced a class of estimators for Tallis entropy by extending the [K-L entropy estimation](kozachenko_leonenko.md) technique which is based on the $K^{th}$-Nearest Neighbour (KNN) search approach, for detail refer to {cite:p}`RenyiTsallisEstimator2008` {cite:p}`LeonenkoRenyiEstimator`.
Let us suppose the $X$ with $N$ datapoints.
First for each points $ X_i $, compute the distances $ \rho(X_i, X_j) $ to all other points $ X_j $ (where $ j \neq i $) and record the $ \rho_{k,N-1}^{(i)} $ as the distance from $ X_i $ to its $K^{th}$-Nearest Neighbour.
Tallis entropy $\hat{H}_{N,k,q}$ is estimated by:

**For $q \neq 1$:**

$$
\hat{H}_{N,k,q} = \frac{1 - \hat{I}_{N,k,q}}{q - 1}
$$
Where:

$$
\hat{I}_{N,k,q} = \frac{1}{N} \sum_{i=1}^N (\zeta_{N,i,k})^{1-q}
$$

$$
\zeta_{N,i,k} = (N-1) C_k V_m (\rho_{k,N-1}^{(i)})^m
$$

- $ V_m = \frac{\pi^{m/2}}{\Gamma(m/2 + 1)} $ is the volume of the unit ball in $ \mathbb{R}^m $,
- $ C_k = \left[ \frac{\Gamma(k)}{\Gamma(k+1-q)} \right]^{1/(1-q)} $,
- $ \rho_{k,N-1}^{(i)} $ is the distance from $ X_i $ to its k-th nearest neighbor.

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

## Test: Tsallis Entropy Estimator

```{code-cell}
import infomeasure as im
im.entropy([1, 2, 3, 4, 5], approach="tsallis", q=2)  # Havrda-Charvát entropy
```



The estimator is implemented in the {py:class}`TsallisEntropyEstimator <infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

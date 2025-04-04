---
file_format: mystnb
kernelspec:
  name: python3
---

(entropy_kozachenko_leonenko)=
# Kozachenko-Leonenko (KL) / Metric / kNN Entropy Estimation
The Shannon differential {ref}`entropy_overview` formula for a continuous random variable $ X $ with density $ p(x)$ is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\int_{X} p(x) \log p(x) \, dx,
$$

where, $p(x)$ is the probability density function(_pdf_).



The ``Kozachenko-Leonenko (KL) entropy estimator`` leverages a **_nearest-neighbor approach_** to estimate the Shannon entropy of a continuous random variable from a finite sample. The estimator approximates entropy as the expectation of the logarithm of the density. Given a sample $ \{x_1, x_2, \dots, x_N\} $, the density at each point $ x_i $ is estimated using the distance $ \epsilon(i) $ to its $ k $-th nearest neighbor. Assuming **_local uniformity_**, the estimated density follows $ \widehat{p}(x_i) \approx c_d \epsilon(i)^d $, where $ c_d $ is the volume of a unit $ d $-dimensional ball. By leveraging **_order statistics_**, the expectation $ E(\log p_i) = \psi(k) - \psi(N) $ is obtained, where $ \psi(x) $ is the **_digamma function_**. Substituting this into the entropy definition leads to the final KL estimator {cite:p}`kozachenko1987sample`, {cite:p}`RevieEstimators`, {cite:p}`miKSG2004`:

$$
\hat{H}(X) = - \psi(k) + \psi(N) + \log c_d + \frac{d}{N} \sum_{i=1}^{N} \log \epsilon(i),
$$

where:
- $\psi$ is the _digamma function_, the derivative of the logarithm of the gamma function $\Gamma(x)$,
- $k$ is the number of nearest neighbors,
- $\epsilon_i$ is twice the distance from $x_i$ to its $k^{th}$ nearest neighbor, representing the diameter of the hypersphere encompassing the $k$ neighbors,
- $c_d$ is the volume of the unit ball in $d$-dimensional space, where $\log c_d = 0$ for the maximum norm and $c_d = \pi^{d/2} / (\Gamma(1 + d/2) \cdot 2^d)$ for Euclidean spaces.


## Usage
This is a test of the entropy KL estimator (as developed above) on synthetically generated Gaussian distributed datasets. Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.

....code showing the usage of K-L estimator...

The estimator is implemented in the {py:class}`KozachenkoLeonenkoEntropyEstimator <infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

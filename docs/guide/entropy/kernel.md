---
file_format: mystnb
kernelspec:
  name: python3
---

(kernel_entropy)=
# Kernel Entropy Estimation
The Shannon differential {ref}`entropy <entropy_overview>` formula for a continuous random variable $ X $ with density $ p(x)$ is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\int_{X} p(x) \log p(x) \, dx,
$$

where $p(x)$ is the probability density function (pdf).

Kernel entropy estimation relies on probability density function (pdf) estimates obtained via **kernel density estimation (KDE)** to approximate the required probability in the above formula. Density estimation involves constructing an estimate of the pdf from the available dataset. KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`. Nearby points contribute more to the estimate, while distant points contribute less. The KDE estimate at a point $x_n$ is given by:

$$
    \hat{p}_r(x_n) = \frac{1}{N r^d} \sum_{n'=1}^{N} K \left( \frac{x_n - x_{n'}}{r} \right),
$$
where
- $N$ is the number of data points,
- $r$ is the bandwidth or kernel radius,
- $d$ is the dimension of the data,
- $x_n$ and $x_{n'}$ are the data points,
- $\hat{p}_r(x_n)$ is the estimated probability density for each data point.
For multivariate kernel functions, the pdf is estimated by dividing by a factor of $r^d$, where $d$ is the number of dimensions. The estimated pdf is then used to compute the Shannon entropy.

This package supports two types of kernel functions:

1. **Box Kernel (Step Kernel):**

   $$
   K = \begin{cases}
      0 & \text{if } |u| \geq 1 \\
       1 & \text{otherwise}
   \end{cases}
   $$
   where $\hat{p}_r(x_n)$ is computed as the fraction of $N$ points within a distance $r$ from $x_n$.
   In higher dimensions, the distance is calculated with the $L_\infty$ norm.
   From the rectangular shape, the kernel gets its name.

2. **Gaussian Kernel:**

   $$
   K(r) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}r^2},
   $$
   providing a smooth decline in weight with increasing distance from $x_n$.

```{tip}
Kernel estimation is model-free but depends on the Kernel-width parameter $(r)$. A small $(r)$ can lead to under-sampling, while a large $(r)$ may over-smooth the data, obscuring details.
```

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

h = im.entropy(data, approach="kernel", kernel="box", bandwidth=0.5)
h_expected = (1 / 2) * np.log(2 * np.pi * np.e * std ** 2)
h, h_expected
```

Comparing the gaussian kernel:

```{code-cell}
im.entropy(data, approach="kernel", kernel="gaussian", bandwidth=0.5), h_expected
```

To access the local values, an estimator instance is needed.

```{code-cell}
est = im.estimator(data, measure="h", approach="kernel", kernel="box", bandwidth=0.5)
est.result(), est.local_vals()
```

For a 2D point cloud, it is as easy to calculate the entropy

```{code-cell}
im.entropy(
    data=rng.normal(loc=0, scale=1, size=(2000, 2)),
    approach="kernel", kernel="box", bandwidth=0.5
)
```

or the local values, as shown before.

The estimator is implemented in the {py:class}`KernelEntropyEstimator <infomeasure.estimators.entropy.kernel.KernelEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.


[//]: # (Not sure if we want to include this everywhere)
```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.kernel.KernelEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

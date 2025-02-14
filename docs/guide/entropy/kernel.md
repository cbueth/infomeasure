---
file_format: mystnb
kernelspec:
  name: python3
---

(kernel_entropy)=
# Kernel Entropy Estimation
The Shannon differential {ref}`entropy_overview` formula for a continuous random variable $ X $ with density $ p(x)$ is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\int_{X} p(x) \log p(x) \, dx,
$$

where, $p(x)$ is the probability density function(_pdf_).

``Kernel entropy estimation`` relies on probability density function (_pdf_) estimates obtained via **kernel density estimation (KDE)** to approximate the required probability in above formula. Density estimation involves constructing an estimate of the _pdf_ from the available dataset. KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`. Nearby points contribute more to the estimate, while distant points contribute less. The KDE estimate at a point $x_n$ is given by:

$$
    \hat{p}_r(x_n) = \frac{1}{N r^d} \sum_{n'=1}^{N} K \left( \frac{x_n - x_{n'}}{r} \right).
$$
where:
- $N$ is the number of data points,  
- $r$ is the bandwidth or kernel radius,  
- $d$ is the dimension of the data,  
- $x_n$ and $x_{n'}$ are the data points,  
- $\hat{p}_r(x_n)$ is the estimated probability density.
For multivariate kernel functions, the **_pdf_** is estimated by dividing by a factor of $r^d$, where $d$ is the number of dimensions. Thus estimated **_pdf_** is then used to compute the Shannon entropy.  

``kernel functions:``  
This package supports two types of kernel functions:  

1. **Box Kernel (Step Kernel):**  
   Defined as:  

   $$
   K(|u| \geq 1) = 0, \quad K(|u| < 1) = 1,
   $$
   where $\hat{p}_r(x_n)$ is computed as the fraction of $N$ points within a distance $r$ from $x_n$.  

2. **Gaussian Kernel:**  
   Defined as:  

   $$
   K(r) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}r^2},
   $$
   providing a smooth decline in weight with increasing distance from $x_n$.  

> Note: 
> Kernel estimation is model-free but depends on the Kernel-width parameter $(r)$. A small $(r)$ can lead to under-sampling, while a large $(r)$ may over-smooth the data, obscuring details.  

## Implementation
This is a test of the entropy kernel estimator (as developed above) on synthetically generated Gaussian distributed datasets. 
Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.

....code showing the usage of kernel estimator...

The estimator is implemented in the {py:class}`KernelEntropyEstimator <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.


[//]: # (Not sure if we want to include this everywhere)
```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kernel.KernelEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

---
file_format: mystnb
kernelspec:
  name: python3
---

(kernel_entropy)=
# Kernel Entropy Estimation
The Shannon {cite:p}`shannonMathematicalTheoryCommunication1948` differential entropy formula is given as:

$$
H(X) = -\int_{X} p(x) \log_b p(x) \, dx,
$$

where $x$ denotes the realizations of the random variable $X$ with probability $p(x)$, and $b$ is the base of the logarithm. Further details can be read in the section {ref}`Entropy / Uncertainty`.

``Kernel entropy estimation`` relies on probability density function (_pdf_) estimates obtained via **kernel density estimation (KDE)** to approximate the required probability in the given formula. Density estimation involves constructing an estimate of the _pdf_ from the available dataset. KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function \(K\) {cite:p}`silverman1986density`. Nearby points contribute more to the estimate, while distant points contribute less. The KDE estimate at a point \(x_n\) is given by:

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

> **Note:** Kernel estimation is model-free but depends on the Kernel-width parameter $(r)$. A small \(r\) can lead to under-sampling, while a large \(r\) may oversmooth the data, obscuring details.  

## Implementation
This is a test of the entropy kernel estimator (as developed above) on synthetically generated Gaussian distributed datasets. 
Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import infomeasure as im

# Define the KDE and entropy computation functions : already defined above

# Function to calculate analytical entropy for a normal distribution with standard deviation sigma
def analytical_entropy(sigma):
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

# Generate a range of standard deviations
std_devs = np.linspace(0.1, 2.0, 100)

# Calculate the analytical entropy values
analytical_entropies = [analytical_entropy(sd) for sd in std_devs]

# Choose some standard deviations to compute the numeric entropy
selected_std_devs = [0.5, 1.0, 1.5, 2.0]
numeric_entropies = []

# Compute the numeric entropy for the selected standard deviations
for sd in selected_std_devs:
    # Generate sample data from a normal distribution
    data = np.random.normal(0, sd, size=1000)
    # Compute the entropy using the numeric approach
    h = im.entropy(data, approach="kernel", bandwidth=0.3, kernel='box')
    numeric_entropies.append(h)

# Plot the analytical entropy as a function of standard deviation
plt.figure(figsize=(10, 5))
plt.plot(std_devs, analytical_entropies, label='Analytical Entropy')

# Plot the numeric entropy values
for i, sd in enumerate(selected_std_devs):
    plt.scatter(sd, numeric_entropies[i], label=f'Numeric Entropy (SD={sd})')

plt.xlabel('Standard Deviation')
plt.ylabel('Entropy')
plt.title('Comparison of Analytical and Numeric Entropy Values')
plt.legend()
plt.grid(True)
plt.show()
```

The estimator is implemented in the {py:class}`KernelEntropyEstimator <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.


[//]: # (Not sure if we want to include this everywhere)
```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kernel.KernelEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

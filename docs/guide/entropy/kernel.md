---
file_format: mystnb
kernelspec:
  name: python3
---

# Kernel Entropy Estimation

Data in the real world is often continuous, and estimating the entropy of continuous random variables is a common problem in statistics and machine learning.
The kernel density estimation (KDE) method is a popular approach for estimating the probability density function of a continuous random variable.
The KDE method uses a kernel function to estimate the density of the data at each point.
So the difference to the discrete entropy estimation is...

KDE Formula:

$$
\hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i)
$$

where:
- $n$ is the number of data points,
- $x_i$ are the data points,
- $K_h$ is the kernel function with bandwidth $h$.
- The entropy is then calculated as...
- The KDE method is a non-parametric method, meaning that it does not assume a specific form for the probability density function.
- The choice of kernel function and bandwidth can have a significant impact on the quality of the density estimate.


## Test: Entropy Estimator
This is a test of the entropy kernel estimator (as developed above) on synthetically generated Gaussian distributed datasets. Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.


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
this work is based on {cite:p}`acharya2024representative`  # TODO: add reference



The estimator is implemented in the {py:class}`KernelEntropyEstimator <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.


[//]: # (Not sure if we want to include this everywhere)
```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kernel.KernelEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

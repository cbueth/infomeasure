---
file_format: mystnb
kernelspec:
  name: python3
---

(entropy_kozachenko_leonenko)=
# Kozachenko-Leonenko (KL) / Metric / kNN Entropy Estimation


The Kozachenko-Leonenko (KL) entropy estimator is a non-parametric method for estimating the entropy of a continuous random variable based on the $k^{th}$-Nearest Neighbor (KNN) distances of the data points.
For simplicity imagine point in 2-D space $R^2$  and the idea is to rank the each points (say each point $z_i = (x_i, y_i) \in R^2$) by its neighbour´s distance $d_{i,j} = \|z_i - z_j\| : d_{i,j1} \le d_{i,j2} \le \cdots$ (supposing $\|\cdot\|$ be a metric).
Finally estimate $H(X)$ from the average distance to the $k$-nearest neighbor.
The full derivation of KL estimator for entropy calculation can be found in following citations {cite:p}`kozachenko1987sample` {cite:p}`RevieEstimators` {cite:p}`miKSG2004` ,here we will just write the final expression:

$$
    H(X) = \psi(N) - \psi(k) + \log c_d + \frac{d}{N} \sum_{i=1}^{N} \log \epsilon_i
$$
where,
- $\psi$ is the _digamma function_, the derivative of the logarithm of the gamma function $\Gamma(x)$,
- $k$ number of neared neighbour,
- $\epsilon_i$ is twice the distance from $x_i$ to its $k^{th}$ nearest neighbor,
- $c_d$ is the volume of the unit ball in the $d$-dimensional space and for the maximum norm, $\log c_d$ becomes zero, for Euclidean, $c_d = \pi^{d/2} / \Gamma(1 + d/2) / 2^d$.

## Test: KL Estimator
This is a test of the entropy KL estimator (as developed above) on synthetically generated Gaussian distributed datasets. Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.

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
    data = np.random.normal(0, sd, size=10000)
    # Compute the entropy using the numeric approach
    entropy = im.entropy(data, approach="kl", k=4, noise_level=1e-8)
    #entropy = im.entropy(data, approach="kl")
    numeric_entropies.append(entropy)

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


The estimator is implemented in the {py:class}`KozachenkoLeonenkoEntropyEstimator <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

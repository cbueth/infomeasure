---
file_format: mystnb
kernelspec:
  name: python3
---

# Entropy Measures by kernel estimation technique

Hello, here I am going to write the formula for the kernel methods.

## Test: Entropy Estimator
This is a test of the entropy kernel estimator (as developed above) on synthetically generated Gaussian distributed datasets. Since there is an analytical function for computing the entropy (H) for a Gaussian distribution, this allows us to check if our estimator's estimates are close to the analytical values.


```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from infomeasure.estimators.kernel import entropy as compute_entropy

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
    entropy = compute_entropy(data, bandwidth=0.3, kernel='box')
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
this work is based on {cite:p}`acharya2024representative`

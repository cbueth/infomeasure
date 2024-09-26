---
file_format: mystnb
kernelspec:
  name: python3
---

# Kernel Entropy Estimation

Kernel entropy estimation technique relies on the probability density function (pdf) estimates as estimated by the **kernel density estimation (KDE)** method. 
Density estimation is about constructing the estimate of density function from the available dataset. 
KDE taken an approach of estimating the density of a reference point by potentially considers all samples and weighting them based on their distance from a reference point with the help of Kernel Function $K$ {cite:p}`silverman1986density`.  
Essentially this will allow for the points near to the reference points to contribute more to the density than for the points further way. 
The KDE estimate of a point $ x_n $ is calculated by following relation:

$$
    \hat{p}_r(x_n) = \frac{1}{N r^d} \sum_{n'=1}^{N} K \left( \frac{x_n - x_{n'}}{r} \right) 
$$
where:
- $N$ is the number of data points,
- $r$ is bandwidth or kernel radius,
- $d$ is the dimension of the data, 
- $x_n$ and $x_{n'}$ are the data points,
- $\hat{p}_r(x_n)$ probability estimates.

In order to take care for the multivariate kernel functions, in the above equation _pdf_ is estimated by dividing by a factor of $r$, or $r^d$ with $d$ being the number of dimensions.
This package allows to implement two different kernel functions: box kernel or step kernel, defined as
$( K (|u| \geq 1) = 0 )$ 
and 
$( K(|u| < 1) = 1 )$,
which basically calculates 
$ ( \hat{p}_r(x_n) )$
as the fraction of the $N$ values situated within a distance $r$ from $x_n$ and second, 
gaussian kernel 
$( \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}r^2} )$ 
offering a gradual decline at the bin boundaries away from $x_n$. 

We use the estimated _pdf_ to compute the Shannon differential entropy by averaging the computed Shannon information content over all samples. 


> Note: Even though Kernel estimation is a model-free technique, but its accuracy hinges on the choice of resolution parameter $( r )$. Picking the right value is challenging: a small value can result in under-sampling, while a large one might gloss over data nuances.

### Test: pdf estimator
Let´s test the KDE function (as implemented in the package) to estimate  the pdf of a RV $X$. The RV $X$ in this case will be a synthetically generated gaussian dataset. 

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Rememebr to implement the kde_probability_density_function as it has been defined in the package

# Generate sample data from a normal distribution for 1D and 2D cases
data_1d = np.random.normal(0, 1, size=(1000, 1))
data_2d = np.random.normal(0, 1, size=(1000, 2))

# Set the bandwidth for the kernel
bandwidth = 0.3

# Create a grid of points for plotting
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Compute the KDE values on the grid for 1D data
# Adjust the loop to use the new function directly
pdf_1d = np.array([kde_probability_density_function(data_1d, np.array([x]), bandwidth, kernel='box') for x in x_grid])

# Compute the KDE values on the grid for 2D data
# Adjust the loop to use the new function directly for 2D points
grid_points_2d = np.vstack([X.ravel(), Y.ravel()]).T
pdf_2d = np.array([kde_probability_density_function(data_2d, point, bandwidth, kernel='box') for point in grid_points_2d]).reshape(X.shape)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot the 1D KDE
axs[0].plot(x_grid, pdf_1d, label='Box KDE')
axs[0].set_title('1D Data PDF')
axs[0].set_xlabel('Data')
axs[0].set_ylabel('Probability Density')

# Plot the 2D KDE
cax = axs[1].pcolormesh(X, Y, pdf_2d, shading='auto')
fig.colorbar(cax, ax=axs[1])
axs[1].set_title('2D Data PDF')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()
```

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

The estimator is implemented in the {py:class}`KernelEntropyEstimator <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.


[//]: # (Not sure if we want to include this everywhere)
```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kernel.KernelEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

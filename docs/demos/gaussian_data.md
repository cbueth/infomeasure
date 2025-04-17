---
file_format: mystnb
kernelspec:
  name: python3
---

(info_estimator_demo)=
# Gaussian Data

In this notebook, we want to demonstrate how to use {ref}`entropy_overview` and {ref}`mutual_information_overview`,
and showcase the different approaches on gaussian random data.

```{code-cell}
import infomeasure as im
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(29615)
```

## Entropy

Generate normal distributed random variables with varying standard deviation $\sigma$.

```{code-cell}
n = 2000
stds = np.linspace(0.1, 10, 20)  # standard deviations to test
# data is an array of shape (len(stds), n)
data_h = rng.normal(0, stds[:, None], (len(stds), n))
```

Let's look at the first time series and plot the first 200 samples.
Also, compare the histograms of the time series for different $\sigma$ values.

```{code-cell}
:tags: [full-width,hide-input]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

m = 200
axes[0].plot(data_h[0])
axes[0].grid()
axes[0].set_xlabel('Number of samples')
axes[0].set_xlim(-10, m)
axes[0].set_ylabel('Value')
axes[0].set_title(
    fr'Normal distributed time series ($\sigma={stds[0]:.2f}$, first {m} samples)')

cmap = plt.get_cmap('viridis')
for i in range(len(stds)):
    hist, bins = np.histogram(data_h[i], bins=20, density=True)
    axes[1].plot(hist, bins[:-1], color=cmap(i / len(stds)), label=(f'std={stds[i]:.2f}' if i in [0, len(stds)//2, len(stds)-1] else None))
axes[1].set_ylim(-20, 20)
axes[1].set_xlim(0, 0.25)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].set_title(
    fr'Histogram for the normal distributed time series')
axes[1].legend()
axes[1].grid()

plt.show()
```

To compare with an expected value, we can calculate the analytical entropy for each $\sigma$ value:

$$
H(X) = \frac{1}{2} \log(2\pi e \sigma^2)
$$

```{code-cell}
def analytical_entropy(sigma, base=im.Config.get('base')):
    """Analytical entropy of a normal distributed random variable
    with standard deviation sigma."""
    h = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    if base == 'e':
        return h
    return h / np.log(base)
```

This is the value we should expect when calculating the global value for the different approaches.
To compute the numerical entropy for all approaches, loop over each $\sigma$ value and approach.
For the `discrete` approach, the values need to be discretized to integers.

```{code-cell}
approaches = {
    'discrete': {},
    'kernel': {'bandwidth': 0.5, 'kernel': 'box'},
    'metric': {'k': 4, 'minkowski_p': 2},
    'renyi': {'alpha': 1},  # alpha=1 is the Shannon entropy
    'ordinal': {'embedding_dim': 2},
    'tsallis': {'k': 4, 'q': 1}  # q=1 is the Shannon entropy
}
entropies = np.zeros((len(approaches), len(stds)))
for i, std in enumerate(stds):
    for j, name, kwargs in zip(range(len(approaches)), approaches.keys(),
                               approaches.values()):
        if name == 'discrete':
            entropies[j, i] = im.entropy(data_h[i].astype(int), approach=name, **kwargs)
        else:
            entropies[j, i] = im.entropy(data_h[i], approach=name, **kwargs)
```

`infomeasure` enables us to use the {py:func}`im.entropy() <infomeasure.entropy>` function, passing the `approach`-name and additional keyword arguments for the corresponding approach.
Like this, the single call `im.entropy(data_h[i], approach='discrete')` already computes the entropy of a discrete distribution.
Let's plot the entropy for each approach and $\sigma$ value, while comparing it with the expected analytical entropy:

```{code-cell}
:tags: [hide-input]
# Compute analytical values (higher precision)
analytical_stds = np.linspace(min(stds) / 10, max(stds), 100)
analytical_entropies = np.array([analytical_entropy(std) for std in analytical_stds])
analytical_entropies_stds = np.array([analytical_entropy(std) for std in stds])

# Plot results
fig = plt.figure(figsize=(8, 5), dpi=300)
# for i, name in enumerate(approaches.keys()):
#     plt.plot(stds, entropies[i], label=name, marker='o', linewidth=3)
# linewith is decreasing with the index, so we see which are hidden
for i, name in enumerate(approaches.keys()):
    plt.plot(stds, entropies[i], label=name, marker='o', linewidth=5 - i / 2.)

# Analytical values
plt.plot(analytical_stds, analytical_entropies, label='Analytical', linewidth=1.8,
         color='black')

plt.xlabel('Standard deviation')
plt.ylabel('Entropy')
plt.ylim(-1, 4)
plt.title('Entropy estimation of Gaussian data')
plt.legend()
plt.grid()
plt.show()
```

All methods match the analytical values very closely,
except the `ordinal` and `discrete` ones.
For the lower $\sigma$, the discretization leads to a deviation.
For $\sigma = 0.1$ all values get converted to 0, producing entropy of 0.
The `ordinal` method returns constant values.
This is to be expected because the symbolization is independent of $\sigma$,
as the distributions of ordinal patterns are invariant under scaling.

```{code-cell}
:tags: [hide-input]
# Plot the percentage error of the estimated entropy compared to the analytical value
fig = plt.figure(figsize=(8, 3), dpi=300)
symbols = ['o', 's', 'd', '^', 'v', 'x']
for i, name in enumerate(approaches.keys()):
    plt.plot(stds,
             100 * (entropies[
                        i] - analytical_entropies_stds) / analytical_entropies_stds,
             label=name, marker=symbols[i], markersize=10,
             linewidth=2, linestyle='--')
# horizontal line at 0
plt.axhline(0, color='black', linewidth=2, zorder=-1)
plt.xlabel(r'Standard deviation $\sigma$')
plt.ylabel(r'Percentage error (%)')
plt.ylim(-25, 10)
plt.legend()
plt.grid()
plt.show()
```

`ordinal` is off the charts, mostly around $-75\%$.

## Mutual Information

For {ref}`mutual_information_overview`, we demonstrate the measures with
multivariate Gaussian data, varyig the correlation coefficient $\rho$.
Analytically, the mutual information between two gaussian random variables $X$ and $Y$
with zero mean and unit variance is given by

$$
I(X; Y) = -\frac{1}{2} \log(1 - \rho^2).
$$

```{code-cell}
# Analytical formula for the mutual information of two Gaussian random variables
def mutual_information_gauss(X, Y):
    """Compute the mutual information between two Gaussian random variables.
    
    Notes
    -----
    ``r`` is the correlation coefficient between X and Y.
    ``I_Gauss`` is the mutual information between X and Y.
    """
    r = np.corrcoef(X, Y)[0, 1]
    I_Gauss = -0.5 * np.log(1 - r ** 2)
    return I_Gauss
```

```{code-cell}
:tags: [hide-input]
def generate_data(N, r):
    cov_matrix = [[10, r], [r, 10]]
    data = np.random.multivariate_normal([0, 0], cov_matrix, N)
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y
    
# Values of r
r_values = np.linspace(1, 9, 9)
# Number of data points
N = 2000
# Generate data for each r value. shape (len(r_values), 2, N)
data_mi = np.array([generate_data(N, r) for r in r_values])
data_mi_discrete = data_mi.astype(int)
```

```{code-cell}
:tags: [hide-input]
# Visualize data - plot on top of each-other, low to higher r values
from matplotlib import colormaps
plt.figure(figsize=(8, 4), dpi=300)
ax = plt.gca()
cmap = colormaps.get_cmap('viridis')
norm = plt.Normalize(vmin=min(r_values/10), vmax=max(r_values/10))
for i in range(len(r_values)):
    plt.scatter(data_mi[i][0], data_mi[i][1], color=cmap(norm(r_values[i]/10)), marker='.')
# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar=plt.colorbar(sm, ticks=r_values/10, ax=ax)
cbar.set_label(r'Correlation Coefficient $\rho$')
plt.title(r'Joint Distribution $(X, Y)$ for varying $\rho$')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()
```

For the lower correlated data points, the scatter plot becomes more scattered.
For higher correlated data points, the scatter plot becomes more aligned.
Again, prompt the {py:func}`im.mutual_information() <infomeasure.mutual_information>` function for each approach.

```{code-cell}
# Compute mutual information for each r value
approaches_mi = {
    'discrete': {},
    'kernel': {'bandwidth': 0.5, 'kernel': 'gaussian'},
    'metric': {'k': 4},
    'renyi': {'alpha': 1.5},  # alpha=1 is the Shannon entropy
    'ordinal': {'embedding_dim': 3},
    'tsallis': {'k': 4, 'q': 1.05}
}
mi_values = {approach: np.zeros((len(r_values))) for approach in approaches_mi.keys()}
for i, r in enumerate(r_values):
    for name, kwargs in approaches_mi.items():
        if name == 'discrete':
            mi_values[name][i] = im.mutual_information(
                data_mi_discrete[i][0], data_mi_discrete[i][1], approach=name, **kwargs
            )
        else:
            mi_values[name][i] = im.mutual_information(
                data_mi[i][0], data_mi[i][1], approach=name, **kwargs
            )
```

The call of `im.mutual_information(data_mi[i][0], data_mi[i][1], approach=name, **kwargs)`
performs the computation of mutual information between the two variables `data_mi[i][0]` and `data_mi[i][1]` using the specified approach with the settings given in `kwargs`.

```{code-cell}
:tags: [hide-input]
# Plot results
fig = plt.figure(figsize=(8, 5), dpi=300)
for name in approaches_mi.keys():
    plt.plot(r_values, mi_values[name], label=name, marker='o', linewidth=3)

# Analytical values
mi_gauss_values = [mutual_information_gauss(data_mi[i][0], data_mi[i][1]) for i in
                   range(len(r_values))]
plt.plot(r_values, mi_gauss_values, label='Analytical', linewidth=1.8, color='black')

plt.xlabel('Correlation coefficient (r)')
plt.ylabel('Mutual Information')
plt.title('Mutual Information vs Correlation Coefficient')
plt.legend()
plt.grid()
plt.show()
```

All approaches resemble the expected behavior of mutual information for Gaussian data.
Again, the `discrete` approach has an offset, due to discretization errors.
Tsallis MI is offset as well, because $q=1.05$ has been used.
For $q=1$, Tsallis MI is identical to Shannon MI.
This confirms that the implemented methods correctly estimate the mutual information for Gaussian data.

## Conclusion

In this notebook, we have explored various approaches estimating Entropy and Mutual Information for Gaussian data using `infomeasure`.
We also visualised the results and compared them to the analytical expectations.
Using this package enables to seamlessly switch estimation approaches and easily find the must suited method for a given dataset and available compute.
We focused on the global values of the measures, but not considered the {ref}`Local Entropy` or {ref}`Local Mutual Information`, neither {ref}`transfer_entropy_overview`.
For a transfer entropy demonstration, find the reproduction of the {ref}`schreiber_article` {cite:p}`Schreiber.paper` on the next page.
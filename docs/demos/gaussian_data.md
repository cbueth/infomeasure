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
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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


```{warning}
This section is under construction. Please check back in a few days for updates.
```


---
file_format: mystnb
kernelspec:
  name: python3
---
(kernel_TE)=
# Kernel TE Estimation
The {ref}`transfer_entropy_overview` from the source process $X(x_n)$ to the target process $Y(y_n)$ in terms of probabilities is written as:

$$
T_{x \rightarrow y}(k, l) = -\sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right)
$$

where
- $y_{n+1}$ is the next state of $Y$ at time $n$,
- $ \mathbf{y}_n^{(l)} = \{y_n, \dots, y_{n-l+1}\} $ is the embedding vector of $Y$ considering the  $ l $ previous states (history length),
- $ \mathbf{x}_n^{(k)} = \{x_n, \dots, x_{n-k+1}\} $ embedding vector of $X$ considering the $ k $ previous states (history length),
- $p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability of the next state of $Y$, its history, and the history of $X$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of next state of $Y$ given the histories of $X$ and $Y$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$.

Kernel TE estimates the required probability density function (_pdf_) via **kernel density estimation (KDE)**, which provides the probability values to be plugged into the above formula {cite:p}`Schreiber.paper,articleKantz,TE_Kernel_Kaiser`.
KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`.
For more detail on _pdf_-estimation and available kernel functions check the {ref}`Kernel Entropy Estimation` section.

```{note}
This TE estimaton techinque offers two different kernel functions: box kernel and gaussian kernel.
 ```


```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(5673267189)

data_x = rng.normal(size=1000)
data_y = np.roll(data_x, 1)
data_control = rng.normal(size=1000)

(im.transfer_entropy(
    data_x,  # source
    data_y,  # target
    approach="kernel", kernel="box", bandwidth=0.7,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach="kernel", kernel="box", bandwidth=0.7,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
))
```

```{code-cell}
:tags: [remove-cell]
from numpy import set_printoptions
set_printoptions(precision=5, threshold=20)
```

For further methods, create an instance of the estimator.

```{code-cell}

est = im.estimator(
    data_x,  # source
    data_y,  # target
    measure='te',  # or 'transfer_entropy'
    approach="kernel", kernel="box", bandwidth=0.7,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
)
est.local_vals()
```

The {ref}`effective_te` method can be accessed like so:

```{code-cell}
est.effective_val()
```

{ref}`hypothesis testing` can also be conducted, with either a permutation test or bootstrapping.

```{code-cell}
est.p_value(n_tests = 50, method="permutation_test"), est.t_score()
```

Data of higher dimension can easily be digested.

```{code-cell}
data_x = rng.normal(size=(1000, 5))  # 5d data
data_y = rng.normal(size=(1000, 3))  # 3d data
im.te(data_x, data_y, approach="kernel", kernel="gaussian", bandwidth=0.5)
```

The estimator is implemented in the {py:class}`KernelTEEstimator <infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

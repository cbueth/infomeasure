---
file_format: mystnb
kernelspec:
  name: python3
---

# Rényi & Tsallis TE Estimation
Let us suppose two RVs $X$ and $Y$ represent time series processes as:

$$
\begin{aligned}
X_{t_n}^{(k)} &= (X_{t_n}, X_{t_n-1}, \ldots, X_{t_n-k+1}),\\
Y_{t_n}^{(l)} &= (Y_{t_n}, Y_{t_n-1}, \ldots, Y_{t_n-l+1}),
\end{aligned}
$$

The {ref}`transfer_entropy_overview` from **$X \to Y$** can be expressed using the formulation of entropy and joint entropy as follows {cite:p}`khinchin1957mathematical,cover2012elements`:

$$
T_{X \rightarrow Y} = I(Y_{t_{n+1}} ; X_{t_n}^{(k)} \,|\, Y_{t_n}^{(l)}) = H(Y_{t_{n+1}}, Y_{t_n}^{(l)}) + H(X_{t_n}^{(k)}, Y_{t_n}^{(l)}) - H(Y_{t_{n+1}}, X_{t_n}^{(k)}, Y_{t_n}^{(l)}) - H(Y_{t_n}^{(l)})
$$

where
- $H(Y_{t_{n+1}}, Y_{t_n}^{(l)})$ is the joint entropy of $Y_{t_{n+1}}$ and its history $Y_{t_n}^{(l)}$.
- $H(X_{t_n}^{(k)}, Y_{t_n}^{(l)})$ is the joint entropy of the histories $X_{t_n}^{(k)}$ and $Y_{t_n}^{(l)}$.
- $H(Y_{t_{n+1}}, X_{t_n}^{(k)}, Y_{t_n}^{(l)})$ is the joint entropy of $Y_{t_{n+1}}$, $X_{t_n}^{(k)}$, and $Y_{t_n}^{(l)}$.
- $H(Y_{t_n}^{(l)})$ is the entropy of the history $Y_{t_n}^{(l)}$.

Rényi TE estimate is computed by plugging-in the entropies and the join entropy estimates by using the estimation method explained in the {ref}`Rényi Entropy Estimation <renyi_entropy>`.

Tsallis TE estimate is computed by plugging-in the entropies and the join entropy estimates by using the estimation method explained in the {ref}`Tsallis Entropy Estimation <tsallis_entropy>`.

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
    approach="renyi", alpha=1.2,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach="renyi", alpha=1.2,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_y,  # target
    approach="tsallis", q=1.2,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach="tsallis", q=1.2,
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
    approach="renyi", alpha=1.2,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
)
est.effective_val()
```

Like this, {ref}`effective_te` can be accessed.
{ref}`Local Transfer Entropy` is not supported for the Renyi and Tsallis TE.

{ref}`hypothesis testing` can be conducted, with either a permutation test or bootstrapping.

```{code-cell}
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
stat_test.p_value, stat_test.t_score, stat_test.confidence_interval(90), stat_test.percentile(50)
```

Data of higher dimension can easily be digested.

```{code-cell}
data_x = rng.normal(size=(1000, 5))  # 5d data
data_y = rng.normal(size=(1000, 3))  # 3d data
im.te(data_x, data_y, approach="tsallis", q=0.9)
```

The estimator is implemented in the {py:class}`RenyiTEEstimator <infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

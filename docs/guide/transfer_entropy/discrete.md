---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_TE)=
# Discrete TE Estimation
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

Discrete TE estimation estimates the required probability mass function (_pmf_) by counting the occurrences of matching configuration in the dataset by keeping the record of the frequencies.
These _pmf_ estimates are then plugged back into the above expression to estimate the TE.
This estimator is simple and computationally efficient.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(5673267189)

data_x = rng.integers(2, size=1000)
data_y = np.roll(data_x, 1)
data_control = rng.integers(2, size=1000)

(im.transfer_entropy(
    data_x,  # source
    data_y,  # target
    approach='discrete',
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
    base = 2,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach='discrete',
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
    base = 2,
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
    approach='discrete',
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
    base = 2,
)
est.local_vals()
```

The {ref}`effective_te` method can be accessed like so:

```{code-cell}
est.effective_val()
```

{ref}`hypothesis testing` can also be conducted, with either a permutation test or bootstrapping.

```{code-cell}
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
stat_test.p_value, stat_test.t_score, stat_test.confidence_interval(90), stat_test.percentile(50)
```

The estimator is implemented in the {py:class}`DiscreteTEEstimator <infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator>` class,
which is part of the {py:mod}`im.measures.transfer_entropy <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

```{code-cell}
import infomeasure as im
source = [0, 1, 0, 1, 0, 1, 0, 1]
target = [0, 0, 1, 1, 0, 0, 1, 1]
im.transfer_entropy(source, target, approach="discrete", base=2)
```

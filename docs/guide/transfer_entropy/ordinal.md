---
file_format: mystnb
kernelspec:
  name: python3
---
(ordinal_TE)=
# Ordinal / Symbolic / Permutation TE Estimation
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

Ordinal MI estimates the required probability density function (_pdf_) based on the ordinal structure {cite:p}`Symbolic_TE`.
The details on the _pdf_ estimation based on ordinal structure by {cite:t}`PermutationEntropy2002`, is provided in {ref}`Ordinal / Symbolic / Permutation Entropy Estimation`.


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
    approach="ordinal", embedding_dim = 3, 
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach="ordinal", embedding_dim = 3, 
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
    approach="ordinal", embedding_dim = 3, 
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

The estimator is implemented in the {py:class}`OrdinalTEEstimator <infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

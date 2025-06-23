---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_TE)=
# Kraskov-Stoegbauer-Grassberger TE Estimation
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
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$,
$\langle \cdot \rangle$ represents the expectation operator.

**Kraskov-Stoegbauer-Grassberger (KSG) TE estimator** adapts the {ref}`KSG_MI` technique and makes it suitable for estimating the TE between source and target variable {cite:p}`article_KSG_TE`.
Similar to MI estimation, it uses the advantage that the {ref}`entropy_kozachenko_leonenko`  for entropy {cite:p}`kozachenko1987sample` holds for any value of the nearest neighbour $k$.
Therefore, one can vary the value of $k$ in each data point in such a way that the radius (distance) of the corresponding $\epsilon$-balls would be approximately the same for the joint and the marginal spaces.
That means the distance is computed in the joint space for the fixed $k$ nearest neighbour, and then it is projected into the marginal spaces.
Following the algorithm one, the expression for the TE is as follows:

$$
TE(X \to Y, u) = \psi(k) + \left\langle \psi \left( n_{\mathbf{y}_n^{(l)}} + 1 \right)
- \psi \left( n_{y_{n+1}, \mathbf{y}_n^{(l)}} + 1 \right)
- \psi \left( n_{\mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}} + 1 \right) \right\rangle_n
$$

where
- $\psi(\cdot)$ denotes the **digamma function**,
- $\langle \cdot \rangle$ represents the expectation operator.
- $ n_x(\cdot) $ refers to the number of neighbors which are with in a hypercube that defines the search range around a statevector, the size of the hypercube in each of the marginal spaces is defined based on the distance to the $k-th$ nearest neighbor in the highest dimensional space.

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
    approach="metric",
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1,
),
 im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach="metric",
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
    approach="metric",
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
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
stat_test.p_value, stat_test.t_score, stat_test.confidence_interval(90), stat_test.percentile(50)
```

Data of higher dimension can easily be digested.

```{code-cell}
data_x = rng.normal(size=(1000, 5))  # 5d data
data_y = rng.normal(size=(1000, 3))  # 3d data
im.te(data_x, data_y, approach="metric")
```

The estimator is implemented in the {py:class}`KSGTEEstimator <infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

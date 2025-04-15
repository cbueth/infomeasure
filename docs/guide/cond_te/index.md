---
file_format: mystnb
kernelspec:
  name: python3
---

(cond_te_overview)=
# Conditional TE
{ref}`Transfer Entropy <transfer_entropy_overview>` (TE) from the source process $X$ to the target process $Y$ can also be conditioned on other possible sources, such as $Z$. In that case, the conditional TE corresponds to the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$, $Z$ and also after considering the past values of target $Y$ itself.
Importantly, the TE can be conditioned on other possible information sources $Z$, to eliminate their influence from being mistaken as that of the source $Y$.

$$
TE(X \to Y \mid Z) = -\sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}}
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)})
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)})} \right)
$$

where
- $p(\cdot)$ represents the probability distribution,
- $\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$,
- $\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$,
- $\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$,
- $y_{n+1}$ is the future state of $Y$.

## Local Conditional TE
Similar to {ref}`Local Conditional H` and {ref}`Local Conditional MI` measures, we can extract the **local or point-wise conditional transfer entropy** as suggested by Lizier _et al._ {cite:p}`Lizier2014,local_TE_Lizier`.
It is the amount of information transfer attributed to the specific realization $(x_{n+1}, \mathbf{X}_n^{(k)}, \mathbf{Y}_n^{(l)})$ at time step $n+1$; i.e., the amount of information transfer from process $X$ to $Y$ at time step $n+1$:

$$
t_{X \rightarrow Y \mid Z}(n+1, k, l) = -\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n)}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{z}_n)} \right)
$$

The known TE can be recovered from the local TE:

$$
T_{X \rightarrow Y \mid Z}(k, l) = \langle t_{X \rightarrow Y}(n + 1, k, l) \rangle,
$$

This package also allows the user to calculate the {ref}`Local Values`.

## CTE Estimation
The CTE expression above can be written as the combination of entropies and joint entropies as follows:

$$
\begin{align}
TE(X \to Y \mid Z) =\,&H(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)})
- H(\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)})\\
&- H(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)})
+ H(\mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}).
\end{align}
$$

While the package uses this formula internally for the Rényi and Tsallis CTE, all other approaches each are calculated with dedicated, probabilistic implementations.



```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(56734267189)

data_x = rng.integers(2, size=1000)
data_y = np.roll(data_x, 1)
data_z = data_x * rng.uniform(size=1000)

cte = im.cte(data_x, data_y, cond=data_z.astype(int), approach='discrete',
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1)
cte_ksg = im.cte(data_x, data_y, cond=data_z, approach='ksg',
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1)
cte_kernel = im.cte(data_x, data_y, cond=data_z, approach='kernel', kernel='box', bandwidth=1.5,
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1)
cte_symbolic = im.cte(data_x, data_y, cond=data_z, approach='symbolic', embedding_dim=3,
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1)
cte, cte_ksg, cte_kernel, cte_symbolic
```

```{code-cell}
:tags: [remove-cell]
from numpy import set_printoptions
set_printoptions(precision=5, threshold=20)
```

The {ref}`Local Conditional TE` is calculated as follows:

```{code-cell}
est = im.estimator(
    data_x, data_y, cond=data_z,
    measure='cte',  # or 'conditional_transfer_information'
    approach='metric',
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1
)
est.local_vals()
```


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: KSG method

   ksg_cond_te
 ```

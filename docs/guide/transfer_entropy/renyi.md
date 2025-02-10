---
file_format: mystnb
kernelspec:
  name: python3
---

# Rényi & Tsallis TE Estimation
[Transfer Entropy](index.md#transfer_entropy_overview) (TE) from the source process $X$ to the target process $Y$ is the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$ after considering the past values of target.

$$T_{X \rightarrow Y}(k, l) = I \left[ \mathbf{X}_n^{(k)}; Y_{n+1} \mid \mathbf{Y}_n^{(l)} \right].$$

Lets us suppose two RVs represent the time series dataset as:

$$
X_{t_n}^{(k)} = (X_{t_n}, X_{t_n-1}, \ldots, X_{t_n-k+1}),

Y_{t_n}^{(l)} = (Y_{t_n}, Y_{t_n-1}, \ldots, Y_{t_n-l+1}),
$$

The **Transfer Entropy from $X \to Y$** can be expressed using the formulation of entropy and joint entropy as follows {cite:p}`khinchin1957mathematical` {cite:p}`cover2012elements`:

$$
T_{X \rightarrow Y} = I(Y_{t_{n+1}} ; X_{t_n}^{(k)} \,|\, Y_{t_n}^{(l)}) = H(Y_{t_{n+1}}, Y_{t_n}^{(l)}) + H(X_{t_n}^{(k)}, Y_{t_n}^{(l)}) - H(Y_{t_{n+1}}, X_{t_n}^{(k)}, Y_{t_n}^{(l)}) - H(Y_{t_n}^{(l)}).
$$
Where:
- $H(Y_{t_{n+1}}, Y_{t_n}^{(l)})$ is the joint entropy of $Y_{t_{n+1}}$ and its history $Y_{t_n}^{(l)}$.
- $H(X_{t_n}^{(k)}, Y_{t_n}^{(l)})$ is the joint entropy of the histories $X_{t_n}^{(k)}$ and $Y_{t_n}^{(l)}$.
- $H(Y_{t_{n+1}}, X_{t_n}^{(k)}, Y_{t_n}^{(l)})$ is the joint entropy of $Y_{t_{n+1}}$, $X_{t_n}^{(k)}$, and $Y_{t_n}^{(l)}$.
- $H(Y_{t_n}^{(l)})$ is the entropy of the history $Y_{t_n}^{(l)}$.

**Rényi TE estimate** is computed by plugging-in the entropy and the join entropy estimates by using the estimation method explained in the {ref}`Rényi Entropy Estimation <renyi_entropy>`.
**Tsallis TE estimate** is computed by plugging-in the entropy and the join entropy estimates by using the estimation method explained in the {ref}`Tsallis Entropy Estimation <tsallis_entropy>`.

## Implementation
The estimator is implemented in the {py:class}`RenyiTEEstimator <infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

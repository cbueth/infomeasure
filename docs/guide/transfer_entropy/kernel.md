---
file_format: mystnb
kernelspec:
  name: python3
---
(kernel_TE)=
# Kernel TE Estimation
[Transfer Entropy](index.md#transfer_entropy_overview) (TE) from the source process $X$ to the target process $Y$ is the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$ after considering the past values of target.

$$T_{X \rightarrow Y}(k, l) = I \left[ \mathbf{X}_n^{(k)}; Y_{n+1} \mid \mathbf{Y}_n^{(l)} \right].$$
where,
- $\mathbf{X}_n^{(k)}$ is a vector/history of the past $k$ states of the source process $X$.
- $\mathbf{Y}_n^{(l)}$ is a vector/history of the past $l$ states of the target process $Y$.

The expression of TE in terms of probabilities is as follows:

$$
T_{x \rightarrow y}(k, l, u) = \sum_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}} 
p(y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}) 
\log \left( \frac{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)})} \right).
$$

Where:
- $y_{n+1+u}$ is the next state of $y$ at time $n+1+u$, accounting for a propagation time $u$.
- $\mathbf{y}_n^{(l)}$ is a vector/history of the past $l$ states of the target process $y$.
- $\mathbf{x}_n^{(k)}$ is a vector/history of the past $k$ states of the source process $x$.
- $p(y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability distribution of the next state of $y$, its history, and the history of $x$.
- $p(y_{n+1+u} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of $y_{n+1+u}$ given the histories of $x$ and $y$.
- $p(y_{n+1+u} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of $y_{n+1+u}$ given only the history of $y$.

**Kernel TE estimate** is computed by plugging-in the probabilities estimates in above expression {cite:p}`Schreiber.paper` {cite:p}`articleKantz` {cite:p}`TE_Kernel_Kaiser`.
The probabilities on above expression are estimated by considering the appropriate kernel functions which is explained in detail in the {ref}`Kernel Entropy Estimation`.

```{note}
- This package allows to implement two different kernel functions: box kernel or step kernel.
 ```

## Implementation
The estimator is implemented in the {py:class}`KernelTEEstimator <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
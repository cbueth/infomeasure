---
file_format: mystnb
kernelspec:
  name: python3
---
(kernel_TE)=
# Kernel TE Estimation
The [Transfer Entropy](index.md#transfer_entropy_overview) (TE) from the source process $X(x_n)$ to the target process $Y(y_n)$ in terms of probabilities is written as:

$$
T_{x \rightarrow y}(k, l) = \sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right).
$$

Where:
- $y_{n+1}$ is the next state of $Y$ at time $n$,
- $ \mathbf{y}_n^{(l)} = \{y_n, \dots, y_{n-l+1}\} $ is the embedding vector of $Y$ considering the  $ l $ previous states (history length),
- $ \mathbf{x}_n^{(k)} = \{x_n, \dots, x_{n-k+1}\} $ embedding vector of $X$ considering the $ k $ previous states (history length),
- $p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability of the next state of $Y$, its history, and the history of $X$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of next state of $Y$ given the histories of $X$ and $Y$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$.

``Kernel TE estimation`` estimates the required probability density function (_pdf_)  via **kernel density estimation (KDE)** which provides the probabilities values to be implemented in above formula {cite:p}`Schreiber.paper` {cite:p}`articleKantz` {cite:p}`TE_Kernel_Kaiser`. KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`. For more detail on _pdf_estimation and available kernel functions check the {ref}`Kernel Entropy Estimation` section.

```{note}
- This package allows to implement two different kernel functions: box kernel or step kernel.
 ```

## Implementation
Example usage of Kernel TE estimator...

The estimator is implemented in the {py:class}`KernelTEEstimator <infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

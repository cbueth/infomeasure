---
file_format: mystnb
kernelspec:
  name: python3
---
(symbolic_TE)=
# Symbolic / Permutation TE Estimation
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

``Symbolic MI estimation`` estimates the required probability density function (_pdf_) based on the ordinal structure {cite:p}`Symbolic_TE`, . The details on the _pdf_ estimation {cite:p}`PermutationEntropy2002`based on ordinal structure is provided in {ref}`Symbolic / Permutation Entropy Estimation`.

## Implementation
Usage example of Symbolic TE estimator...


The estimator is implemented in the {py:class}`SymbolicTEEstimator <infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

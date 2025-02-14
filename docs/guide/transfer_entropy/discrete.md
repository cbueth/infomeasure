---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_TE)=
# Discrete TE Estimation
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

``Discrete TE estimation`` estimates the required probability mass function (_pmf_) by counting the occurrences of matching configuration in the dataset by keeping the record of the frequencies. These _pmf_ estimates are then plugged back into the above expression to estimate the TE. This estimator is simple and computationally efficient.

## Implementation
Example usage of Discrete TE estimator...

The estimator is implemented in the {py:class}`DiscreteTEEstimator <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>` class,
which is part of the {py:mod}`im.measures.transfer_entropy <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator
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
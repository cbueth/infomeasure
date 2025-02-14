---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_TE)=
# Kraskov-Stoegbauer-Grassberger TE Estimation
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
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$,
- - $\langle \cdot \rangle$ represents the expectation operator.

**Kraskov-Stoegbauer-Grassberger (KSG) TE estimator** adapts the {ref}`KSG_MI` technique and make it suitable for estimating the TE between source and target variable {cite:p}`article_KSG_TE`. Similar to  MI estimation , it takes an advantage that the  {ref}`entropy_kozachenko_leonenko`  for entropy {cite:p}`kozachenko1987sample` holds for any value of the nearest neighbour $k$ . Therefore, one can vary the value of $k$ in each data point in such a way that the radius (distance) of the corresponding $\epsilon$- balls would be approximately the same for the joint and the marginal spaces. That means the distance is computed in the joint space for the fixed $k$ nearest neighbour, and then it is projected into the marginal spaces. Following the algorithm one, the expression for the TE is as follows:

$$
TE(X \to Y, u) = \psi(k) + \left\langle \psi \left( n_{\mathbf{y}_n^{(l)}} + 1 \right)
- \psi \left( n_{y_{n+1}, \mathbf{y}_n^{(l)}} + 1 \right)
- \psi \left( n_{\mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}} + 1 \right) \right\rangle_n.
$$

where:
- $\psi(\cdot)$ denotes the **_digamma function_**,
- $n_{\mathbf{y}_n^{(l)}}$ is the counts in space $\{ \mathbf{y}_n^{(l)} \}$,
- $n_{y_{n+1}, \mathbf{y}_n^{(l)}}$ is the counts in joint $\{ y_{n+1}, \mathbf{y}_n^{(l)} \}$ space, 
- $n_{\mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}$ is the counts in joint $\{ \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)} \}$ space.


## Implementation
Example usage of KSG TE estimator...


The estimator is implemented in the {py:class}`KSGTEEstimator <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

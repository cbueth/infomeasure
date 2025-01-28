---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_TE)=
# Kraskov-Stoegbauer-Grassberger TE Estimation
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


**Kraskov-Stoegbauer-Grassberger (KSG) TE estimator** technique uses the improved version of  [Kernel estimation of MI](index.md#Kernel_MI) by using the  [KL estimator](index.md#entropy_kozachenko_leonenko) {cite:p}`Symbolic_TE`.  
It relies on the technique of dynamically changing the kernel width for adjusting the density of samples in provided space. 
The basic idea is to estimate the densities in the $X$ and $Y$ spaces using the distances to the $k-th$ nearest neighbors, and then relate these density estimates to mutual information using the **digamma function**.
The methodology is based on the article "Estimating mutual information" {cite:p}`miKSG2004`. 

$$
TE (X \rightarrow Y;u) = \psi(k) + \frac{1}{N} \sum_t \left[ \psi\left( n_{y_{dy_{t-1}}} + 1 \right) - \psi\left( n_{y_t y_{dy_{t-1}}} + 1 \right) - \psi\left( n_{y_{dy_{t-1}} x_{dx_{t-u}}} + 1 \right) \right]
$$

Where:
- $\psi$ is the **digamma function**.
- $k$ is the number of nearest neighbors.
- $n(·)$ are counts of points in respective marginal spaces within the k-th nearest neighbor distance.


## Implementation
The estimator is implemented in the {py:class}`KSGTEEstimator <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
## Usage

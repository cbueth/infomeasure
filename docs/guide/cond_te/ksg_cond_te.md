---
file_format: mystnb
kernelspec:
  name: python3
---
(ksg_cond_te)=
#  KSG Conditional TE Estimation
The Kraskov–Grassberger–Stögbauer (KSG) method was originally developed for estimating MI ({ref}`KSG_MI`) {cite:p}`miKSG2004`.
Ever since it has  been adapted to estimate other information measures such as {ref}`cond_te_overview` (CTE) {cite:p}`cond_te_Ostargard`.  The CTE between two variables $X$ and $Y$ by conditioning the third variable $Z$ is obtained by the following expression:

$$
TE(X \to Y \mid Z) = \psi(k) + \left\langle \psi \left( n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right)
- \psi \left( n_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right)
- \psi \left( n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right) \right\rangle_n
$$

where
- $\psi(\cdot)$ is the **_digamma_** function,
- $k$ is the number of nearest neighbors considered,
- $\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$,
- $\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$,
- $\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$,
- $y_{n+1}$ is the future value of $Y$,
- $n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ are the counts of neighbors in the joint space of past $Y$ and $Z$,
- $n_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ are the counts of neighbors in the joint space including future $Y$ to past of $Y$ and $Z$,
- $n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ are the counts of neighbors in the joint space of past $X$, $Y$, and $Z$,
- $\langle \cdot \rangle_n$ represents the expectation over $n$.


```{code-cell}
import infomeasure as im

x = [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86]
y = [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24]
z = [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85]

im.cte(x, y, cond=z, approach='ksg')
```

The {ref}`Local Conditional TE` is calculated as follows:

```{code-cell}
est = im.estimator(
    x, y, cond=z,
    measure='cte',  # or 'conditional_transfer_entropy'
    approach='ksg'
)
est.local_vals()
```
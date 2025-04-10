---
file_format: mystnb
kernelspec:
  name: python3
---
(ksg_cond_te)=
#  KSG cond_te Estimation
Kraskov–Grassberger–Stögbauer (KSG) method was originally developed for estimating MI ({ref}`KSG_MI`) {cite:p}`miKSG2004`, ever since it has  been adapted to estimate other information measures such as {ref}`cond_te_overview` (CTE) {cite:p}`cond_te_Ostargard`.  The CTE between two variables $X$ and $Y$ by conditioning the third variable $Z$, ie. CTE ($TE(X \to Y \mid Z$) is obtained by following expression:

$$
TE(X \to Y \mid Z) = \psi(k) + \left\langle \psi \left( n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right)
- \psi \left( n_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right)
- \psi \left( n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right) \right\rangle_n
$$

where,
- $\psi(\cdot)$ is the **_digamma_** function,
- $k$ is the number of nearest neighbors considered,
- $\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$,
- $\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$,
- $\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$,
- $y_{n+1}$ is the future value of $Y$,
- $n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is counts in the joint space of past $Y$ and $Z$,
- $n_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is counts in the joint space including future $Y$ to past of $Y$ and $Z$,
- $n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is counts in the joint space of past $X$, $Y$, and $Z$,
- $\langle \cdot \rangle_n$ represents the expectation over $n$.

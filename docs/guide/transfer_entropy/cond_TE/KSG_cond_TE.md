---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_cond_TE)=
#  KSG cond_TE Estimation
Kraskov–Grassberger–Stögbauer (KSG)  was originally developed for estimating MI {cite:p}`miKSG2004`, ever since it has  been adapted to estimate the CMI and CTE 
Explain about condition TE estimation....

$$
TE(X \to Y \mid Z; u) = \psi(k) + \left\langle \psi \left( n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right) 
- \psi \left( n_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right) 
- \psi \left( n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}} + 1 \right) \right\rangle_n.
$$

where:
- $\psi(\cdot)$ is the digamma function.
- $k$ is the number of nearest neighbors considered.
- $\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$.
- $\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$.
- $\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$.
- $y_{n+1+u}$ is the future value of $Y$ at prediction horizon $u$.
- $n_{\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is the number of nearest neighbors in the joint space of past $Y$ and $Z$.
- $n_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is the number of nearest neighbors including future $Y$.
- $n_{\mathbf{x}_n^{(k)}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}}$ is the number of nearest neighbors in the joint space of past $X$, $Y$, and $Z$.
- $\langle \cdot \rangle_n$ represents the expectation over $n$.


Article to cite: cond_TE_Ostargard
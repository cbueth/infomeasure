---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_cond_MI)=
#  KSG cond_MI Estimation
Kraskov–Grassberger–Stögbauer (KSG) method was originally developed for estimating MI ({ref}`KSG_MI`) {cite:p}`miKSG2004`, ever since it has  been adapted to estimate other information measures such as {ref}`cond_MI_overview` (CMI) {cite:p}`CMI_KSG_Frenzel_Pompe`.
The CMI in between two variables $X$ and $Y$ by conditioning the third variable $Z$, ie. CMI $(I(X; Y \mid Z))$ is obtained by following expression:


$$
I(X; Y \mid Z) = \psi(k) + \langle \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1) \rangle
$$

where,
- $k$ is the number of nearest neighbors,
- $ n_x(\cdot) $ refers to the number of neighbors which are with in a hypercube that defines the search range around a statevector,  the size of the hypercube in each of the marginal spaces is defined based on the distance to the $k-th$ nearest neighbor in the highest dimensional space.
- $\psi(\cdot)$ denotes the _digamma function_,
- $\langle \cdot \rangle$ represents the expectation operator.

Similarly, the local conditional MI estimator is:

$$
i(x; y \mid z) = \psi(k) +  \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1)
$$

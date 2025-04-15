---
file_format: mystnb
kernelspec:
  name: python3
---

(ksg_cond_mi)=
#  KSG Conditional MI Estimation
The Kraskov–Grassberger–Stögbauer (KSG) method was originally developed for estimating MI ({ref}`KSG_MI`) {cite:p}`miKSG2004`.
Ever since it has been adapted to estimate other information measures such as {ref}`cond_mi_overview` (CMI) {cite:p}`CMI_KSG_Frenzel_Pompe`.
The CMI in between two variables $X$ and $Y$ by conditioning the third variable $Z$ is obtained by the following expression:


$$
I(X; Y \mid Z) = \psi(k) + \langle \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1) \rangle
$$

where
- $k$ is the number of nearest neighbors,
- $n_x(\cdot)$ refers to the number of neighbors which are with in a hypercube that defines the search range around a statevector. 
  The size of the hypercube in each of the marginal spaces is defined based on the distance to the $k$-th nearest neighbor in the joint space.
- $\psi(\cdot)$ denotes the _digamma function_,
- $\langle \cdot \rangle$ represents the expectation operator.

Similarly, the local conditional MI estimator is:

$$
i(x; y \mid z) = \psi(k) +  \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1)
$$


```{code-cell}
import infomeasure as im

x = [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86]
y = [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24]
z = [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85]

im.cmi(x, y, cond=z, approach='ksg')
```

The {ref}`Local Conditional MI` is calculated as follows:

```{code-cell}
est = im.estimator(
    x, y, cond=z,
    measure='cmi',  # or 'conditional_mutual_information'
    approach='ksg'
)
est.local_vals()
```


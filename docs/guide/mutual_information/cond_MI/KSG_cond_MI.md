---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_cond_MI)=
#  KSG cond_MI Estimation

Explain about the conditional MI estimation...

$$
I(X; Y \mid Z) = \psi(k) + \langle \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1) \rangle.
$$

- $k$ is the number of nearest neighbors,
- $n_z(i)$ is the count of samples in the marginal space $\{z\}$,
- $n_{xz}(i)$ is the count of samples in the joint space $\{x, z\}$,
- $n_{yz}(i)$ is the count of samples in the joint space $\{y, z\}$,
- $\psi(\cdot)$ denotes the _digamma function_,
- $\langle \cdot \rangle$ represents the expectation operator.


local conditional MI estimator is:

$$
i(x; y \mid z) = \psi(k) +  \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1)
$$


{cite:p}`CMI_KSG_Frenzel_Pompe`

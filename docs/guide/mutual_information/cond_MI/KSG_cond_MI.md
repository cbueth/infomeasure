---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_cond_MI)=
#  KSG cond_MI Estimation

Explain about the conditional MI estimation...

$$
I(X; Y \mid Z) = \psi(K) + \langle \psi(\nu_z(i) + 1) - \psi(\nu_{xz}(i) + 1) - \psi(\nu_{yz}(i) + 1) \rangle.
$$

- $K$ is the number of nearest neighbors.
- $\nu_z(i)$ is the count of samples in the marginal space $\{z\}$.
- $\nu_{xz}(i)$ is the count of samples in the joint space $\{x, z\}$.
- $\nu_{yz}(i)$ is the count of samples in the joint space $\{y, z\}$.
- $\psi(\cdot)$ denotes the digamma function.
- $\langle \cdot \rangle$ represents the expectation operator.




{cite:p}`CMI_KSG_Frenzel_Pompe`

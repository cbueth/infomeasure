---
file_format: mystnb
kernelspec:
  name: python3
---

(tsallis_entropy)=
# Tsallis Entropy Estimation

The Tsallis entropy is a generalization of the Shannon entropy that includes a parameter $q$ that allows for a continuous range of entropy measures between the Shannon entropy ($q = 1$) and the Havrda-Charvát entropy ($q = 2$) {cite:p}``.

[//]: # (Tsallis, C. Possible generalization of Boltzmann-Gibbs statistics. J Stat Phys 52, 479–487 (1988). https://doi.org/10.1007/BF01016429)

```bibtex
@article{tsallis1988possible,
  title={Possible generalization of Boltzmann-Gibbs statistics},
  author={Tsallis, Constantino},
  journal={Journal of statistical physics},
  doi = {10.1007/BF01016429},
  volume={52},
  pages={479--487},
  year={1988},
  publisher={Springer}
}
```

_longer summary paragraph with citations_

```{code-cell}
import infomeasure as im
im.entropy([1, 2, 3, 4, 5], approach="tsallis", q=2)  # Havrda-Charvát entropy
```



The estimator is implemented in the {py:class}`TsallisEntropyEstimator <infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

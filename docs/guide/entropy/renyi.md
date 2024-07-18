---
file_format: mystnb
kernelspec:
  name: python3
---

(renyi_entropy)=
# Rényi Entropy Estimation

The Rényi entropy is a generalization of the Shannon entropy that includes a parameter $\alpha$ that controls the sensitivity of the entropy to the probabilities of the different states of the random variable {cite:p}``.

[//]: # ({cite:p}``. Maybe https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181?tab=ArticleFirstPage)

```bibtex
@inproceedings{renyi1961measures,
  title={On measures of entropy and information},
  author={R{\'e}nyi, Alfr{\'e}d},
  booktitle={Proceedings of the fourth Berkeley symposium on mathematical statistics and probability, volume 1: contributions to the theory of statistics},
  volume={4},
  pages={547--562},
  year={1961},
  organization={University of California Press}
}
```

The Rényi entropy is defined as:

$$
H_\alpha(X) = \frac{1}{1 - \alpha} \log \left( \sum_{x \in X} p(x)^\alpha \right)
$$

[//]: # (please check the formula)

where $X$ is the set of possible values of the random variable, $p(x)$ is the probability of the value $x$ occurring, and $\alpha$ is the parameter that controls the sensitivity of the entropy.

```{admonition} Alfréd Rényi (1921–1970)
:class: tip
Alfréd Rényi was a Hungarian mathematician who made significant contributions to probability theory, information theory, and combinatorics. He is known for the Rényi entropy, Rényi divergence, and Rényi's parking constants.

    A mathematician is a device for turning coffee into theorems
    -- Alfréd Rényi

{cite:p}`suzuki2002history`
```

For $\alpha = 1$, the Rényi entropy reduces to the Shannon entropy.
Proof is left as an exercise to the reader 😆. (I think it needs $\sum_{x \in X} p(x) = 1$ to work)

```{code-cell}
import infomeasure as im
im.entropy([1, 2, 3, 4, 5], approach="renyi", alpha=2)
```


The estimator is implemented in the {py:class}`RenyiEntropyEstimator <infomeasure.measures.entropy.renyi.RenyiEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.renyi.RenyiEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

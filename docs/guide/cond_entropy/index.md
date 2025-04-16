---
file_format: mystnb
kernelspec:
  name: python3
---

(cond_entropy_overview)=
# Conditional H
Conditional entropy is defined as the remaining uncertainty of a variable after considering the information from another variable. In other words, it is the remaining unique information of the first variable after having the knowledge of the conditional variable.

Let $X$ and $Y$ be two RVs with the marginal probability as $p(x)$ and $p(y)$, and conditional probability distribution of $X$ conditioned on $Y$ denoted as $p(x|y)$ (or $p(y|x)$ vice versa), then the conditional entropy is calculated by as:

$$
H(X \mid Y) = - \sum_{x,y} p(x, y) \log p(x \mid y)
$$

One can use the chain rule and express the above expression in terms of **Joint Entropy** $H(X,Y)$ and marginal entropy (eg: $H(X)$ and $H(Y)$) as follows:

$$
H(X \mid Y) = H(X,Y) - H(X)
$$


```{sidebar} **Joint Entropy**
The joint entropy represents the amount of information gained by jointly observing two RVs.

$$
H(X, Y) = - \sum_{x,y} p(x, y) \log p(x, y)
$$
```

This package does not offer methods to compute conditional entropy, as both simple {ref}`entropy` and {ref}`joint entropy` are offered, and can be combined.
For further information measures, e.g., mutual information and transfer entropy, this package offers dedicated, probabilistic implementations which minimize bias compared to entropy combinations.

## Local Conditional H
Similar to shannon {ref}`Local Entropy` $h(x)$, one can also define **local or point-wise conditional entropy** $h(x \mid y)$ as follows:

$$
h(x \mid y) = - \log p(x \mid y)
$$
This local conditional entropy also satisfies the chain rule as its average counterparts, hence one can express the local conditional entropy as:

$$
h(x \mid y) = h(x,y) - h(x)
$$

Joint entropy can be accessed from the usual entropy an estimator interface.
To signal that the random variables should be considered jointly, the random variables should be passed as a {py:class}`tuple`
.

```{code-cell}
import infomeasure as im

x = [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
y = [1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0]
h_x = im.entropy(x, approach="discrete")        # x marginal
h_y = im.entropy(y, approach="discrete")        # y marginal
h_xy = im.entropy((x, y), approach="discrete")  # (x, y) joint
h_x, h_y, h_xy
```

The number of RVs is arbitrary:

```{code-cell}
z = [2, 1, 1, 3, 2, 1, 3, 2, 2, 3, 2, 1, 3, 2, 3]
im.entropy((x, y, z), approach="discrete")  # (x, y, z) joint
```

The local values need to, again, be accessed via an estimator class instance.

```{code-cell}
est_xy = im.estimator((x, y), measure="h", approach="discrete")  # (x, y) joint estimator
est_xy.result(), est_xy.local_vals()
```

Joint entropy works for all approaches in {py:func}`im.entropy() <infomeasure.entropy>`.

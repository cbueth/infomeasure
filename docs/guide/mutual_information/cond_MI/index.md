(cond_MI_overview)=
# Conditional MI
The conditional MI is the shared information between two RVs (eg: $X$ & $Y$), with the knowledge of the third conditional variable (eg: $Z$).  

$$
I(X;Y \mid Z) = \sum_{x, y, z} p(x,y,z) \log_2 \frac{p(x \mid y,z)}{p(x \mid z)}
$$
$$
= \sum_{x, y, z} p(x,y,z) \log_2 \frac{p(x,y,z)p(z)}{p(x,z)p(y,z)}
$$
$$
= H(X \mid Z) - H(X \mid Y,Z)
$$

The above expression can be expressed in the form of entropies and joint entropies as follows:

$$
I(X;Y \mid Z) = - H(X,Z,Y) + H(X,Z) + H(Z,Y) - H(Z)
$$

While **estimating conditional MI**, the above formulation has been used to compute the respective entropies and joint entropies from available estimation techniques (detail: {ref}`Types of Estimation techniques available`. Hence, user can choose the desired technique to estimate the conditional MI. However, one has to be careful about the biases arising form the summation/subtraction of estimated entropies as computed in different dimensions. 
The KSG method is known to have reduce such biases, we here have implemented the dedicated formulation to computed the conditional MI via KSG as explained in subsequent section [subsequent section](index.md#KSG_cond_MI).

``Local conditional MI``

Similar to that of local conditional H, **local conditional MI**, as defined by Fano {cite:p}`fano1961transmission`, as follows:

$$
i(x; y \mid z) = \log_b \frac{p(x \mid y, z)}{p(x \mid z)}
$$
$$
= h(x \mid z) - h(x \mid y, z)
$$

Now the conditional MI can be calculated as the expected value of its local counterparts {cite:p}`Lizier2014`, as follows:

$$
I(X; Y \mid Z) = \langle i(x; y \mid z) \rangle.
$$

> Note:
> The conditional MI $I(X;Y \mid Z)$ can be either larger or smaller than its non-conditional counter-part (i.e $I(X;Y )$), this leads to the idea of **Synergy** and **redundancy**(cite)).
> CMI is symmetric under the same condition $Z$, $I(X;Y \mid Z) =  I(Y;X \mid Z)$.

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: KSG method

   KSG_cond_MI
```

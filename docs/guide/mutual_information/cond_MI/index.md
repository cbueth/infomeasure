(cond_MI_overview)=
# Conditional MI
{ref}`Mutual Information <mutual_information_overview>` (MI) in between two processes $X$ and $Y$ can also be conditioned on other processes, such as $Z$, known as conditional MI. Such conditional MI now provides the shared information between  $X$ & $Y$ by considering the knowledge of the third conditional variable (eg: $Z$) and is written as $I(X;Y \mid Z)$.  

$$
I(X;Y \mid Z) = \sum_{x, y, z} p(x,y,z) \log \frac{p(x \mid y,z)}{p(x \mid z)}
$$
$$
= \sum_{x, y, z} p(x,y,z) \log \frac{p(x,y,z)p(z)}{p(x,z)p(y,z)}
$$
$$
= H(X \mid Z) - H(X \mid Y,Z)
$$

#### Local conditional MI
Similar to that of {ref}`Local Conditional H`, **local or point-wise conditional MI**, as defined by Fano {cite:p}`fano1961transmission`, as follows:

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
> The conditional MI $I(X;Y \mid Z)$ can be either larger or smaller than its non-conditional counter-part (i.e $I(X;Y )$), this leads to the idea of **Synergy** and **redundancy** and can be addressed by _information decomposition_ approach (cite)).
> CMI is symmetric under the same condition $Z$, $I(X;Y \mid Z) =  I(Y;X \mid Z)$.

## CMI Estimation
The CMI expression can be expressed in the form of entropies and joint entropies as follows:

$$
I(X;Y \mid Z) = - H(X,Z,Y) + H(X,Z) + H(Z,Y) - H(Z)
$$
 
While `estimating conditional MI`, the above formulation has been used to compute the respective entropies and joint entropies from available estimation techniques (detail: {ref}`Types of Estimation techniques available`. Hence, user can choose the desired technique to estimate the conditional MI, which are:
- Discrete estimation [{ref}`discrete_entropy`]  
- Symbolic estimation [{ref}`symbolic_entropy`]  
- Kernel estimation [{ref}`kernel_entropy`]  

However, one has to be careful about the biases arising form the differing dimensionality of the states spaces across the terms in above equation. The KSG method is known to have reduce such biases, we here have implemented the dedicated formulation to compute the conditional MI via KSG as explained in subsequent section.

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: KSG method

   KSG_cond_MI
```

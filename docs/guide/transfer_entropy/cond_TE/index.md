(cond_TE_overview)=
# Conditional TE
{ref}`Transfer Entropy <transfer_entropy_overview>` (TE) from the source process $X$ to the target process $Y$ can also be conditioned on other possible sources, such as $Z$. In that case, the conditional TE corresponds to the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$, $Z$ and also after considering the past values of target $Y$ itself.
Importantly, the TE can be conditioned on other possible information sources $Z$ , to eliminate their influence from being mistaken as that of the source $Y$.

$$
TE(X \to Y \mid Z) = \sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}} 
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}) 
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)})} \right).
$$

where:
- $p(\cdot)$ represents the probability distribution,
- $\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$,
- $\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$,
- $\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$,
- $y_{n+1}$ is the future state of $Y$.

#### Local Conditional TE
Similar to {ref}`Local Conditional H` and {ref}`Local Conditional MI` measures, we can extract the **local or point-wise conditional transfer entropy** as suggested by _Lizier et al._ {cite:p}`Lizier2014_localinfomeasure`{cite:p}`local_TE_Lizier`.  It is the amount of information transfer attributed to the specific realization $(x_{n+1}, \mathbf{X}_n^{(k)}, \mathbf{Y}_n^{(l)})$ at time step $n+1$; i.e., the amount of information transfer from process $X$ to $Y$ at time step $n+1$:

$$
t_{X \rightarrow Y \mid Z}(n+1, k, l) = \log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n)}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{z}_n)} \right)
$$

The TE as we know can be written as the global average of the local TE:

$$
T_{X \rightarrow Y \mid Z}(k, l) = \langle t_{X \rightarrow Y}(n + 1, k, l) \rangle,
$$

## CTE Estimation 
The CTE expression above can be written as the combination of entropies and joint entropies as follows:

$$
TE(X \to Y \mid Z) = H(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}) 
- H(\mathbf{y}_n^{(l)}, \mathbf{z}_n^{(m)}) 
- H(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}) 
+ H(\mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}, \mathbf{z}_n^{(m)}).
$$

While `estimating conditional TE`, the above formulation has been used to compute the respective entropies and joint entropies from available estimation techniques (detail: {ref}`Types of Estimation techniques available`. Hence, user can choose the desired techniques to estimate the conditional TE, which are:
- Discrete estimation [{ref}`discrete_entropy`]  
- Symbolic estimation [{ref}`symbolic_entropy`]  
- Kernel estimation [{ref}`kernel_entropy`]  

However, one has to be careful about the biases arising form the differing dimensionality of the states spaces across the terms in above equation. The KSG method is known to have reduce such biases, we here have implemented the dedicated formulation to compute the conditional TE via KSG as explained in subsequent section.

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: KSG method

   KSG_cond_TE
 ```

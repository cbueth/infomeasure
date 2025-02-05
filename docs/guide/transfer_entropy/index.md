(transfer_entropy_overview)=
# Transfer Entropy **(TE)**
Transfer entropy (TE) from the source process $X$ to the target process $Y$ is the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$ after considering the past values of target.
It is simply the reduction in the uncertainty in the target variable due to another source variable that is not already explained by the target variables´past. 
Equivalently, TE is the amount of information that a source process provides about the target process´s next state that was not contained in the target´s past states {cite:p}`Schreiber.paper` .

Lets us assume two time series process $X(x)$ and $Y(y)$ as source and target variables then $T_{X \rightarrow Y}$ from source to target is written as:

$$
T_{x \rightarrow y}(k, l, u) = \sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}} 
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}) 
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right).
$$

Where:
- $y_{n+1}$ is the next state of $y$ at time $n+1$, accounting for a propagation time $u$.
- $ \mathbf{y}_n^{(l)} = \{y_n, \dots, y_{n-l+1}\} $ is a past states of  $ Y $, such that it depends on the $ l $ previous values (history).
- $ \mathbf{x}_n^{(k)} = \{x_n, \dots, x_{n-k+1}\} $ is a past states of  $ X $, such that it depends on the $ k $ previous values (history).
- $p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability distribution of the next state of $y$, its history, and the history of $x$.
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of $y_{n+1+u}$ given the histories of $x$ and $y$.
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of $y_{n+1+u}$ given only the history of $y$.

TE is the model-free and directional measure between two processes, making it popular in investigating the dynamical and directional transfer of information in many systems. However, one has to be careful in interpreting the results, as well as in constructing the state space along with the appropriate choice of the length of history (i.e $k$, $l$) to be considered.

We can further add the source to destination time propagation or lag $(u)$ in the expression above and rewrite it as:

$$
T_{x \rightarrow y}(k, l, u) = \sum_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}} 
p(y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}) 
\log \left( \frac{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)})} \right).
$$

TE reduces to well-known Granger causality (upto a factor of 2) for the multivariate gaussian processes (cite). 

``Local Transfer Entropy:``

Similar to entropy and mutual information measures, we can extract the **local transfer entropy** as suggested by Lizier et al. {cite:p}`Lizier2014_localinfomeasure`.  It is the amount of information transfer attributed to the specific realization $(x_{n+1}, \mathbf{X}_n^{(k)}, \mathbf{Y}_n^{(l)})$ at time step $n+1$; i.e., the amount of information transfer from process $X$ to $Y$ at time step $n+1$:

$$
t_{X \rightarrow Y}(n+1, k, l) = \log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right)
$$

The TE as we know can be written as the global average of the local TE:

$$
T_{X \rightarrow Y}(k, l, u) = \langle t_{X \rightarrow Y}(n + 1, k, l) \rangle,
$$

The local TE values can be negative unlike its global counterpart, this means the source is misleading about the prediction of target´s next step. 

> Note:
> - The package allows user to obtain both the local and global (average) values to the TE computation.
> - The package allows user to set the desired propagation time $u$ between the variables. The default value is set to $u=0$. 

``Effective Transfer Entropy:``

The time series data as available from the real word is usually biased due to the finite size effect. 
Depending on the type of estimators implemented the bias can be small or big but it is usually present. 
In order to correct the bias from the finite sample side effect, it is necessary to estimate the expected values of TE estimator for finite data that are close as possible to the original data but doesn´t represent the information transfer.
We can crease such surrogate dataset for TE bias correction which has the same finite length and the same auto-correlation properties to that of original data. 
At the same time, the surrogates should be guaranteed to have no predictive information transfer. This can be achieved by destroying the temporal precedence structure between the source  and the target processes, that would be underlying a potential predictive information transfer in the original data.
We here used a slightly modified TE estimator, called \textit{effective} TE {cite:p}`articleKantz`, defined as the difference between the TE and the one calculated on surrogate (randomly shuffled) data:

$$
eTE = TE(X \rightarrow Y) - TE(X_{\text{shuffled}} \rightarrow Y).
$$


> Note:
> The package has an option to obtain eTE computation.

## List of Estimation Techniques Implemented:

```{eval-rst}
.. toctree::
   :maxdepth: 2
   transfer_entropy/cond_TE/index

.. toctree::
   :maxdepth: 2
   :caption: Discrete RV

   discrete

.. toctree::
   :maxdepth: 2
   :caption: Continuous RV
   
   kernel
   kraskov_stoegbauer_grassberger
   symbolic
   renyi
```

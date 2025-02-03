(transfer_entropy_overview)=
# Transfer Entropy
Transfer entropy (TE) from the source process $X$ to the target process $Y$ is the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$ after considering the past values of target.
It is simply the reduction in the uncertainty in the target variable due to another source variable that is not already explained by the target variables´past. 
Equivalently, TE is the amount of information that a source process provides about the target process´s next state that was not contained in the target´s past states {cite:p}`Schreiber.paper` .
Lets us assume two time series process $X(x)$ and $Y(y)$ as source and target variables.
The transfer entropy (TE) captures the average mutual information from realizations $x_n^{(k)}$ of the state $\mathbf{X}_n^{(k)}$ of a source time-series process $X$ to the corresponding realizations $y_{n+1}$ of the next value $Y_{n+1}$ of the target time-series process $Y$, conditioned on realizations $\mathbf{Y}_n^{(l)}$ of the previous state $\mathbf{Y}_n^{(l)}$:

$$T_{X \rightarrow Y}(k, l) = I \left[ \mathbf{X}_n^{(k)}; Y_{n+1} \mid \mathbf{Y}_n^{(l)} \right].$$

TE is the model-free and directional measure between two variables/processes, making it popular in investigating the directional transfer of information in many systems. 
However, one has to be careful in interpreting the results, as well as in constructing the state space along with the appropriate choice of the length of history to be considered.  

``Local Transfer Entropy:`` 

Similar to entropy and mutual information measures, we can extract the **local transfer entropy** as a local conditional mutual information, which is the amount of information transfer attributed to the specific realization {cite:p}`Lizier2014_localinfomeasure`:
$(x_{n+1}, \mathbf{X}_n^{(k)}, \mathbf{Y}_n^{(l)})$ at time step $n+1$; i.e., the amount of information transfer from process $X$ to $Y$ at time step $n+1$:

$$t_{X \rightarrow Y}(n+1, k, l) = i(\mathbf{x}_n^{(k)}; y_{n+1} \mid \mathbf{Y}_n^{(l)}).$$

We can further add the source to destination time propagation or lag $(u)$ in the expression above and rewrite it as:

$$
t_{X \rightarrow Y}(n + 1, k, l, u) = i(\mathbf{X}_n^{(k)}; y_{n+1+u} \mid \mathbf{Y}_n^{(l)}).
$$

The global TE can be written as the average of the local TE:

$$
T_{X \rightarrow Y}(k, l, u) = \langle t_{X \rightarrow Y}(n + 1, k, l, u) \rangle,
$$

$$
T_{X \rightarrow Y}(k, l, u) = I(\mathbf{X}_n^{(k)}; Y_{n+1+u} \mid \mathbf{Y}_n^{(l)}).
$$

> Note:
> The package allows user to obtain both the local and global (average) values to the MI computation.
> The package allows user to set the desired time-lag between the series. The default value is set to $u=0$, no lag. 

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
> The package also has an option to obtain eTE computation.

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

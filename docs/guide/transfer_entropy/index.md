(transfer_entropy_overview)=
# Transfer Entropy **(TE)**
Transfer entropy (TE) from the source process $X$ to the target process $Y$ is the amount of uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$, after considering the past values of the target $Y$.
It is simply the reduction in the uncertainty in the target variable due to another source variable that is not already explained by the target variables' past.
Equivalently, TE is the amount of information that a source process provides about the target process's next state that was not contained in the target's past states {cite:p}`Schreiber.paper` .

Let us assume two time series process $X(x_n)$ and $Y(y_n)$ as source and target variables then $T_{X \rightarrow Y}$ from source to target is written as:

$$
T_{x \rightarrow y}(k, l) = -\sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right)
$$

where
- $y_{n+1}$ is the next state of $Y$ at time $n$,
- $ \mathbf{y}_n^{(l)} = \{y_n, \dots, y_{n-l+1}\} $ is the embedding vector of $Y$ considering the $l$ previous states (history length),
- $ \mathbf{x}_n^{(k)} = \{x_n, \dots, x_{n-k+1}\} $ embedding vector of $X$ considering the $ k $ previous states (history length),
- $p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability of the next state of $Y$, its history, and the history of $X$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of next state of $Y$ given the histories of $X$ and $Y$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$.

TE is the model-free and directional measure between two processes, making it popular in investigating the dynamical and directional transfer of information in many systems. However, one has to be careful in interpreting the results, as well as in constructing the state space along with the appropriate choice of the length of history (i.e., $k$, $l$) to be considered.

We can further add the source to destination time propagation or lag $u$ in the expression above, and rewrite it as:

$$
T_{x \rightarrow y}(k, l, u) = -\sum_{y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}
p(y_{n+1+u}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})
\log \left( \frac{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1+u} \mid \mathbf{y}_n^{(l)})} \right).
$$

## Local Transfer Entropy
Similar to {ref}`Local Entropy` and {ref}`Local Mutual Information`, we can extract the **local or point-wise transfer entropy** as suggested by Lizier _et al._ {cite:p}`Lizier2014,local_TE_Lizier`.  It is the amount of information transfer attributed to the specific realization $(x_{n+1}, \mathbf{X}_n^{(k)}, \mathbf{Y}_n^{(l)})$ at time step $n+1$; i.e., the amount of information transfer from process $X$ to $Y$ at time step $n+1$:

$$
t_{X \rightarrow Y}(n+1, k, l) = -\log \left(\frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right)
$$

The TE as we know can be written as the global average of the local TE:

$$
T_{X \rightarrow Y}(k, l, u) = \langle t_{X \rightarrow Y}(n + 1, k, l) \rangle
$$

The local TE values can be negative, unlike its global counterpart; this means the source is misleading about the prediction of target's next step.

```{note}
- The package allows user to obtain both the {ref}`Local Values` and {ref}`Global value` of the TE computation.
- The package allows user to set the desired propagation time $u$ between the variables.
  The default value is set to $u=0$.
```

(effective_te)=
## Effective Transfer Entropy (eTE)

Real world time series data are usually biased due to the finite size effect.
Depending on the type of estimators implemented, the bias can be small or big, but it is usually present.
In order to correct the bias from the finite sample size effect, it is necessary to estimate the expected values of the TE estimator for finite data that is as close as possible to the original data but doesn't represent the information transfer.
We can crease such a surrogate dataset for TE bias correction which has the same finite length and the same auto-correlation properties to that of the original data.
 At the same time, the surrogates should be guaranteed to have no predictive information transfer.
This can be achieved by destroying the temporal precedence structure between the source and the target processes.
Effective TE is defined as the difference between the original TE and TE calculated on surrogate (randomly shuffled) data {cite:p}`articleKantz`:

$$
\operatorname{eTE} = \operatorname{T}_{X \rightarrow Y} - \operatorname{T}_{X_{\text{shuffled}} \rightarrow Y}
$$

Directly shuffling the entire source series might not be the valid approach as it destroys the source history vector $\mathbf{x}_n^{(k)}$ for $k > 1$.
Instead, we shuffle the $\mathbf{x}_n^{(k)}$ vector itself, and compute the surrogate TE, subsequently the eTE {cite:p}`lizierJIDTInformationTheoreticToolkit2014`.

```{tip}
The package has an option to obtain eTE computation, see {ref}`Effective value`.
```


## List of Estimation Techniques Implemented:

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Discrete RV

   discrete

.. toctree::
   :maxdepth: 1
   :caption: Continuous RV

   kernel
   kraskov_stoegbauer_grassberger
   ordinal
   renyi_tsallis
```

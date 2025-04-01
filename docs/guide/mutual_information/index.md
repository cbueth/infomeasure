(mutual_information_overview)=
# Mutual Information **(MI)**
Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$. In another word, MI measures the average reduction in uncertainty about $X$ that results from learning the value of $Y$, or vice versa {cite:p}`cover2012elements`. Mathematically MI is expressed as:

$$
I(X;Y) = \sum_{x, y} p(x, y) \log \frac{p(x,y)}{p(x) p(y)}
$$
where,
- $p(x,y)$ is the joint probability distribution for the occurrence of joint state $(x,y)$,
- $p(x)$ and $p(y)$ is the marginal probability distribution of $X$ and $Y$ respectively.

MI allows to detect the both liner and non-liner relationships between the variables hence making it very popular in the investigating of systems showing complex behaviours. MI can also be thought as the measure of mutual dependence between the random variables, with $I(X; Y) = 0$ signifying the in-dependencies between the $X$ and $Y$.

#### Local Mutual Information
Similar to {ref}`Local Entropy`, one can build the **local or point-wise mutual information**  directly from its average counterparts by aligning with the average definition of MI {cite:p}`Lizier2014` {cite:p}`manning1999foundations` {cite:p}`fano1961transmission` .
The local MI values can be either positive or negative , in contrast to the local entropy which cannot take negative values. One can interpret the negative local MI value (i.e$i(x: y)=-ve$ ) as knowledge of the event $y$ increasing the uncertainty about $x$, being _misinformative_.
However, these local MI will always average to the non-negative global MI value {cite:p}`Lizier2014_localinfomeasure`.
The local MI (or shared information content) between the two events $x$ and $y$, also known as **point-wise mutual information** is given as:

 $$
   i(x; y) = \log_b \left( \frac{p(x, y)}{p(x) p(y)} \right)
  $$

The MI can be expressed as the average of local MI:

$$
I(X; Y) = \langle i(x: y) \rangle.
$$

> Note:
> The package allows user to obtain both the local and global (average) values to the MI computation.

#### Time-lagged Mutual Information
If the RV are time series then one can implement the time-lagged MI in between the $X$ and $Y$ time series.

$$
I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log \frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}
$$
where,
- $p(x_t,y_t)$: the joint probability distribution at time $t$,
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities of $X_t$ and $Y_t$ respectively,
- $u$: the propagation time or the lag between two time series.

> Note:
> The package allows user to set the desired time-lag between the series. The default value is set to $u=0$, no lag.

## MI estimation
When estimating MI, several factors must be considered (ref: {ref}`Estimation`). First, identify whether the dataset is discrete or continuous. Then, select an appropriate estimator, which can be broadly categorized into parametric and non-parametric techniques. This package provides methods for both discrete and continuous random variables, and the non-parametric techniques, with detailed explanations and implementation guidelines available in the subsequent pages.

### List of MI Estimation Techniques Implemented:

```{eval-rst}
.. toctree::
   :maxdepth: 2
   mutual_information/cond_MI/index

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

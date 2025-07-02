(entropy_overview)=
# Entropy **(H)**

Entropy is the amount of uncertainty associated with a random variable (RV).
On the flip side, this uncertainty is nothing but the lack of information.
The larger the information required to accurately predict the state of RV, the higher is the uncertainty we initially had about it; hence information and uncertainty can be seen as two sides of the same coin.
Thus, the information/uncertainty $H(X)$ associated with a RV $X$ is a quantification of the possibility of making predictions about the occurrence of the next state beyond chance {cite:p}`cover2012elements`.

## Information Measures
In this python package, we will deal with three different measures of entropy, as follows:
### Shannon Entropy
**Hartley** (1928) devised a formula to compute the amount of information associated with an unknown $x$ of which we know nothing except it belongs to the set having $N$ elements.
The information he proposed was given by Hartley's formula as follows {cite:p}`Hartley1928TransmissionOI`:

$$
H(x) = log_2(N).
$$

It was only 20 years later, in 1948, that **Claude Shannon** in his seminal paper "_Mathematical Theory of Communication_" {cite:p}`shannonMathematicalTheoryCommunication1948` developed a mathematical measure, defined as entropy, to quantify the amount of information $H(X)$ produced by a source variable $X$.

$$
H(X) = -\sum_{x \in X} p(x) \log_b p(x),
$$

where

```{sidebar} Continuous Variable
For a continuous random variable $X$ with the probability distribution function $p(x)$, the differential entropy is written as:

$$
H(X) = -\int_{X} p(x) \log_b p(x) \, dx
$$
The differential entropy is closely related to the Shannon entropy {cite:p}`cover2012elements`.
```

- $X$: The set of possible values of the random variable.
- $p(x)$: The probability of the value $x$ occurring.
- $b$: The base of the logarithm.
  - If $b = 2$, the unit of information is "bit."
  - If $b = e$, the unit of information is "nat."

The shannon formulation is generic.
In the case of equiprobable outcomes, the special case of Hartley emerges.
Thus, the Shannon entropy quantifies the average amount of information we expect to gain when observing specific outcomes or equivalently the average decrease in uncertainty about the possible values of an RV.
Shannon's motivation for using his mathematical formalism (entropy) was to determine whether the data stream can be encoded in such a way that even after it has been sent through the channel noisy enough to corrupt the data during the transmission,
the original data stream can be reconstructed in an error-free way at the receiver end. The interested reader can read his paper {cite:p}`shannonMathematicalTheoryCommunication1948` for his findings, but here we would like to state another way to understand the Shannon entropy formulation in terms of messages associated with the RV (source).
It is the measure of the amount of information in a message expressed in binary digits needed to express the message using the most appropriate way to code to get the shortest sequence.

```{note}
The base of the algorithm in the entropy formula only changes the value of the entropy by a multiplicative constant, hence using one form to another is only a matter of convenience.
Our package supports using any information unit.
Default is the natural unit (nats).
If you want to use bits or another base, find the {ref}`package configuration`.
```

#### Local Entropy
The **local information** measure, also referred to as a **point-wise** information-theoretic measure, characterizes the local information associated with the individual value points i.e. $x$ (at each time step or observation) rather than the average information associated with the variables $X$ {cite:p}`Lizier2014`.
Applied to time series data, the local information measure can uncover dynamic structures that averaged measures overlook, as it characterizes the information attributed at each local point in time.
The **local entropy** of an outcome $x$ of measurement of the variable $X$ is given by:

$$
h(x) = -\log_b p(x).
$$

$h(x)$ is the information contain attributed to the individual measurement $x$.
Practically, $H(X)$ is the **average** or **expectation value** of the local information content for each outcome $x$ in $X$:

$$
H(X) = \langle h(x) \rangle.
$$

```{note}
- The package allows user to obtain both the local and global (average) values to the Entropy computation. How to access: {ref}`Local Values`.
- A lower-case symbol is used to denote local information-theoretic measures in this documentation.
```

As successful as Shannon’s information theory has been, with time, it was clear that it is capable of dealing only with a limited class of systems one might hope to address in statistical physics.
There was a growing interest to look for a more general form of information measure applicable to diverse complex systems, such as stock market returns, protein folding, percolation, etc.
The additivity of independent mean information was the most natural axiom to attack, see {cite:t}`khinchin1957mathematical` on axiomatic derivation of Shannon’s entropy.
Basically, the axiom states that: if $P$ and $Q$ are discrete generalized probability distributions of two independent random variables, then $H[P Q] = H[P]+H[Q]$.
On this level, two modes of reasoning can be formulated. One may either keep the additivity of independent information but utilise more general definition of means, or keep the usual definition of linear means but generalize the additivity law, leading us to two generalized class of entropies: **Rényi entropy**, **Tsallis Entropy**.

(renyi-alpha-entropy)=
### Renyi $\alpha$-Entropy
Alfréd Rényi (mid-60's) derived a generalized family of one-parameter entropy as an exponentially weighted mean of unexpectedness functional (i.e $-log(p)$) known as **Renyi $\alpha$-Entropy**, for detail check {cite:p}`renyi1976selected,jizbaInformationTheoryGeneralized2004`.
Rényi $\alpha$-Entropy is the most general class of information measure preserving the additivity for independent systems and compatible with Kolmogorov's probability axioms {cite:p}`kolmogoroff1933`.

Let $\alpha > 0$, and $\xi_n$ be a random variable with probability distribution $\mathcal{P} = (p_1, ..., p_n)$, where $\sum_{i=1}^{n} p_i = 1$ and $H_\alpha[\mathcal{P}]$ is such that:

$$
H_\alpha[\mathcal{P}] := \frac{1}{1-\alpha} \log_2 \left( \sum_{i=1}^{n} p_i^\alpha \right).
$$
Where,
- $\alpha$: A positive real number greater than 0.
- $\xi_n$: A random variable.
- $\mathcal{P}$: A probability distribution, given by $\mathcal{P} = (p_1, ..., p_n)$.
- $p_i$: The probability associated with the $i$-th outcome.
- $H_\alpha[\mathcal{P}]$: Rényi entropy of the probability distribution $\mathcal{P}$.

Its distinguishing property is that the small values of probabilities $p_i$ are emphasized for $\alpha < 1$ and, on the contrary, larger probabilities are emphasized for $\alpha > 1$. For $\alpha = 1$ Rényi entropy reduces to Shannon Entropy.
Rényi $\alpha$-Entropy class can be in particular interesting for the system where additivity (in Shannon sense) is, however, not always preserved, especially in nonlinear complex systems, e.g., when we have to deal with long range forces.

(tsallis-q-entropy)=
### Tsallis Entropy
**Tsallis Entropy (q-order entropy)**, or Havrda and Charvát entropy, is another possible generalization of information measure by modifying the additivity law.
From the infinite possible generalizations, the so-called q–additivity prescription has been used on q–calculus to formalize the entire approach in a unified manner.
For details, refer to {cite:p}`articleTsallis,Tsallis1999,Tsallis1998`.
The corresponding measure of information is called Tsallis, non–extensive, or q-order entropy.

The Tsallis entropy $S_q$ for a probability distribution $\mathcal{P} = (p_1, ..., p_n)$ is defined as:

$$
S_q = \frac{1}{1 - q} \left[ \sum_{k=1}^{n} (p_k)^q - 1 \right],
$$
where:
- $\mathcal{P}$: A probability distribution, given by $\mathcal{P} = (p_1, ..., p_n)$.
- $p_k$: The probability associated with the $k$-th outcome.
- $q$: A real number greater than 0, which is a parameter of the entropy.
- $S_q$: Tsallis entropy of the probability distribution $\mathcal{P}$.

In the $q \to 1$ limit, the Jackson sum (q-additivity) reduces to ordinary summation, and the Tallis entropy reduces to Shannon Entropy.
This class of entropy measure is in particularly useful in the study in connection with long–range correlated systems and with non–equilibrium phenomena.

## Entropy Estimation
   When estimating entropy, several factors must be considered (see {ref}`Estimation`). First, identify whether the dataset is discrete or continuous. Then, select an appropriate estimator, which can be broadly categorized into parametric and non-parametric techniques. This package provides methods for both discrete and continuous random variables, and the non-parametric techniques, with detailed explanations and implementation guidelines available in the subsequent pages.

```{note}
- This package consists of non-parametric estimation techniques for both the discrete and continuous RV.
- All the estimators are based on Shannon information, except explicitly mentioned: Rényi entropy estimator & Tsallis entropy estimator
```

### List of Entropy Estimators

The infomeasure package provides a comprehensive suite of entropy estimators for both discrete and continuous random variables. For guidance on selecting the appropriate estimator for your data, see the {ref}`estimator_selection_guide`.

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Discrete RV

   discrete

.. toctree::
   :maxdepth: 1
   :caption: Continuous RV

   kernel
   kozachenko_leonenko
   ordinal
   renyi
   tsallis
```

#### Available Discrete Estimators

The package includes the following discrete entropy estimators:

- **Basic Estimators**: Discrete (MLE), Miller-Madow
- **Bias-Corrected**: Grassberger, Shrinkage (James-Stein)
- **Coverage-Based**: Chao-Shen, Chao-Wang-Jost
- **Bayesian**: Bayesian (with multiple priors), NSB, ANSB
- **Specialized**: Zhang, Bonachela

Each estimator has different strengths and is suitable for different data characteristics and sample sizes. The {ref}`discrete_entropy` page provides detailed information about all available estimators with examples and usage guidance.

#### Available Continuous Estimators

For continuous data, the package provides:

- **Kernel-based**: Kernel density estimation
- **Nearest-neighbor**: Kozachenko-Leonenko (KL)
- **Ordinal patterns**: For time series analysis
- **Generalized measures**: Rényi α-entropy, Tsallis q-entropy

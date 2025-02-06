(cond_entropy_overview)=
# Conditional H
Conditional entropy is defined as the remaining uncertainty of a variable after considering the information from another variable. In another word, it is the remaining unique information in the first variable after having the knowledge of the conditional variable.
Let $X$ and $Y$ be two RVs on some probability space with the marginal probability as $p(x)$ and $p(y)$ and conditional probability distribution of $X$ conditioned on $Y$, i.e. $p(x|y)$ (or $p(y|x)$ for vice-versa situation), then the conditional entropy is calculated by as:

$$
H(X \mid Y) = - \sum_{x,y} p(x, y) \log p(x \mid y) 
$$

One can use the chain rule and express the above expression in terms of **Joint Entropy** $(H(X,Y))$ and marginal entropy (eg: $H(X)$ and $H(Y)$) as follows:

$$
H(X \mid Y) = H(X,Y) - H(X) 
$$


````{sidebar} **Joint Entropy**
The joint entropy represents the amount of information gained by observing jointly two RVs.

$$
H(X, Y) = - \sum_{x,y} p(x, y) \log p(x, y)
$$
````
We will use this formulation to compute the conditional entropy in our package. The joint and marginal entropy are computed using all available estimation methods, meaning that the selected estimation method for conditional entropy corresponds to the one used for computing the respective joint and marginal entropy

#### Local Conditional H
Similar to shannon {ref}`Local Entropy` $h(x)$, one can also define **local or point-wise conditional entropy** $h(x \mid y)$ as follows:

$$
h(x \mid y) = - \log p(x \mid y)
$$
This local conditional entropy also satisfies the chain rule as its average counterparts, hence one can express the local conditional entropy as:

$$
h(x \mid y) = h(x,y) - h(x) 
$$

## Implementation

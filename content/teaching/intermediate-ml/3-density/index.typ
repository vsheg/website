#import "../template.typ": *
#show: template

= Normal distribution

== Univariate
A random variable $xi$ is said to have a normal distribution with mean $mu$ and variance
$sigma^2$ if its probability density function (pdf) is given by

$
  f_xi (x) = 1 / (sigma sqrt(2 pi)) exp{ -1 / 2 ((x - mu) / sigma)^2 }
$

where $mu$ is the mean and $sigma^2$ is the variance of the distribution. More compactly,
it can be written as

$
  xi tilde cal(N)(mu, sigma^2)
$

== Uncorrelated multivariate
A random vector $bold(xi) = vec(xi_1, dots.v, xi_k)$ is said to have an uncorrelated
multivariate normal distribution with mean $bold(mu) = vec(mu_1, dots.v, mu_k)$ and
variances $sigma_1^2, dots, sigma_k^2$ if the pdf of every random component of $bold(xi)$
is given by

$
  f_(xi_j) (x) = 1 / (sigma_j sqrt(2 pi)) exp{ -1 / 2 ((x - mu_j) / sigma_j)^2 }
$

where $mu_j$ is the mean and $sigma_j^2$ is the variance of the $j$-th component.
#margin[
  The uncorrelated multivariate normal distribution is a special case of the general multivariate normal. When components are uncorrelated, the covariance matrix is diagonal, which simplifies many calculations.
]

All components of $bold(xi)$ are assumed to be independent, so the joint pdf of $bold(xi)$ is
the product of the pdfs of its components:

$
  f_bold(xi) (x_1, ..., x_k) &= product_(i=1)^k f_(xi_i) (x_i) \
  &= product_(i=1)^k 1 / (sigma_i sqrt(2 pi)) exp{ -1 / 2 ((x_i - mu_i) / sigma_i)^2 }
$

== Covariance matrix
All variance parameters $sigma_1^2, dots, sigma_k^2$ can be combined into a covariance
matrix $Sigma$. The covariance matrix is a symmetric positive definite matrix that
describes the covariance between the components of $bold(xi)$.

$
  Sigma = dmat(sigma_1^2, dots.down, sigma_k^2)
$

Here, the covariance matrix is diagonal (all off-diagonal elements are zero), because we
assumed that the components of $bold(xi)$
are uncorrelated, i.e., $Cov[xi_i, xi_j] = 0$ for all $i != j$.

The pdf of the multivariate normal distribution can be written in terms of the covariance
matrix:

$
  f_bold(xi) (x_1, ..., x_k) = exp{ -1 / 2 (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) } / sqrt((2 pi)^k det Sigma)
$ <multivariate-normal-distribution>

#margin[
  For a sample $X = {x_1, ..., x_ell} subset RR$, the variance is the average of the squared
  differences from the mean:
  $ Var[X] := 1 / ell sum_(i=1)^ell (x_i - macron(x))^2. $

  Given another sample $Y = {y_1, ..., y_ell} subset RR$, the _co_-variance between two
  samples is characterized by how much they vary together:
  $
    Cov[X, Y] := 1 / ell sum_(i=1)^ell (x_i - macron(x)) dot (y_i - macron(y)).
  $

  Both per sample variance and two samples covariance can be combined into a covariance
  matrix.
  $
    Sigma = mat(Cov[x, x], Cov[x, y]; Cov[y, x], Cov[y, y]) = mat(Var[x], Cov[x, y]; Cov[y, x], Var[y]).
  $
  It will be shown below that this is equivalent to the covariance matrix for a sample of 2D
  vectors $bold(v)_i = vec(x_i, y_i) in RR^2$.
]

The covariance matrix $Sigma$ above is a diagonal matrix, but in general, it's a symmetric
positive definite matrix that describes the covariance between the components of $bold(xi)$:

$
  Sigma := mat(
    Cov[xi_1, xi_1], dots, Cov[xi_1, xi_k]; dots.v, dots.down, dots.v; Cov[xi_k, xi_1], dots, Cov[xi_k, xi_k]
  ).
$

If we substitute the non-diagonal covariance matrix $Sigma$ into the pdf, we get the
general form of the multivariate normal distribution.
#margin[
  To characterize _co_-variance of multiple samples
  $
    X_1 & = {x_(1,1), ..., x_(1,ell)}, quad
    ..., quad
    X_k & = {x_(k,1), ..., x_(k,ell)}
  $
  all together, we combine them into one sample of $k$-dimensional data:
  $ V = {bold(v)_1, ..., bold(v)_ell}, quad bold(v)_i = vec(x_(1,i), dots.v, x_(k,i)). $

  The covariance between any two samples $X_t$ and $X_q$ is
  $ Cov[X_t, X_q] := 1 / ell sum_(i=1)^ell (x_(t,i) - macron(x)_t) dot (x_(q,i) - macron(x)_q). $

  Generally, for a sample of vectors $bold(v)_1, ..., bold(v)_ell in RR^k$:
  $
    Sigma :&= 1 / ell sum_(i=1)^ell (bold(v)_i - macron(bold(v))) (bold(v)_i - macron(bold(v)))^Tr \
    &= 1 / ell sum_(i=1)^ell Cov[bold(v)_i, bold(v)_i] \
    &= Ex[(bold(v) - macron(bold(v))) (bold(v) - macron(bold(v)))^Tr].
  $
  which resembles the variance but in multiple dimensions.
]

Technically, each component of $Sigma$ is the covariance between the corresponding
components
$ Sigma_(i,j) := Cov[xi_i, xi_j] = Ex[(xi_i - mu_i) (xi_j - mu_j)]. $

The term $det Sigma$ is the generalized variance.

== Mahalanobis distance
The distance between a point $bold(x)$ and the distribution $cal(N)(bold(mu), Sigma)$ can
be measured using the Mahalanobis distance.
#margin[
  Quadratic form $Q(bold(x))$ is a scalar function of a vector $bold(x)$ that can be
  expressed as a weighted sum of the squares of the components of $bold(x)$:

  $ Q(bold(x)) = sum_(i,j) w_(i,j) x_i x_j. $

  These weights can be gathered into a matrix $W$, and the quadratic form can be written as
  a matrix product:

  $ Q(bold(x)) = bold(x)^Tr W bold(x). $
]

The premise is that the covariance matrix $Sigma$ captures the correlations between the
components of $bold(xi)$. The Mahalanobis distance is a measure of how many standard
deviations away a point $bold(x)$ is from the mean $bold(mu)$, taking into account the
correlations between the components of $bold(xi)$.

We can define a quadratic form
$
  Q(bold(x)) :&= (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) \
  &= sum_(i,j) (x_i - mu_i) (Sigma^(-1))_(i,j) (x_j - mu_j).
$

The square root of this quadratic form $sqrt(Q(bold(x)))$ is the Mahalanobis distance between a point $bold(x)$ and the distribution $cal(N)(bold(mu), Sigma)$.

= Multivariate Normal Distribution

== General form
The probability density function of the multivariate normal distribution is given by:

$
  f_bold(xi) (x_1, ..., x_k) := exp{ -1 / 2 (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) } / sqrt((2 pi)^k det Sigma)
$

where:
- $bold(xi) = vec(xi_1, dots.v, xi_k)$ is the random vector
- $bold(mu) := Ex bold(xi) = vec(Ex xi_1, dots.v, Ex xi_k)$ is the mean vector
- $Sigma_(i,j) := Cov[xi_i, xi_j] = Ex[(xi_i - mu_i) (xi_j - mu_j)]$ is the covariance matrix (symmetric positive definite)
- $det Sigma$ is the generalized variance
- $Q(bold(x)) = (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu))$ is a quadratic form
- $sqrt(Q(bold(x)))$ is the Mahalanobis distance between a point $bold(x)$ and the distribution $cal(N)(bold(mu), Sigma)$

In compact notation, this is written as:
$bold(xi) tilde cal(N)(bold(mu), Sigma)$

== Uncorrelated components
When the components of the distribution are uncorrelated:
$Cor[xi_i, xi_j] = 0 quad forall i != j$

This has geometric implications - the axes of the probability density ellipsoid are parallel to the coordinate axes.

The covariance matrix simplifies to a diagonal matrix:
$Sigma = diag(sigma_1^2, ..., sigma_k^2)$

For uncorrelated components, the multivariate normal pdf can be expressed as a product of univariate normal pdfs:
$
  f_bold(xi) (x_1, ..., x_k) :&= exp{ -1 / 2 (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) } / sqrt((2 pi)^k det Sigma) \
  &= product_(i=1)^k exp{-1 / 2 ((x_i - mu_i) / sigma_i)^2} / (sigma_i sqrt(2 pi))
$

This factorization enables simpler parameter estimation methods.

== Decorrelation transformation
#margin[
  Decorrelation is a critical preprocessing step in many machine learning applications. It simplifies the data by removing linear dependencies between features.
]

For correlated components, we need to find a transformation that makes the components uncorrelated. When components of a multivariate normal distribution are correlated, the covariance matrix is not diagonal.

To decorrelate the components:

1. Apply spectral decomposition to the covariance matrix (a special case of SVD for symmetric matrices):
$Sigma = V S V^Tr, quad S = diag(lambda_1^2, ..., lambda_k^2)$

2. The quadratic form can be rewritten as:
$
  Q = (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu))
  &= (bold(x) - bold(mu))^Tr (V S V^Tr)^(-1) (bold(x) - bold(mu)) \
  &= (bold(x) - bold(mu))^Tr V^(-T) S^(-1) V^(-1) (bold(x) - bold(mu)) \
  &= (bold(x) - bold(mu))^Tr V S^(-1) V^Tr (bold(x) - bold(mu)) \
  &= (V^Tr (bold(x) - bold(mu)))^Tr S^(-1) (V^Tr (bold(x) - bold(mu)))
$

3. Define the decorrelation transformation:
$bold(x)' := V^Tr bold(x)$

4. The transformed parameters are:
$
  bold(mu)' &= V^Tr bold(mu) \
  Sigma' &= S
$

5. To transform parameters back to the original space:
$
  bold(mu) &= V bold(mu)' \
  Sigma &= V S V^Tr
$

#margin[
  For orthogonal (rotation) matrices:
  $V^(-1) = V^Tr, quad V^(-T) = V$
]

= Parameter Estimation for Normal Distribution

== Maximum likelihood estimation
When a sample is generated from a Gaussian distribution, we can estimate its parameters using maximum likelihood estimation:

$
  pdv(, mu) ln L(bold(mu), Sigma | X^ell) = 0 => hat(bold(mu)) = 1 / ell sum_(bold(x) in X^ell) bold(x) \
  pdv(, Sigma) ln L(bold(mu), Sigma | X^ell) = 0 => hat(Sigma) = 1 / ell sum_(bold(x) in X^ell) (bold(x) - hat(bold(mu)))(bold(x) - hat(bold(mu)))^Tr
$

#margin[
  The MLE estimates are asymptotically unbiased and efficient, making them optimal for large samples. For smaller samples, however, the covariance estimate is biased.
]

== Multicollinearity issues
#margin[
  Multicollinearity is a statistical phenomenon where two or more predictor variables in a model are highly correlated. This creates redundant information that doesn't contribute uniquely to the model's explanatory power.
]

When estimating parameters from data, multicollinearity can cause problems:

- The sample covariance matrix $hat(Sigma)$ is constructed from $ell$ rank-1 matrices, so $rank hat(Sigma) <= ell$
- If the number of features exceeds the sample size, $hat(Sigma)$ will be singular ($det = 0$)
- A singular covariance matrix cannot be inverted, making it impossible to evaluate the density function

Solutions to multicollinearity include:
1. Reducing the number of features through feature selection methods (PCA, SFS, etc.)
2. Increasing the sample size
3. Adding regularization to the covariance matrix:
$hat(Sigma)' <- hat(Sigma) + tau I$
4. Assuming uncorrelated features (diagonal covariance matrix)

= Non-parametric Density Estimation

== Parametric vs non-parametric approaches
#margin[
  The choice between parametric and non-parametric methods involves a bias-variance tradeoff. Parametric methods generally have higher bias but lower variance, while non-parametric methods have lower bias but higher variance.
]

**Parametric methods** for density estimation assume a specific distribution shape (e.g., normal), which simplifies calculations and requires less data. However, they can be inaccurate if the actual data distribution differs significantly from the assumed form and are limited in capturing complex structures.

**Non-parametric methods** offer greater flexibility and can provide more accurate estimates for complex and multimodal distributions without assuming a specific form. However, they typically require larger data sets, are more computationally intensive, and their results are harder to interpret due to the lack of explicit parameters.

== Empirical density estimator
The simplest non-parametric estimator for a probability density function is:

$
  hat(f)(bold(x)) := 1 / ell sum_(bold(x)' in X^ell) Ind(bold(x) = bold(x)'),
$

and for the cumulative distribution function:

$
  hat(F)(bold(x)) := 1 / ell sum_(bold(x)' in X^ell) Ind(x'_1 <= x_1) dots.c Ind(x'_k <= x_k)
$

where:
- $f$ is a pdf/pmf
- $F$ is a cdf
- $bold(x)' = (x'_1, dots, x'_k)$ is a point from the sample $X^ell$
- $bold(x) = (x_1, dots, x_k)$ is a new point

== Histogram density estimator
A more practical approach divides the feature space into bins:

$
  hat(f)(bold(x)) := 1 / ell dot \# (B(bold(x)) sect X^ell),
$

and for the CDF:

$
  hat(F)(bold(x)) := 1 / ell sum_B Ind(B <= B(bold(x))) dot \# B
$

where:
- $X^ell subset.eq union.sq.big_B B$ is the partition into bins
- $\# B$ is the bin size
- $B(bold(x))$ is the specific bin containing $bold(x)$
- $B <= B'$ iff $and.big_(j=1)^k sup B_j <= sup B'_j$ allows ordering of bins
- $n_j$ is the number of bins for the $j$th feature
- $h_j := (f_j^max - f_j^min) / n$ is the corresponding bin width

== Window averaging
#margin[
  The choice of window width $h$ is critical in kernel density estimation and presents the classic bias-variance tradeoff.
]

Instead of bins, we can use a window function centered at each data point:

$
  hat(f)(bold(x)) := 1 / ell dot 1 / (2h) sum_(bold(x)' in X^ell) Ind(norm(bold(x) - bold(x)') / h < 1),
$

and for the CDF:

$
  hat(F)(bold(x)):= 1 / ell sum_(bold(x)' in X^ell) Ind(bold(x)' <= bold(x) plus.circle h)
$

where:
- $h$ is the window width (radius)
- $bold(x)' <= bold(x)$ means all components are less or equal
- $plus.circle$ means componentwise addition
- $K(r) = 1 / 2 Ind(|r| < 1)$ is a kernel function

= Kernel Density Estimation

== Parzen-Rosenblatt window method
#margin[
  Kernel density estimation is often called the Parzen-Rosenblatt window method after its developers Emanuel Parzen and Murray Rosenblatt.
]

There are two main approaches for multivariate density estimation:

1. Product kernel approach (assumes local independence):
$
  hat(f)(bold(x)) := 1 / ell sum_(bold(x)' in X^ell) product_(j=1)^k 1 / h_j dot K((bold(x)^j - bold(x)'^j) / h_j)
$

2. Multivariate kernel approach:
$
  hat(f)(bold(x)) := 1 / ell dot 1 / V(h) sum_(bold(x)' in X^ell) K( rho(bold(x), bold(x)') / h )
$

where:
- $K(r)$ is a kernel function satisfying:
  - $integral K(r) dd(r) = 1$
  - $K(r) > 0$
  - For all $r > 0$: $K(r) arrow.br$ (non-increasing)
- $V(h) := integral_(-oo)^(+oo) K( rho / h ) dd(rho)$

== Kernel functions
Several standard kernel functions are used in practice:
- Epanechnikov kernel: $E(r) = 3 / 4 (1-r^2)_+$
- Quartic (biweight) kernel: $Q(r) = 15 / 16 (1-r^2)^2 dot Ind(|r| >0 1)$
- Triangle kernel: $T(r) = (1-|r|)_+$
- Gaussian kernel: $G(r) = 1 / sqrt(2 pi) e^(-r^2 / 2)$
- Uniform kernel: $Pi(r) = 1 / 2 dot Ind(|r| >= 1)$

#margin[
  The Epanechnikov kernel is theoretically optimal in terms of mean integrated squared error, but in practice, the choice of kernel usually has less impact than the bandwidth selection.
]

== Bandwidth selection
The optimal bandwidth can be found by minimizing the cross-validation criterion:

$
  Q(h) = - sum_(bold(x) in X^ell) ln hat(f) (bold(x) | X^ell \\ bold(x), h) -> min_h
$

where:
- $h$ is the window width (radius)
- $hat(f)$ is the estimated density function
- The notation indicates leave-one-out cross-validation

#margin[
  Cross-validation provides a data-driven approach to bandwidth selection, avoiding both oversmoothing and undersmoothing.
]

This approach:
1. Excludes each point from the training set
2. Estimates the density at that point using the remaining points
3. Minimizes the negative log-likelihood of these estimates

== Relationship to other methods
#margin[
  The connection between these methods highlights the unified theoretical foundation underlying non-parametric approaches.
]

Kernel methods form a common framework that includes:

1. Density estimation: $a_1(bold(x)) = 1 / (ell dot V(h)) sum_(bold(x)' in X^ell) K( (rho(bold(x), bold(x)')) / h )$
2. Classification (Parzen window): $a_2(bold(x)) = arg max_(y in Y) sum_(bold(x)' in X^ell) Ind(y(bold(x)') = y) dot K( (rho(bold(x), bold(x)')) / h)$
3. Regression (Nadaraya-Watson): $a_3(bold(x)) = (sum_(bold(x)' in X^ell) y(bold(x)') dot K( (rho(bold(x), bold(x)')) / h)) / (sum_(bold(x)') K( (rho(bold(x), bold(x)')) / h))$

In these methods:
- $rho$ is a distance function
- $K(rho)$ is a similarity function (larger distance means smaller similarity)

= Mixture Models

== Definition
#margin[
  Mixture models provide a flexible framework that can approximate any continuous distribution with arbitrary precision.
]

A mixture model combines multiple probability distributions:

$
  f(bold(x)) := sum_(n=1)^N w_n dot f_n (bold(x) | bold(theta)_n)
$

where:
- $bold(x) ~ cal(D)_1, ..., cal(D)_N$ indicates data generated by $N$ different distributions
- $w_n := Pr(bold(x) ~ cal(D)_n)$ is the probability of being generated by the $n$th distribution
- $sum_(n=1)^N w_n = 1, quad w_n >= 0$
- $f_n (bold(x) | bold(theta)_n)$ is the pdf/pmf of the $n$th distribution

== Parameter estimation
The log-likelihood function for a mixture model is:

$
  l(bold(w), bold(theta)) = sum_(bold(x) in X^ell) ln sum_(n=1)^N w_n dot f_n (bold(x) | bold(theta)_n) -> max_(bold(w), bold(theta))
$

where:
- $sum_n w_n = 1, quad w_n >= 0$ are constraints
- $f_n (bold(x) | bold(theta)_n)$ is the pdf/pmf of the $n$th distribution

Direct optimization is challenging because the logarithm of a sum doesn't simplify easily. The EM algorithm provides an iterative solution.

== Fixed point method
#margin[
  The fixed point method underlies many iterative algorithms in machine learning, including the EM algorithm.
]

Before introducing EM, it's helpful to understand the fixed point iteration method:

$x_(n+1) = f(x_n)$

This method converges if:
$|f'(x^*)| < 1$

where $x^*$ is the fixed point such that $f(x^*) = x^*$.

= Expectation-Maximization Algorithm

== EM algorithm
#margin[
  The EM algorithm was formalized by Dempster, Laird, and Rubin in 1977, though similar approaches had been used earlier.
]

The EM algorithm iteratively optimizes parameters $w_n$ and $bold(theta)_n$ for mixture components. Each iteration consists of two steps:

1. **Expectation step (E-step)**: Calculate the posterior probability for each data point:
$
  w'_n (bold(x)) := Pr[ bold(x) ~ f_n | bold(x) ] = (w_n dot f_n (bold(x) | bold(theta)_n)) / (sum_(m=1)^N w_m f_m(bold(x), bold(theta)_m))
$

2. **Maximization step (M-step)**: Update parameters:
$
  bold(theta)_n <- arg max_(bold(theta)) sum_(bold(x) in X^ell) w'_n (bold(x)) dot ln f(bold(x) | bold(theta)_n)
$
$
  w_n <- (1) / (ell) sum_(bold(x) in X^ell) w'_n (bold(x))
$

== Theoretical foundation
#margin[
  The EM algorithm can be derived using the Lagrangian function and Karush-Kuhn-Tucker conditions.
]

The EM algorithm optimizes a Lagrangian function:

$
  Q(bold(w), Theta) = sum_(bold(x) in X^ell) ln(sum_(n=1)^N w_n dot f_n (bold(x) | bold(theta)_n)) - lambda dot (sum_(n=1)^N w_n - 1) -> max_(bold(w), Theta)
$

where:
- $Theta = [bold(theta)_1, dots, bold(theta)_N]$ is the parameters matrix
- $lambda$ is the Lagrange multiplier
- $f_n$ is the pdf/pmf of the $n$th distribution
- $w_n := Pr[bold(x) ~ f_n]$ is the probability of coming from the $n$th distribution

== Variants of EM

**Generalized EM (GEM)** relaxes the maximization requirement:
- Standard EM: $bold(theta)^((t+1)) <- arg max_(bold(theta)) ell(bold(theta))$
- GEM: $bold(theta)^((t+1)) <- bold(theta)^* : ell(bold(theta)^*) > ell(bold(theta)^((t)))$

**Stochastic EM (SEM)** optimizes parameters using sampled subsets:
- Generate samples from the estimated distributions
- Optimize parameters independently for each component
- This often accelerates convergence and can use standard maximum likelihood methods

== Determining the number of components
Several approaches can determine the optimal number of distributions $N$ in a mixture:

1. **Greedy addition**: Start with fewer components; add new components if the likelihood for some data points is below a threshold
2. **Greedy deletion**: Start with more components; remove components with small weights
3. **AddDel**: Combination of both approaches
4. **Regularization**: Use cross-entropy regularization to encourage sparsity in component weights

#margin[
  Information criteria like AIC or BIC can also be used to select the optimal number of components, balancing model complexity with goodness-of-fit.
]

== Hierarchical EM
Hierarchical EM extends the standard algorithm to restore hierarchical relationships in the data. It operates by greedily adding components and splitting components with low likelihood. This approach is useful for clustering and enhancing data understanding by revealing structure where single clusters may have multiple subclusters.

= Comparing Density Estimation Methods

== Framework comparison
Different approaches to density estimation include:

1. **Parametric**: Assumes a specific functional form
$hat(f)(bold(x)) = f(bold(x) | bold(theta))$

2. **Non-parametric kernel**: Based on local estimations around each training point
$f(bold(x)) = 1 / ell sum_(bold(x)' in X^ell) 1 / V(h) K( (rho(bold(x), bold(x)')) / h )$

3. **Mixture models**: Combines multiple distributions
$hat(f)(bold(x)) = sum_(n=1)^N w_n dot f_n (bold(x) | bold(theta)_n)$

#margin[
  Mixture models provide a unified framework - when $N=1$, they reduce to parametric methods, and as $N$ approaches the sample size, they approximate non-parametric methods.
]

These approaches represent a spectrum: mixture models generalize both parametric methods (when $N=1$) and non-parametric approaches (when $N$ equals the sample size).

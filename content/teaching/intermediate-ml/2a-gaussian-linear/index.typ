#import "@preview/physica:0.9.3": *
#import "../defs.typ": *
#import "../template.typ": *

#show: template

= Weighted Least Squares (WLS)

== Intro
#margin(
  title: [Heteroscedasticity],
)[
  can be eliminated by applying weighted LS.

  For a model with non-constant variance of the error term:

  $
    bold(y) = X bold(beta) + bold(epsilon), quad Var[epsilon(bold(x))] = y(bold(x))^2 dot sigma^2
  $

  To apply WLS, the weights must have a negative square unit:

  $
    w(bold(x)) = 1 / y(bold(x))^2
  $

  This leads to the transformations:

  $
    bold(y)' = bold(y) / sqrt(w), quad bold(x)' = bold(x) / sqrt(w), quad bold(epsilon)' = bold(epsilon) / sqrt(w)
  $

  The weight matrix is:

  $
    W = dmat(1 / y(bold(x)_1)^2, dots.down, 1 / y(bold(x)_ell)^2)
  $

  and

  $
    bold(y)' = sqrt(W) bold(y), quad bold(x)' = sqrt(W) bold(x), quad bold(epsilon)' = sqrt(W) bold(epsilon)
  $

  Now, the model can be formulated as a homoscedastic least squares problem:

  $
    bold(y)' = X' bold(beta) + bold(epsilon)', quad Var[epsilon'(bold(x))] = sigma^2
  $
]
The *Weighted Least Squares (WLS)* method extends ordinary least squares by incorporating
observation-specific weights. The basic model structure remains similar to OLS:

$
  hat(y)(bold(x)) = bold(x)^Tr hat(bold(beta)) + epsilon(bold(x))
$

where $bold(x)$ is the vector of features, $bold(beta)$ is the vector of parameters, $y(bold(x))$ is
the target variable, and $epsilon(bold(x))$ is the error term.

- Each observation $bold(x)$ has associated weights $w(bold(x))$ that reflect the importance
  of that particular observation.

- This method minimizes the weighted sum of squared residuals:

$
  RSS = sum_(bold(x) in X^ell) w(bold(x)) dot (y(bold(x)) - hat(y)(bold(x)))^2 -> min_bold(beta).
$

- The solution to this minimization problem is given by:

$
  bold(beta)^* = ub((X^Tr W X)^(-1) X^Tr W, X^+_W) bold(y),
$

where $W$ is the diagonal matrix of weights, and $X^+_W$ is the weighted pseudo-inverse.

== Weight matrix
For a weighted

$
  RSS = sum_(bold(x) in X^ell) w(bold(x)) dot (y(bold(x)) - hat(y)(bold(x)))^2
$

let's introduce the weight matrix:

$
  W :&= diag(w(bold(x)_1), ..., w(bold(x)_ell))
     &= dmat(w(bold(x)_1), dots.down, w(bold(x)_ell))
$

== Matrix form
#margin(
  title: [Quadratic form],
)[
  is a function of the form:

  $
    Q(x_1, ..., x_n) = sum_(i=1)^n sum_(j=1)^n a_(i,j) x_i x_j.
  $

  Coefficients $a_(i,j)$ can be arranged in a symmetric matrix $A$, and the quadratic form
  can be written in matrix form as:

  $
    Q(bold(x)) = bold(x)^T A bold(x).
  $
]
Thus, we can rewrite the RSS in matrix form as a quadratic form:

$
  RSS = (bold(y) - X bold(beta))^Tr W (bold(y) - X bold(beta)).
$

== Back to standard LS
The weighted LS problem can be easily reformulated as a standard LS problem by replacing
the original variables with transformed ones:

$
  bold(y)' := W^(1/2) bold(y), quad X' := W^(1/2) X, quad bold(epsilon)' := W^(1/2) bold(epsilon)
$

Substituting these transformations into the original model, we get:

$
  bold(y)' = X' bold(beta) + bold(epsilon)'
$

== Analytical solution
Now, let's solve for $bold(beta)$ in the transformed model. Since $W$ and $W^{1/2}$ are
diagonal matrices, transposing them results in the same matrix:

$
  bold(beta)^* = X'^+ bold(y)' = (X'^Tr X')^(-1) X'^Tr bold(y)'
$

Expanding the expressions:

$
  bold(beta)^*
    &= ((W^(1/2) X)^T W^(1/2) X)^(-1) (W^(1/2) X)^Tr W^(1/2) bold(y) \
    &= (X^Tr W X)^(-1) X^Tr W bold(y)
$

Therefore, the solution is:

$
  bold(beta)^* = ub((X^Tr W X)^(-1) X^Tr W, X^+_W) bold(y)
$
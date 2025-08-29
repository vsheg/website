#import "../template.typ": *
#show: template

= GLM: Logistic regression

#note(
  title: [Logistic regression],
)[
  A model predicts if the patient is a vegan ($y = 1$) or not ($y = 0$) by a result of
  cholesterol test and gets the result $x$ mmol/L. A binary response modeled with Bernoulli
  distribution $y tilde cal(B)(p)$, where $p := Pr[y=1]$ is the probability of being a
  vegan.

  Bernoulli distribution belongs to the exponential family; in canonical form, it is:

  $ y tilde f(y|theta) = (1-sigma(theta)) dot e^(theta dot y), $

  where $theta = logit(p) = log(p/(1-p))$ is the logit function.

  Fitting the parameter $theta$ on the historical data:

  $ theta^* = arg max_theta sum_((x^*, y^*) in (X, Y)^ell) f(y = y^*|x=x^*, theta) $

  Then, to make a prognosis, the probability of being a vegan is calculated:

]

#margin(
  [
    In logistic regression, the connection between the input $bold(x)$ and the probability $p$ is
    modeled as:

    $ ub(log p/(1-p), logit p) = beta_0 + beta_1 dot x $

    The inverse of the logit function is the sigmoid function $sigma(theta) = (1 + e^(-theta))^(-1)$:
    $ p = logit^(-1)(beta_0 + beta_1 dot x) = sigma(beta_0 + beta_1 dot x) $

    The parameters $beta_0$ and $beta_1$ are trained on the historical data:

    $ ell(beta_0, beta_1) = log product_i p_i^(y_i) (1-p_i)^(1-y_i) -> max_(beta_0, beta_1). $

    Then, to make a prognosis, the probability of a bad outcome is calculated:

    $ hat(p)(bold(x)) = sigma(beta_0 + beta_1 dot x). $

  ],
)

#margin(
  title: [NB],
)[
  We used distribution of $y tilde cal(B)(p)$ to derive the loss function:

  $ ell = log product_i Pr[y_i = 1]^(y_i) dot Pr[y_i = 0]^(1-y_i), $

  but we ignore the distribution of $y$ when we make a prediction, we are only interested in
]

#margin(
  title: [Exponential family],
)[
  Bernoulli distribution $cal(B)(p)$ belongs to the exponential family, the exponential
  family parameter $theta$ can be calculated from the Bernoulli parameter $p$, then $ y
  tilde Exp(theta). $
]

= Generalized Linear Models (GLM)

== Introduction to GLM
A generalized linear model (GLM) extends ordinary linear regression by allowing for
response variables that follow any exponential family distribution. The general form is: $ Y
tilde f(y | bold(theta)) = exp[bold(theta) dot T(y) - A(bold(theta)) + C(y)] $
== Making Predictions
To make a prediction in GLM, we estimate the conditional expectation (canonical mean
parameter):
$ hat(y)(bold(x)) := Ex[Y | X = bold(x)] equiv mu $
For most cases, the sufficient statistics T is trivial, and we can obtain the needed
expectation from the distribution parameters:
$ hat(y)(bold(x)) := Ex[Y | bold(theta)] = Ex[Y | bold(theta) = F bold(beta)] = Ex[Y | X =
bold(x)] $
For example, in logistic regression, the mean parameter corresponds to probability:
$ mu := Ex[Y | bold(theta)] = Pr[Y = 1 | bold(theta)] = Pr[Y = 1 | X = bold(x)] = hat(p) $
== Mean and Link Functions

=== Mean Function
The mean function describes the expected value of the response variable Y (or sufficient
statistics T(Y)) given current parameters:
$ bold(mu) := Ex[T(bold(Y)) | bold(theta)] $
=== Link Function
The link function connects linear parameters θ = Xβ (linear predictor) to the expected
value (canonical mean):
$ bold(mu) = psi(bold(theta)) $
Its inverse calculates parameters:
$ bold(theta) = psi^(-1)(bold(mu)) $
== GLM as Linear + Nonlinear Transforms
GLM combines linear and nonlinear transformations:

1. Linear predictor computation:
$ theta(bold(x)) = bold(beta)^Tr bold(x) = M bold(x) $ $ bold(theta) = X bold(beta) $
2. Link function application:
$ psi colon quad theta = psi(mu) $ $ psi(Ex[y|bold(x)]) = bold(beta)^Tr bold(x) $ $ psi(Ex[y|X])=
X bold(beta) $
3. Final prediction via inverse link function:
$ hat(y)(bold(x)) = Ex[y|bold(x)] = psi^(-1)(bold(beta)^Tr bold(x)) $
== Logistic Regression as GLM
Logistic regression is a special case of GLM using Bernoulli distribution:
$ y_i tilde cal(B)(p), quad Pr[y_i = 1] = p $
In canonical form:
$ y_i tilde f(y | theta) = (1-sigma(theta)) dot e^(theta dot y) $
The link function can be found from:
$ psi = (grad_bold(theta) A)^(-1) $
Where:
$ pdv(A, theta) = sigma(theta) = 1/(1 + e^(-theta)) $ $ psi(mu) = sigma^(-1)(p) = ln
p/(1-p) = "logit" p $
= GLM: Cross-entropy and log-loss

== Model
Logistic regression represents a special case of GLM where the binary response variable
$Y$ follows a Bernoulli distribution: $
  y_i tilde cal(B)(p), quad p := Pr[y_i = 1]
$
Here,
$p$ represents the success probability in a single trial. The canonical form of the
Bernoulli distribution is: $
  y_i tilde f(y|theta) = sigma(-theta) dot e^(theta dot y), quad sigma(theta) = 1 / (1 +
  e^(-theta))
$
Starting from the general GLM form:
$
  Y tilde f(y|bold(theta)) = exp[bold(theta) dot T(y) - A(bold(theta)) + C(y)]
$
We can derive both cross-entropy and log-loss directly, assuming only the Bernoulli
distribution of
$Y$.

== Link Function
The link function $psi$ connects the response variable's mean $mu = Ex[Y]$ to the
distribution's canonical parameters $bold(theta)$: $ bold(mu) = psi(bold(theta)) $
In GLM, we assume the canonical parameters are linear:
$ theta_i = bold(x)_i^Tr bold(beta), quad bold(theta) = X bold(beta) $
where
$bold(beta)$ represents the linear coefficients corresponding to features in $bold(x)$.

For the Bernoulli distribution, the link function takes the form: $
  psi(mu) = log mu/(1-mu) = "logit" mu
$
#columns(
  2,
)[
  == Cross-entropy Loss
  We begin with the log-likelihood function $l(theta)$ for the Bernoulli-distributed
  response variable $Y$, assuming $theta = bold(x)^Tr bold(beta)$:

  $
    l(theta) &= log product_i f(y_i|theta) \
             &= log product_i sigma(-theta) dot e^(theta dot y_i) \
             &= sum_i {theta dot y_i + log sigma(-theta)} \
             &= sum_i {theta dot y_i + log 1 / (1+e^(-(-theta)))} \
             &= sum_i {y_i log mu / (1 - mu) + log 1 / (1 + mu / (1 - mu))} \
             &= sum_i {y_i log mu / (1 - mu) + log (1 - mu) / (1 - mu + mu)} \
             &= sum_i {y_i log mu - y_i log (1 - mu) + log (1 - mu)} \
             &= sum_i {y_i log mu + (1 - y_i) log (1 - mu)} \
             &= sum_i {y_i log p + (1 - y_i) log (1 - p)} \
             &= l(p(bold(beta))) -> max_(bold(beta))
  $

  #colbreak()

  == Log-loss
  The log-loss function $ell(M)$ can be derived by taking the negative log-likelihood:

  $
    -l(theta) &= -sum_i {theta dot y_i + log e^(-theta) / (1+e^(-theta))} \
              &= sum_i cases(
      -log e^(theta) + log e^(-theta) / (1+e^(-theta)) \, &"if" y = 1,
      -log e^(-theta) / (1+e^(-theta)) \, "if" y = 0,

    ) \
              &= sum_i cases(log(1+e^(-theta)) \, "if" y = 1, log(1+e^(theta)) \, "if" y
    = 0) \
              &= sum_i log(1 + e^(theta dot sgn y_i)) \
              &= sum_i log(1 + e^(bra bold(x)_i, bold(beta) ket dot sgn y_i)) \
              &= sum_i log(1 + e^(-M_i)) \
              &= ell(M(bold(beta))) -> min_(bold(beta))
  $
]

== Making Predictions
To make a prediction:
$
  hat(p)(bold(x)) = mu(bold(x)) = psi(theta = bold(x) dot bold(beta)) = 1 / (1 + e^(bold(x)
  dot bold(beta)))
$


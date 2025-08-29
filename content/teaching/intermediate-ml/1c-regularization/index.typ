#import "../template.typ": *
#show: template

= General concept of regularization
#margin[
  The regularization method traces back to A.N. Tikhonov's work in 1963, who proposed it for
  solving ill-posed problems where formal mathematical solutions are meaningless.

  A linear system $bold(y) = X bold(beta)$ has no solution when $X$ is singular ($det X = 0$),
  rank-deficient ($rank X < k$), or when data contains errors preventing $bold(y) = X bold(beta)$ from
  being satisfied:

  $ norm(y - X bold(beta))_2^2 ->^(?) min_beta. $

  Adding constraints on model parameters $bold(beta)$ through a regularizer $R(bold(beta))$ term
  shrinks the solution space, making it possible to find (sometimes) a practically useful
  approximate solution:

  $ norm(y - X bold(beta))_2^2 + R(bold(beta)) ->^"OK" min_beta. $
]
Regularization is a technique that imposes constraints on model parameters to control the
solution space. By limiting where parameters can be found, it effectively shrinks the
space of possible solutions. This reduction in parameter flexibility not only enhances
model generalizability but also helps prevent overfitting by focusing on simpler
solutions.

The term derives from Latin "regula" (rule) and "regularis" (in accordance with rules),
reflecting its role in establishing systematic constraints on model behavior.

Through these controlled parameter constraints and reduced solution space, regularization
helps create simpler, more robust models by reducing their sensitivity to noise in the
training data.

= Probabilistic interpretation of regularization

== Probabilistic framework
Consider a joint distribution of data $bold(x) in RR^k, y in RR$ and model's parameters $bold(theta) in RR$:

$ bold(x), y, bold(theta) tilde.op X, Y, Theta. $

+ The prior distribution of $bold(x)$ is independent of parameters $bold(theta)$ and can
  assumed to be uniform and ignored in the model:
  $ X ~ f_X (bold(x)|bold(theta)) :> f_X (bold(x)) ~ cal(U). $

+ The posterior distribution of responses $y$ depends on parameters $bold(theta)$ and
  specific data point $bold(x)'$, following a semi-probabilistic model formalism. The model
  is specified by defining the conditional distribution of responses $Pr[y|bold(x) = bold(x)^*, bold(theta)]$ given
  a specific $bold(x) = bold(x)^*$ and model parameters $bold(theta)$. When the parameters
  are fitted, we make predictions for new data points $bold(x)'$ by maximizing the
  probability of a response $y$ given $bold(x)'$:
  #margin[Support of a random variable $Y$ is the set of all possible values $y^*_1, y^*_2, ...$ that $Y$ can
    take with non-zero probability:
    $ supp Y = {y^*_1, y^*_2, ...} $]
  $ a_(bold(theta)) (bold(x)') = arg max_(y in supp Y) ub(f_Y (y|bold(x) = bold(x)', bold(theta) = hat(bold(theta))), "model"). $

+ The prior distribution of parameters $bold(theta)$ is assumed to be known and defined by
  the hyperparameter vector $bold(gamma)$:
  $ f_Theta (bold(theta)) :> f_Theta (bold(theta)|bold(gamma)) ~ Theta(bold(gamma)) $

== Applying MAP
The joint distribution of data and parameters can be rewritten as a product of conditional
pdfs:

#margin[
  We omit random variables $X, Y, Theta$ in the pdf's underscripts for brevity. Just look at
  the arguments before the bar to understand to which random variable the pdf refers: e.g., $f(x, y|theta)$ means $f_(X,Y) (bold(x),y|theta)$.
]

#margin[
  $
    Pr[y|x, theta] &= Pr[{Y = y}|{X = x}, {Theta = theta}] \
                   &= Pr[{Y = y}|{X = x}{Theta = theta}] \
                   &= Pr[{Y = y}{X = x}{Theta = theta}] / Pr[{X = x}{Theta = theta}] \
                   &= Pr[x, y, theta] / Pr[x, theta]
  $
]

$
  f(bold(x), y, bold(theta)) &= f(y|bold(x),bold(theta)) dot f(bold(x), bold(theta)) \
                             &= f(y|bold(x), bold(theta)) dot cancel(f(bold(x)|bold(theta))) dot f(bold(theta)|bold(gamma)) \
                             &= f(y|bold(x), bold(theta)) dot f(bold(theta)|bold(gamma))
$

As it was mentioned, the canceled prior distribution of data $f (bold(x)|bold(theta))$ is
independent of the model parameters $bold(theta)$. We ignore it (or assume uniform).

Still, we didn't ignore the prior distribution of parameters $bold(theta)$, which is $f(bold(theta)|bold(gamma))$.
Because of that, it's MAP (Maximum a Posteriori) estimation, not MLE (Maximum Likelihood
Estimation).

#margin[
  A pdf $f(y|x, theta)$ becomes a likelihood function when we consider it as a function of
  arguments behind the bar, e.g.
  - $h(y) := f(y|x = x^*, theta = theta^*)$ is still a pdf of $y$ given $x = x^*$ and $theta = theta^*$.
  - $g(theta) := f(y = y^*|x = x^*, theta)$ is already a likelihood function of $theta$ given $x = x^*$ and $y = y^*$.
]

== Finding parameters
For specific training samples $y^*, bold(x)^*$ and predefined hyperparameters $bold(gamma)^*$,
we write the joint likelihood of data and model parameters and maximize it:

$
  ell(bold(theta))
    &=
  log product_((bold(x)^*, y^*) in (X, Y)^ell) ub(
    f(y = y^*|bold(x)=bold(x)^*, bold(theta)) dot f(bold(theta)|bold(gamma)=bold(gamma)^*),
    "MAP",

  ) \

    &= sum_((bold(x)^*, y^*) in (X, Y)^ell) {log f(y = y^*|bold(x)=bold(x)^*, bold(theta)) + log f(bold(theta)|bold(gamma)=bold(gamma)^*)} \
    &= sum_((bold(x)^*, y^*) in (X, Y)^ell) ub(log f(y = y^*|bold(x)=bold(x)^*, bold(theta)), "log-likelihood") + ub(lambda dot log f(bold(theta)|bold(gamma)=bold(gamma)^*), "prior regularizer"),

  )
  -> max_(bold(theta))
$<eq-regularizer-general>

The second term is the regularizer, its strength is defined by constant $lambda$ and
hyperparameters $bold(gamma)$. Regularizer narrows the space in which the parameters can
be found. The more narrow the space, the more constrained the model is.

After finding the parameter vector estimate $hat(bold(theta))$, predictions for a new data
point $bold(x)'$ can be made by substituting the estimate $hat(bold(theta))$ into the
model $f_Y (y|bold(x) = bold(x)', bold(theta) = hat(bold(theta)))$:

$
  a_hat(bold(theta)) (bold(x)') = arg max_(y in supp Y) f_Y (y|bold(x) = bold(x)', bold(theta) = hat(bold(theta)))
$

== Loss-function
Probabilistic regularizer @eq-regularizer-general can be rewritten as the empirical risk
where it becomes an additional loss function:

$
  R(bold(theta)) = ub(sum_(bold(x) in X^ell) cal(L)(bold(x)), "data fitting") + ub(lambda dot cal(L)_"reg"(bold(theta)), "parameters" \ "regularization ") -> min_(bold(theta))
$

= $L_2$-norm regularization (Tikhonov regularization)

On a privious step, we found the general form of the regularazer assuming prior
distribution of parameters $f_Theta (bold(theta)|bold(gamma))$:

$
  R(bold(theta)) = sum_(bold(x) in X^ell) cal(L)(bold(x)) + lambda dot cal(L)_"reg" (bold(theta)) -> min_(bold(theta)).
$

== Model:
Here we make specific choices of the prior distributions parameters $f_Theta (bold(theta)|bold(gamma))$ and
the data:
+ All parameters $bold(theta) :> bold(beta)$ are independent and linear, so the joint
  distribution is a product of the individual distributions:
  #margin[These distributions impose prior constraints on the model coefficients, effectively
    reducing the solution space]
  $ f_Theta (bold(theta) = bold(beta)|bold(gamma)) = product_(j=1)^k f (beta_j|bold(gamma)) $

+ Each parameter $beta_j$ follows a Gaussian distribution with two hyperparameters common
  for all individual distributions: mean $gamma_1 :> mu = 0$ and standard deviation $gamma_2 :> tau$:
  $ f (beta_j|bold(gamma)) :> f (beta_j|mu, tau) = 1/(sqrt(2 pi) tau) e^(-beta_j^2 \/ 2 tau^2) ~ N(mu, tau). $
+ The data is generated by a linear model with Gaussian noise $N(0, sigma)$:
  #margin[Both errors $bold(epsilon) ~ N(bold(0), sigma I)$ and model's parameters $beta ~ N(bold(0), tau I)$ follow
    multivariate Gaussian distributions with zero mean and different covariance matrices $sigma I$ and $tau I$ respectively.]
  #margin[Dimensions of vectors $bold(epsilon)$ and
    $bold(beta)$ are different:
    $ dim bold(epsilon) = ell, quad dim bold(beta) = k. $

    All corresponding distribution parameters have appropriate dimensions:
    $ dim bold(0)_bold(epsilon) = ell, quad dim bold(0)_bold(beta) = k, \
    dim sigma I = ell times ell, quad dim tau I = k times k. $
  ]
  $ y(bold(x)) = beta^Tr bold(x) + epsilon(bold(x)), quad epsilon(bold(x)) ~ N(0, sigma). $

== Applying MAP
#margin[Here we denote pdf underscripts with letters corresponding to random variables]
For any arbitrary model we can estimate the error term $hat(bold(epsilon))$ as difference
between the predicted $hat(bold(y))$ and the actual $bold(y)$ responses. The posterior
distribution of parameters $bold(beta)$:
$
  f_bold(beta) (bold(beta)|bold(gamma), bold(epsilon) = hat(bold(epsilon)))
    &= (f_(bold(beta), bold(gamma), bold(epsilon)) (bold(beta), bold(gamma), bold(epsilon) = hat(bold(epsilon)))) / (f_(bold(gamma), bold(epsilon)) (bold(gamma), bold(epsilon)=hat(bold(epsilon)))) \
    &= (f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon))|bold(beta), bold(gamma)) dot f_(bold(beta), bold(gamma)) (bold(beta), bold(gamma))) / (f_(bold(gamma), bold(epsilon)) (bold(gamma), bold(epsilon)=hat(bold(epsilon)))) \
    &= (f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon))|bold(beta), bold(gamma)) dot f_bold(beta) (bold(beta)|bold(gamma)) dot f_bold(gamma) (bold(gamma))) / (f_(bold(gamma), bold(epsilon)) (bold(gamma), bold(epsilon)=hat(bold(epsilon)))) \
    &= (f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon))|bold(beta)) dot f_bold(beta) (bold(beta)|bold(gamma)) dot cancel(f_bold(gamma) (bold(gamma)))) / (cancel(f_bold(gamma) (bold(gamma))) dot f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon)))) -> max_bold(beta)\
$
As $f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon)))$ is independent of $bold(beta)$,
we cancel it out:
$
  frame(
    f_epsilon (bold(epsilon) = hat(bold(epsilon))|bold(beta)) dot f_bold(beta)(bold(beta))
    -> max_bold(beta).
  )
$

== Independence
We applied MAP and wrote the optimization problem, now continue with substituting the
specific distributions:

$
  f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon))|bold(beta)) = product_(bold(x)^* in
  X^ell) e^(-epsilon(bold(x) = bold(x)^*|bold(beta))^2 \/ 2 sigma^2)
$

The error estimates are directly related to the data:
$
  hat(epsilon)(bold(x) = bold(x)^*|bold(beta)) = hat(y)(bold(x)^*|bold(beta)) - y(bold(x)^*)
$
Let's write the prior distribution of parameters:
$
  f_bold(beta) (bold(beta)) = product_(j=1)^k e^(-beta_j^2 / 2 tau^2)
$
The data distribution can be written through the error distribution:
$
  f_Y (bold(y)=bold(y)^*|bold(beta)) = f_bold(epsilon) (bold(epsilon) = bold(y) -
  bold(y)^*|bold(beta))
$
Let's write the posterior distribution of parameters:
$
  f_bold(beta) (bold(beta)|bold(epsilon)=hat(bold(epsilon))) := product_(bold(x) in X^ell)
  e^(-hat(epsilon)(bold(x)|bold(beta))^2 / 2 sigma^2) dot product_(j=1)^k e^(-beta_j^2 / 2
  tau^2)
$
Let's write the likelihood function (log-loss):
$
  ell(bold(epsilon), bold(beta)|X) := -sum_(bold(x) in X^ell) epsilon(bold(x)|bold(beta))^2
  / (2 sigma^2 )- sum_(j=1)^k beta_j^2 / (2 tau^2) -> max_(bold(beta))
$
Let's rewrite it as empirical risk minimization:
$
  Q(bold(beta)) = sum_(bold(x) in X^ell) (hat(y)(bold(x)) - y(bold(x)))^2 + (2 sigma^2) / (2
  tau^2) sum_(j=1)^k beta_j^2 -> min_(bold(beta))
$ $
  Q(bold(beta)) = norm(bold(y) - X bold(beta))_2^2 + lambda dot norm(bold(beta))_2^2 ->
  min_bold(beta)
$
In
$L_1$ regularization, everything is similar, but the errors are described by the Laplace
distribution.

= $L_1$-norm regularization
$
  Q(bold(beta)) = norm(bold(y) - X bold(beta))_2^2 + lambda dot norm(bold(beta))_1 ->
  min_bold(beta)
$
Unlike
$L_2$ regularization, LASSO assumes model errors follow the Laplace distribution,
characterized by heavy tails and a sharp peak: $
  epsilon tilde pdf(epsilon|mu,b) = 1/(2b) exp (-abs(epsilon - mu)) / b
$
Using MAP for parameter estimation:
$
  f_bold(beta) (bold(beta)|bold(epsilon)=hat(bold(epsilon))) = (f_epsilon
  (bold(epsilon)=hat(bold(epsilon))|bold(beta)) dot f_bold(beta) (bold(beta))) /
  cancel(f_bold(epsilon) (bold(epsilon) = hat(bold(epsilon))))\
  = product_(bold(x)^* in X^ell) e^(-epsilon(bold(x) = bold(x)^*|bold(beta))^2 / 2 sigma^2)
  dot
  product_(j=1)^k 1/(2 b) e^(-abs(beta_j) / b)\
  -> max_bold(beta)
$
Let's write the likelihood function (log-loss):
$
  ell(bold(epsilon), bold(beta)|X) := -sum_(bold(x) in X^ell) epsilon(bold(x)|bold(beta))^2
  / (2 sigma^2 )- sum_(j=1)^k abs(beta_j) / b - cancel(k dot ln 2 b) -> max_(bold(beta))
$
Let's rewrite it as empirical risk minimization:
$
  Q(bold(beta)) = sum_(bold(x) in X^ell) (hat(y)(bold(x)) - y(bold(x)))^2 + (2 sigma^2) / b
  sum_(j=1)^k abs(beta_j) -> min_(bold(beta))
$

= $L_0$ regularization

= Geometric interpretation of regularization

= Ridge, LASSO, and Elastic Net regression

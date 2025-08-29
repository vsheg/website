#import "../template.typ": *
#show: template

= Quantile $QQ_q$ of a random variable

== Quantile of a sample
Given an unordered sample $y_1, y_2, ..., y_ell$, we can construct a sorted sample $y^((1)) <= y^((2)) <= ... <= y^((ell))$,
where $y^((i))$ is the $i$th smallest value of the sample ($i = 1..ell$), also known as the $i$th _order statistic_.

#margin[
  Some order statistics:
  - $y^((1)) = min Y$ is the 1st order statistic
  - $y^((2))$ is the 2nd order statistic (2nd smallest value)
  - $y^((ell\/2))$ is the median, which divides the sample in half
  - $y^((ell)) = max Y$ is the last ($ell$th) order statistic
]

#margin[NB][
  In $y^((i))$, values $i = 1..ell$ are integers, while in $y^((q))$, values $q in [0..1]$ are
  fractional, e.g., $y^((10))$ refers to the 10th order statistic, while $y^((0.1))$ refers to the
  0.1-quantile.
]

Informally, the $q$-quantile $y^((q))$ is the value that divides the ordered sample into two parts
with proportions $q : (1 - q)$. However, this definition is ambiguous. One practical approach is to
use different formulas for $q dot ell in.not NN$ and $q dot ell in NN$:

#margin[
  In practice, other definitions of quantiles $y^((q))$ are also used, e.g., $y^((q)) := y^((ceil(q dot ell)))$ for
  any $ell$
]

$
  y^((q)) := cases(
    y^((ceil(q dot ell))) & quad q dot ell "is not integer",
    1/2 (y^((q dot ell)) + y^((q dot ell + 1))) & quad q dot ell "is integer"
  ,

  )
$

== Quantile of a random variable
For a random variable $Y$, the quantile function, denoted either as $QQ_q [Y]$ or $y^((q))$, is
defined as the inverse of its CDF:

$ QQ_q [Y] := cdf_Y^(-1)(q) = inf { y | cdf_Y (y) >= q }, $ <eq-quantile-def>

where $inf$ denotes the infimum, which is the greatest lower bound, i.e., $QQ_q [Y]$ is the smallest
value $y$ for which the probability $Pr[Y <= y]$ is at least $q$.

#grid(
  columns: 2,
  example[
    For a uniform distribution on interval $[a,b]$, the CDF is:

    $
      cdf_Y (y^*) = cases((y^* - a)/(b - a)\, & y^* in [a..b], 0\, & y^* < a, 1\, & y^* > b
      )
    $

    The corresponding quantile function is:

    $ QQ_q [Y] = cdf_Y^(-1)(q) = a + q dot (b - a). $
  ],
  example[
    For a sample ${y_1, ..., y_ell}$, the empirical CDF

    $ cdf_Y (y) := 1 / ell dot sum_(i=1)^ell Ind(y_i <= y) $

    can be used in @eq-quantile-def to define quantiles $QQ_q$.
  ],
)

#margin[
  For a continuous random variable $Y$, the probability of $Y$ being less than or equal to $y^*$
  is given by the cumulative distribution function (CDF):

  $ cdf_Y (y^*) := Pr[Y <= y^*] = integral_(y=-oo)^(y^*) pdf_Y (y) dd(y), $

  where $pdf_Y (y)$ is the probability density function (PDF).
]

#margin[
  For a discrete random variable $Y$, the CDF is defined as:

  $ cdf_Y (y^*) := Pr[Y <= y^*] = sum_(y <= y^*) pmf_Y (y) $

  where $pmf_Y (y)$ is the probability mass function (PMF).
]

#margin[
  Some important quantiles:
  - $QQ_0 [Y] = min Y$ is the minimum value
  - $QQ_(1\/4) [Y]$ is the 1st quartile ($Q_1$)
  - $QQ_(1\/2) [Y]$ is the median or 2nd quartile ($Q_2$)
  - $QQ_(3\/4) [Y]$ is the 3rd quartile ($Q_3$)
  - $QQ_1 [Y] = max Y$ is the maximum value

  Percentiles are also quantiles, e.g. $QQ_(0.95) [Y]$ is the 95th percentile.
]

// TODO: add example of quantile to illustrate infimum

== Quantile $QQ_q$ and probability $Pr$
CDF maps real numbers $y in RR$ to probabilities $p in [0..1]$:

$
  cdf_Y (y) := Pr[Y <= y] colon y -> p.
$

Quantile $QQ_q$, being the inverse of CDF, maps probabilities $p in [0..1]$ to real numbers $y in RR$:

$
  QQ_p [Y] = cdf_Y^(-1) (p) colon p -> y,
$

We specifically denote probability $p$ as $q$ to emphasize its connection to quantiles.

#margin[
  Technically, $QQ_q [Y]$ is a function of $q$, and it is usually denoted as $Q_Y (p)$, similar to PDF $pdf_Y (y)$ and
  CDF $cdf_Y (y)$.

  However, the notation $QQ_q [Y]$ is used here to emphasize the analogy between quantiles $QQ_q [Y]$ and
  expectation $Ex[Y]$.
]

The meaning of $QQ_q [Y]$ is that it is the value of $Y$ such that the probability of $Y$ being less
than or equal to $QQ_q [Y]$ is $q$:

$ Pr[Y <= QQ_q [Y]] = q. $ <eq-quantile-probability>

== Conditional quantile $QQ_q [Y|X]$
The generalization of the quantile $QQ_q [Y]$ to the conditional case is straightforward; it's
defined as the inverse of the conditional CDF:

$ QQ_q [Y|X] := cdf_(Y|X)^(-1)(q) = inf { y | cdf_(Y|X)(y) >= q }, $

where $cdf_(Y|X)(y) := Pr[Y <= y|X]$ is the conditional CDF defined via the conditional PDF $pdf_(Y|X)(y) equiv pdf_Y (y|X)$ (continuous
case) or PMF $pmf_(Y|X)(y) equiv pmf_Y(y|X)$ (discrete case).

The meaning of the conditional quantile $QQ_q [Y|X]$ is that it is the value of $Y$ such that the
probability of $Y$ being less than or equal to $QQ_q [Y|X]$ given $X$ is $q$:

$
  Pr[ thick Y <= QQ_q [Y|X] thick | thick X thick] = q.
$ <eq-quantile-probability-conditional>


= Quantile loss $cal(L)_q$

== Check-loss
Consider an asymmetric loss function parameterized by $q in (0, 1)$:
#margin[
  This loss function is also called the _pinball loss_ and _quantile loss_
]

$
  cal(L)_q (epsilon)
  :&= cases(q dot epsilon & quad epsilon >= 0, -(1-q) dot epsilon & quad epsilon < 0
  ) \
  &= epsilon dot q dot Ind(epsilon >=0) - epsilon dot (1-q) dot Ind(epsilon<=0)
  ,
$ <eq-check-loss>

#margin[
  Strictly speaking, this is an estimation of the error: $hat(epsilon) := y - hat(y)$; for different
  estimations of $hat(y)$, there are different $hat(epsilon)$
]

where $epsilon := y - hat(y)$ is the error term (residual) and $hat(y)$ is the prediction of a
regression model.

#margin({
  // NOTE: when plotted on the same graph, it looks like a six-legged spider: not illustrative at all
  let x = lq.linspace(-2, 2)

  let sub-figure(q) = figure(
    caption: [Check loss $cal(L)_#q (epsilon)$],
    lq.diagram(
      width: 3cm,
      height: 3cm / 2,
      xlabel: $epsilon$,
      lq.plot(
        mark: none,
        x,
        x.map(epsilon => if (epsilon >= 0) { q * epsilon } else { -(1 - q) * epsilon }),
      ),
    ),
  )

  multi-figure(sub-figure(0.25), sub-figure(0.5), sub-figure(0.75))
})

== Constant model
Let's first consider the simplest case, where we look for $a^*$ in the family of all constant models $a^* in {a | a = const}$.

#margin[
  For a pair $(bold(x), y)$ taken from the joint distribution $pdf(bold(x), y)$, a function
  $hat(y)(bold(x)) = a^*(bold(x))$ that minimizes $cal(R)(a = a^*)$ can be found by minimizing $cal(R)(a)$:

  $
    cal(R)(a) &:= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (a(bold(x)), y)] \
    &= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (y - a(bold(x)))] \
    &= integral cal(L)_q (y - a(bold(x))) dot pdf(bold(x), y) dot dd(bold(x)) dd(y) \
    &= integral cal(L)_q (y - a(bold(x))) dot dd(cdf(bold(x), y)) -> min_a
  $
]

The empirical risk (expected check-loss) can be expressed as:

$
  cal(R)(a) &= integral cal(L)_q (focus(epsilon = y - a)) dd(cdf(bold(x), y)) comment(a(bold(x)) -> a = const) \
            &= integral cal(L)_q (epsilon = y - a) focus(dd(cdf(y))) comment("as" a "is not a function of" bold(x)) \

            &= limits(integral)_focus(y - a >= 0) cal(L)_q (y - a) dd(cdf(y)) + limits(integral)_focus(y - a < 0) cal(L)_q (y - a) dd(cdf(y)) #comment[nonoverlapping regions: $epsilon >= 0$ and $epsilon < 0$] \

            &= limits(integral)_(y >= a) focus((y - a) dot q) dd(cdf(y)) - limits(integral)_(y < a) focus((y - a) dot (1-q)) dd(cdf(y)) -> min_a #comment[expand $cal(L)_q (epsilon)$ according to @eq-check-loss] \
$

== Risk minimization
The integral is split at $a = a^*$ into two independent regions: $(-infinity..a^*)$ and $[a^*..+infinity)$.
By differentiating both integrals with respect to $a$, we can find $a^*$:

$
  pdv(, a) cal(R)(a) &= focus(q) dot integral_(y>=a) pdv(, a) (y - a) dd(cdf(y)) - focus((1-q)) dot integral_(y < a) pdv(, a) (y - a) dd(cdf(y)) #comment[constants] \
                     &= -q dot focus(integral_(y = a^*)^(+oo) dd(cdf(y))) + (1-q) dot focus(integral_(y=-oo)^(a^*) dd(cdf(y))) comment(dd(cdf(y)) = pdf(y) dd(y)) \
                     &= -q dot (1 - cdf_Y (a)) + (1-q) dot cdf_Y (a) = -q + cdf_Y (a) comment(cdf_Y (a) equiv cdf(Y = a)) \
$

At the extreme point $a = a^*$, the derivative of the risk is zero:

$ -q + cdf_Y (a^*) = 0. $

Thus, the optimal constant model $a^*$ is the $q$-quantile of the random variable $Y$:

$ a^* = cdf_Y^(-1) (q) = QQ_q [Y]. $

== Implications
We assumed that $a$ is a constant function of $bold(x)$ and derived the optimal constant model $hat(y)(bold(x)) = a^*$ that
minimizes the empirical risk (expected check-loss) $cal(R)(a)$. Notably, if we differentiate $cal(R)(a)$ with
respect to any general function $a(bold(x))$, the result remains the same.

Minimizing the check loss $cal(L)_q (epsilon)$ for a regression model $hat(y)(bold(x))$ is
equivalent to finding the $q$-quantile of the random variable $Y$. Therefore, the algorithm $a^*$ derived
from solving the minimization problem $cal(R) = Ex[cal(L)_q] -> min$
effectively predicts the $q$-quantile of $Y$.
#margin[
  Some implications of minimizing @eq-check-loss:

  #show math.equation: math.display

  - $Ex[Y] = arg min_bold(theta) sum_(bold(x) in X^ell) {y(bold(x)) - hat(y)(bold(x)|bold(theta))}^2$

  - $"med" Y = arg min_bold(theta) sum_(bold(x) in X^ell) abs(y(bold(x)) - hat(y)(bold(x)|bold(theta)))$

  - $QQ_q [Y] = arg min_bold(theta) sum_(bold(x) in X^ell) cal(L)_q (y(bold(x)) - hat(y)(bold(x)|bold(theta)))$
]

== Quantile parameter $q$
By using the check loss $cal(L)_q (epsilon)$, we can train a regression model $hat(y)_q (bold(x))$
that predicts the $q$-quantile of the random variable $Y$ given the input $bold(x)$:

$ hat(y)_q (bold(x)) = QQ_q [Y|X=bold(x)], $

where $hat(y)_q$ *depends both on hyperparameter $q$* and on the input $bold(x)$. This means that
*predictions $hat(y)_q
(bold(x))$ are different for different values of $q$*.

Likewise, the error term (residual) depends on $q$:

$ epsilon_q (bold(x)) = QQ_q [Y|X=bold(x)] - y(bold(x)) = hat(y)_q - y, $

and the check loss in @eq-check-loss is actually $cal(L)_q (epsilon) equiv cal(L)_q (epsilon_q)$.

= Expectation $Ex$ and median $QQ_(1\/2)$

== Minimization of MSE
The expectation $Ex[Y]$ is the average value of a random variable $Y$. It can be found by minimizing
quadratic loss (MSE):

#margin[
  Differentiating the quadratic loss with respect to $a$ gives:
  $
    pdv(,a) & Ex[(Y - a(X))^2 | X=bold(x)^*] \
    &= pdv(,a) Ex[Y^2 - 2 Y a(X) + a(X)^2 | X=bold(x)^*] \
    &= Ex[-2 Y + 2 a(X) | X=bold(x)^*] \
    &= -2 Ex[Y | X=bold(x)^*] + 2 a(bold(x)^*) = 0
  $

  Rearranging gives:
  $ a(bold(x)^*) = Ex[Y|X=bold(x)^*] $
]

$ Ex[Y|X = bold(x)^*] = arg min_a Ex[ (Y - a(X))^2 | X=bold(x)^* ], $ <eq-min-mse-estimator>

which holds for both conditional $Ex[Y|X]$ and unconditional $Ex[Y]$ expectations.

The algorithm $a^*$ that minimizes the average quadratic loss has the lowest MSE among all possible estimators and sometimes is called the _minimum mean squared error_ (MMSE) estimator, which is more commonly known as the _least squares_ (LS) estimator.

In other words, minimization of quadratic loss is *one of (many) possible ways* to find a good model $a^*$. During training, this model learns
how to predict conditional expectation $Ex[Y|X = bold(x)^*]$ for a given $bold(x)^*$, then we use it to predict the expectation $Ex[Y|X = bold(x)']$
for previously unseen data points $bold(x)'$.

This estimator has good theoretical guarantees (e.g., unbiasedness, minimum variance, etc. under certain conditions) and
because of that is the first choice for most regression problems.

== Minimization of MAE
An alternative estimator $a$ is obtained when instead of minimizing the quadratic term $epsilon^2$, we replace it with absolute difference $abs(epsilon)$, which
is equivalent to minimizing mean absolute error (MAE). Interestingly, this gives us the median of the random variable $Y$:

#margin[
  Given training data $(bold(x)^*, y^*) in (X, Y)^ell$, the empirical estimation according to @eq-min-mse-estimator and @eq-min-mae-estimator
  can be expressed as:
  $
    cal(R)(a) = 1 / ell dot sum_((bold(x)^*, y^*) in (X, Y)^ell) ub(cal(L)(y^* - a(bold(x)^*)), (Y - a(X))^2 | X=bold(x)^* \ "or" \ abs(Y - a(X)) | X=bold(x)^*) -> min_a.
  $
]

$
  QQ_(1\/2) [Y|X = bold(x)^*] = arg min_a Ex[ abs(Y - a(X)) | X = bold(x)^*]
$ <eq-min-mae-estimator>

Indeed, MAE is directly connected to the quantile loss $cal(L)_q (epsilon)$. For $q = 1\/2$, the quantile loss is simply the absolute
value of the error $epsilon$ (we ignore $1\/2$ factor):

$
  cal(L)_(1\/2) (epsilon) = cases(
    1\/2 dot epsilon & quad epsilon >= 0,
    -1\/2 dot epsilon & quad epsilon < 0
  ) quad = abs(epsilon) / 2
$

#margin[
  This model is also referred to as the Least Absolute Deviations (LAD) estimator.
]

For $q = 1\/2$, the quantile $QQ_(1\/2) [Y]$ corresponds to the value $y^*$ such that
$Pr[Y <= y^*] = 1\/2$; i.e., the value $y^*$ cuts the distribution of $Y$ in half. This is what the
median ($Q_(1\/2)$) of a random variable $Y$ is.

#margin[
  #figure(
    caption: [PDF of standard Normal and Laplace distributions. Laplace has heavier tails.],
    {
      let x = lq.linspace(-4, 4)
      let y-norm = x.map(x => 1 / calc.sqrt(2 * calc.pi) * calc.exp(-x * x / 2))
      let y-laplace = x.map(x => 1 / 2 * calc.exp(-calc.abs(x)))

      lq.diagram(
        width: 4cm,
        height: 3cm,
        xlabel: $x$,
        ylabel: $pdf_Y (y)$,
        lq.plot(mark: none, x, y-norm, label: $cal(N)(mu = 0, sigma = 1)$),
        lq.plot(mark: none, x, y-laplace, label: $"Laplace"(mu = 0, b = 1)$),
      )
    },
  ) <fig-laplacian-dist>
]

== Laplace distribution
Quadratic loss is derived from assuming a Gaussian distribution of $Y$. Formally, absolute loss comes from assuming a Laplace distribution of $Y$:

#margin[
  Median is more robust to outliers than mean:
  - In an ordered sample ${y_1, ..., y_ell}$, adding an outlier $y'$ shifts the mean *proportionally* to its magnitude:

    $ Delta Ex[Y] = y' / (ell+1). $

  - In the worst case an *extreme* outlier $y' << y_1$ or $y' >> y_ell$ can only
    shift the median to the adjacent element:

    $ y^((ell\/2 - 1)) - y^((ell\/2)) <= Delta QQ_(1\/2) [Y] <= y^((ell\/2 + 1)) - y^((ell\/2)). $
]


$
  pdf_Y (y) = 1 / (2b) dot e^(-abs(y - mu) / b),
$

where $b$ is the scale parameter and $mu$ is the mean. As the Laplace distribution is symmetric, the
mean $Ex[Y]$ is equal to the median $QQ_(1\/2) [Y]$.

The Laplacian distribution's heavier tails (@fig-laplacian-dist) assign higher probabilities to extreme values, making models based on it more robust to outliers.

== Likelihood
Assuming observations $(bold(x)^*, y^*)$ are i.i.d. and algorithm $a$ predicts the conditional mean
$mu = Ex[Y|X=bold(x)^*] = QQ_(1\/2) [Y|X=bold(x)^*]$, the likelihood function is given by:

$
  LL = product_((bold(x)^*, y^*) in (X, Y)^ell) f_Y (y^*|bold(x)^*)
  = product_((bold(x)^*, y^*) in (X, Y)^ell) 1 / (2b) dot e^(-abs(y^* - a(bold(x)^*)) \/ b),
$

Maximizing the likelihood function is equivalent to minimizing MAE:

$
  log LL &= sum_((bold(x)^*, y^*) in (X, Y)^ell) {-log 2b - abs(y^* - a(bold(x)^*)) / b} \
  &= -1 / b ub(sum_((bold(x)^*, y^*) in (X, Y)^ell) abs(y^* - a(bold(x)^*)), ell dot "MAE") - ub(ell dot log 2b, const) -> max_a
$

== Measures of central tendency
Expectation $Ex[Y]$ and median $QQ_(1\/2) [Y]$ are two distinct measures of central tendency for a random variable $Y$. In some cases, they
are equal, but in general they are not. This distinction leads to two different regression models:

#margin[
  Quantile regression is not limited to $q = 1\/2$; we can construct a regression model for any
  conditional quantile $QQ_q [Y|X]$ where
  $q$ is a hyperparameter
]

#grid(
  columns: 2,
  [
    - In ordinary least squares (LS), we predict the expected value of a random variable:
    $
      hat(y)(bold(x)) = Ex[Y|X = bold(x)].
    $
  ],
  [
    - In median regression, we build a model that predicts the conditional median:
    $ hat(y) (bold(x)) = QQ_(1\/2) [Y|X=bold(x)]. $
  ],
)

= Quantile regression

#margin[
  Quantile regression was introduced by Roger Koenker and Gilbert Bassett in @koenker1978regression.

  For a short overview and examples see @koenker2001quantile and @Koenker2005quantile for details.
]

== Probabilistic model
Suppose the distribution of the data $(bold(x), y)$ is modeled as a joint distribution $pdf(bold(x), y)$.
Our goal is to predict the quantile $QQ_q [Y] = fun(bold(x))$ for a given $bold(x)$, i.e., to
predict the conditional quantile $QQ_q [Y|X=bold(x)]$.

== Optimization problem
The empirical risk is defined as the average quantile loss @eq-check-loss over the distribution $pdf(bold(x), y)$.
By minimizing the empirical risk, we can find the optimal model $a^*(bold(x))$ that predicts the
quantile $QQ_q [Y|X=bold(x)]$:

$
  a^*(bold(x)) = arg min_a ub(Ex_((bold(x), y) ~ pdf(bold(x), y)) [ cal(L)_q (y - a(bold(x))) ], cal(R)(a)).
$

== Practical reformulation
From the theoretical expression of the empirical risk, we can derive a practical reformulation of
the quantile regression problem.

#margin[
  In LS regression, only one prediction $hat(y)(bold(x)) = Ex[Y|X=bold(x)]$ exists, with a single residual $epsilon(bold(x)) := y(bold(x)) - hat(y)(bold(x))$.

  In quantile regression, $hat(y)_q (bold(x))$ is parameterized by $q$, producing multiple possible predictions $QQ_q [Y|X=bold(x)]$ for the same random variable, with corresponding residuals $hat(epsilon)_q (bold(x)^*) := hat(y)_q (bold(x)^*) - y(bold(x)^*)$
]

For a specific pair $(bold(x)^*, y^*)$ drawn from the joint distribution $pdf(bold(x), y)$ represented
by a training set $(X, Y)^ell$, the empirical risk can be expressed via the check loss
@eq-check-loss:

$
  cal(R)(a) = 1 / ell dot sum_((bold(x)^*, y^*) in (X, Y)^ell) cal(L)_q (y^* - a(bold(x)^*)) -> min_a.
$

The model $a(bold(x)) equiv a(bold(x)|bold(theta); q)$ can be any general regression model
supporting custom loss functions or the quantile loss $cal(L)_q$ specifically.

#let quantile-model-plot(file) = {
  let data = lq.load-txt(read(file), header: true)
  let x = data.remove("x")
  let y = data.remove("y")

  let label-fn(col) = if (col == "mean") { $Ex$ } else { $QQ_#col$ }

  lq.diagram(
    width: 4cm,
    height: 3cm,
    legend: (position: right + bottom),
    lq.scatter(x, y, size: 3pt, stroke: none, color: ghost-color),
    ..data.keys().map(col => lq.plot(mark: none, x, data.at(col), label: label-fn(col))),
  )
  // TODO: add link to code
  // FIX: legend overlaps with the plot
}

#grid(
  columns: (1fr, 1fr),
  uplift[== Linear quantile regression
    The conditional quantile $QQ_q [Y|X]$ can be modeled as a linear function of predictors $bold(x)$:

    #margin(
      figure(
        caption: [Linear quantile regression for non-normaly distributed noise],
        quantile-model-plot("linear/out.csv"),
      ),
    )

    $
      QQ_q [Y|X = bold(x)] = bra bold(x), bold(beta) ket, quad beta_j equiv beta_j (q),
    $ <eq-linear-quantile-regression>

    where $bold(beta)(q)$ is a vector of regression coefficients, and $beta_j (q) = beta_(j|q) in RR$ are
    regression coefficients for the feature $bold(x)^j$ and a _predefined hyperparameter_ $q$.
    Coefficients $beta_j (q)$ are estimated by minimizing the empirical risk:

    $
      cal(R)(bold(beta)) &= 1 / ell dot sum_(bold(x) in X^ell) cal(L)_q (y(bold(x)) - bra bold(x), bold(beta) ket) \
      &-> min_(bold(beta)).
    $
  ],
  uplift[== Neural quantile regression
    Neural networks inherently support custom loss functions and can model conditional quantiles $QQ_q [Y|X]$ as
    well (@fig-neural-quantile-regression). A model predicting conditional quantiles $QQ_q [Y|X]$ must
    be trained with a quantile loss, which can be easily implemented:

    #margin[
      #figure(
        caption: [Quantile regression performed by a neural network],
        quantile-model-plot("nn/out.csv"),
      ) <fig-neural-quantile-regression>
    ]

    ```python
    class QuantileLoss(L.LightningModule):
        def __init__(self, q: float):
            super().__init__()
            self.q = q

        def forward(self, y_pred, y_true):
            epsilon = y_true - y_pred
            return T.where(
                epsilon >= 0,
                self.q * epsilon,
                (self.q - 1) * epsilon,
            ).mean()
    ```
  ],

  uplift[== Gradient boosting quantile regression
    Quantile loss @eq-check-loss is differentiable if $epsilon != 0$:

    #margin[#figure(
        caption: [Quantile regression performed by a gradient boosting model],
        quantile-model-plot("boosting/out.csv"),
      ) <fig-boosting-quantile-regression>
    ]

    $
      pdv(, epsilon) cal(L)_q (epsilon) = cases(q & quad epsilon > 0, -(1-q) & quad epsilon < 0
      ) med ,
    $

    thus, gradient boosting can approximate the quantile function $QQ_q [Y|X]$ to handle non-linear
    dependencies between features and quantiles (@fig-boosting-quantile-regression).],
  uplift[== Ensemble models
    Multiple base algorithms $a_t (bold(x))$ can be combined to create an ensemble model

    $
      A(bold(x)) = 1 / T dot sum_(t=1)^T a_t (bold(x)).
    $

    If each base algorithm $a_t (bold(x))$ is trained to predict quantiles $QQ_q [Y|X]$, the ensemble $A(bold(x))$ will
    estimate the expectation of the quantile $Ex [QQ_q [Y|X]]$.],
)

= Convergence and reliability of quantile regression parameters

== Linear quantile regression
For linear quantile regression @eq-linear-quantile-regression, the conditional quantile $QQ_q [Y|X]$ is
modeled as a linear function of predictors $bold(x)$. The theoretical properties of linear quantile
parameters $hat(bold(beta))(q)$ such as convergence and variance can be derived, though the analysis
is more complex than for traditional Gaussian regression.

== Parameter expectation
All regression coefficients $beta_j = beta_j (q)$ are functions of $q$. Under appropriate conditions
(independent observations with finite second moments), the asymptotic distribution of the quantile
regression estimator $hat(bold(beta))(q)$ is unbiased:

$ hat(bold(beta))(q) -> bold(beta)(q), $

i.e., theoretically the estimator $hat(bold(beta))(q)$ converges to the expected value of the
parameter $bold(beta)(q)$ as the sample size $ell$ approaches infinity:

$ hat(bold(beta))(q) -> Ex[bold(beta)(q)]). $ <eq-quantile-linear-parameter-expectation>

== Parameter variance
The estimator $hat(bold(beta))(q)$ is asymptotically normally distributed with variance#margin[and mean $bold(beta)(q) = Ex[bold(beta)(q)]$ according to @eq-quantile-linear-parameter-expectation]

$
  Var[bold(beta)(q)] -> ub(1/ell, "I") dot ub(q dot (1-q), "II") dot ub(D^(-1) Omega D^(-1), "III").
$ <eq-quantile-linear-parameter-variance>

The variance in @eq-quantile-linear-parameter-variance depends on three terms:

+ The 1st multiplier determines the convergence rate of the estimator $hat(bold(beta))(q)$ as a
  function of the sample size $ell$; the larger the sample size, the smaller the variance.

+ The 2nd multiplier depends on the quantile $q$. As $q$ approaches the tails (0 or 1), this term
  decreases, which would seemingly lower the variance. It reduces variance if isolated, however, this
  is not the primary contributor to overall variance.

  #margin(
    figure(
      caption: [
        $q dot (1-q)$ term in @eq-quantile-linear-parameter-variance reaches its maximum at $q = 0.5$
      ],
      {
        let x = lq.linspace(0, 1)
        lq.diagram(
          width: 4cm,
          height: 2cm,
          xlabel: $q$,
          ylabel: $q dot (1-q)$,
          lq.plot(mark: none, x, x.map(q => q * (1 - q))),
        )
      },
    ),
  )

  + The 3rd multiplier is the sandwich variance estimator, which depends on both the estimated
    parameters $hat(bold(beta))(q)$ and the robust variance matrix $Omega$. Typical formulations
    include:

    $
      D(hat(y)_q) = 1 / ell dot sum_(bold(x) in X^ell) hat(f)_(Y|X) (hat(y)_q (bold(x))) dot bold(x) bold(x)^Tr
    $

    $
      hat(Omega) = 1 / ell dot sum_(bold(x) in X^ell) (q - Ind(y(bold(x)) <= hat(y)_q (bold(x)) )) dot bold(x) bold(x)^Tr
    $

#margin[While $q dot (1-q)$ decreases near the tails, the sandwich term $D^(-1) Omega D^(-1)$ becomes poorly
  estimated and tends to dominate.
]

Consequently, the *variance of estimated parameters $hat(bold(beta))(q)$ increases as $q$
approaches 0 or 1*. In practice, predictions near the median are typically more precise, while
predictions for extreme quantiles (e.g., 0.01 or 0.99) are less reliable.

== Bad statistical guarantee
While ordinary least squares (OLS) estimates benefit from the Gauss-Markov theorem, which
establishes OLS as the best linear unbiased estimator (BLUE) under classical assumptions, quantile
regression follows different asymptotic properties.

#margin[
  The variance of the quantile regression estimator is larger than that of OLS, especially for extreme
  quantiles
]

Quantile regression estimators remain unbiased and consistent, but their variance behavior is more
complex. As shown in equation @eq-quantile-linear-parameter-variance, the variance depends on both
the quantile level $q$ and the underlying data distribution through the sandwich estimator term $D^(-1) Omega D^(-1)$.

In practice, quantile regression estimates exhibit higher statistical variability than OLS
estimates, particularly for extreme quantiles (e.g., $q < 0.1$ or $q > 0.9$). This occurs because:

1. The sparsity of data in the tails leads to less reliable sandwich term estimation
2. The conditional density at extreme quantiles becomes more difficult to estimate accurately
3. The effective sample size for determining extreme quantiles is effectively reduced

This statistical efficiency trade-off is a necessary cost of gaining robustness to outliers and
insights into the complete conditional distribution.


= Robustness of quantile regression

== Non-normality (skew, heavy tails, multimodality)
Quantile regression models *conditional quantiles*, capturing skewed or heavy-tailed distributions
*without relying on normality assumptions*. OLS assumes normality and may produce misleading results
when this assumption is violated.

Given uniform $epsilon ~ cal(U)(-4, 4)$, look (@fig-quantile-non-normality) at three different models with non-normal additive
noises: $3 epsilon$, $epsilon^3$, and $epsilon_+ + 4 epsilon_-$.

#margin[
  - $epsilon_+ := max{0, epsilon}$
  - $epsilon_- := min{epsilon, 0}$
]

#figure({
  let data = lq.load-txt(read("robustness/out.csv"), header: true)
  let x = data.remove("x")

  let plot(y-name, y-label: $y$) = lq.diagram(
    width: 3cm,
    height: 3cm,
    xlabel: $x$,
    ylabel: y-label,
    ylim: (-10, 10),
    xlim: (-10, 10),
    legend: (position: bottom + right),
    lq.scatter(x, data.at(y-name), size: 2pt),
    lq.plot(x, data.at(y-name + "_pred_ls"), stroke: 2pt, mark: none, label: $EE$),
    lq.plot(x, data.at(y-name + "_pred_qr"), stroke: 2pt, mark: none, label: $QQ_(1\/2)$),
    lq.line((-10, -10), (10, 10)),
  )

  grid(
    columns: (1fr, 1fr, 1fr),
    plot("y_uniform_1", y-label: $y = x + 3 epsilon$),
    plot("y_uniform_2", y-label: $y = x + epsilon^3$),
    plot(
      "y_uniform_3",
      y-label: $y = x + epsilon_+ + 4 epsilon_-$,
    ),
  )
}) <fig-quantile-non-normality>


#{
  let data = lq.load-txt(read("distributions/out.csv"), header: true)
  let x = data.remove("x")
  let y = data.remove("y")

  let plot(col-name, residual: false) = lq.diagram(
    width: 3cm,
    height: 3cm,
    xlabel: $x$,
    ylabel: if not residual { $y = f(x) + epsilon_#col-name.split("_").at(1)$ } else {
      $epsilon_#col-name.split("_").at(1)$
    },
    ylim: (-4, 4),
    if not residual {
      lq.scatter(x, data.at(col-name).zip(y).map(el => el.sum()), size: 2pt)
    } else {
      lq.scatter(x, data.at(col-name), size: 2pt)
    },
    if not residual {
      lq.plot(x, y, stroke: 2pt, mark: none)
    },
  )

  grid(
    columns: (1fr, 1fr, 1fr),
    ..data.keys().map(col => plot(col)),
    ..data.keys().map(col => plot(col, residual: true)), // TODO: switch to KDE or histogram
  )
}

== Heteroscedasticity
Quantile regression does not assume homoscedasticity (constant variance). Instead, it models
different parts of the conditional distribution independently, allowing for varying spread (e.g.,
wider or narrower intervals) across predictors. OLS assumes homoscedasticity (or equal weight of all
observations).

== Robustness to outliers and noise
By focusing on quantiles rather than the mean, quantile regression reduces sensitivity to random
noise and outliers, emphasizing specific distributional trends. Quantile regression also *does not
assume any specific noise distribution*. In OLS, a few outliers can have a pronounced effect on
parameter estimates.

== Censoring
Censoring arises when the response variable $y$ is not fully observed. For instance, in clinical trials, the exact value of $y$ may be unavailable for some patients. If a patient exits a longitudinal study, we only know they survived up to time $y$, but their true survival time might be much longer (@fig-censoring-plot).

#margin[

  #figure(
    caption: [
      Time-to-death plot from the start of a clinical trial. Circles represent patients whose exact time-to-death is known,
      while crosses represent patients who withdrew from the study.
    ],
    lq.diagram(
      xlabel: [time $y$, years],
      yaxis: (ticks: none),
      width: 3cm,
      height: 2cm,
      margin: 20%,
      lq.hstem(
        (2.3, 3.2, 6.2, 7.0, 8, 9),
        (8, 7, 5, 4, 2, 1),
      ),
      lq.hstem(
        (5.1, 7.5),
        (6, 3),
        mark: "x",
      ),
    ),
  ) <fig-censoring-plot>
]

In standard Gaussian regression, censoring results in a bias in estimates since observations are truncated. Quantile regression allows to model different quantiles $q$ of the distribution: some areas of the distribution may not be affected by
censoring, while others are. By choosing appropriate quantiles, we can obtain reliable estimates even in the presence of censoring.

However, in general it's better to experiment with different losses and other models, that work with
censoring implicitly.

== Invariance
Quantile regression is *invariant to monotonic transformations* of $y$ like logarithm or square
root. In OLS this is not the case, although transformations are sometimes used to normalize data.

= Interpretation of linear quantile coefficients $hat(beta)_j (q)$

#margin(
  // TODO: add subfigure reference labels to the plot
  multi-figure(
    caption: [
      Quantile regression coefficients for ACTG 320 dataset
    ],
    ..{
      let data = lq.load-txt(read("aids/out.csv"), header: true)
      let x = data.remove("quantile")

      let plot-coeff(col) = {
        let y = data.at(col)
        let lim = calc.max(..y.map(calc.abs))

        let diagram = lq.diagram(
          ylim: if (col == "intercept") { auto } else { (-lim, lim) },
          xlim: (0, 1),
          ylabel: $hat(beta)_#raw(col)$,
          xlabel: [quantile $q$],
          width: 4cm,
          height: 2.5cm,
          margin: 15%,
          lq.plot(x, y, mark-size: 2pt),
        )

        [#figure(diagram) #label("fig-aids-" + col.replace("_", "-"))]
      }

      data.keys().map(col => plot-coeff(col))
    },
    label: <fig-aids-quantile-plot>,
  ),
)

== Impact on target variable
Quantile regression coefficients $beta_j (q)$ represent the impact of a unit change in predictor $bold(x)^j$ on the response variable at specific quantiles. Unlike OLS coefficients, they capture how features influence different parts of the target distribution.

By examining how $hat(beta)_j (q)$ varies across quantile levels, we can guess how predictors affect various segments of the conditional distribution, revealing effects that are not directly observable in standard regression.

== Data
The ACTG 320 clinical trial, initiated in 1997 by Merck, was designed to evaluate the effectiveness of the
antiretroviral drug indinavir when used in a triple-drug regimen compared to a standard two-drug
treatment for HIV patients.

#figure(
  caption: [
    ACTG 320 dataset features (simplified)
  ],
  table(
    columns: (auto, 1fr),
    [Variable], [Description],
    [`time` \ (target)],
    [Follow-up time to AIDS progression or death (in days). Represents the time from enrollment to the
      event (end of study or death).],

    [`age`], [Age of the patient at the time of enrollment (in years).],
    [`cd4_cell_count`], [Baseline CD4 T-cell count (cells/mL), a key indicator of immune function.],
    [`race_*`], [Indicator variables representing the patient's race.],
    [`group_*`], [Indicator variables representing the treatment group.],
  ),
) <tab-aids-320-features>

The associated dataset contains aprox. 1,150 records of HIV-infected patients who were
randomized to receive either the novel triple-drug regimen or the conventional two-drug therapy.

// TODO: Add reference to https://search.r-project.org/CRAN/refmans/GLDreg/html/actg.html

== Quantile regression
The target variable is `time`, representing the follow-up duration. Linear quantile regression

$ QQ_q [Y|X] = sum_j beta_j dot bold(x)^j, quad beta_j equiv beta_j (q) $

was used to estimate the impact of various linear predictors $bold(x)^j$ from @tab-aids-320-features
on the time $y$ to AIDS progression or death.

Quantile regression coefficients $beta_j (q)$ as functions of quantile $q$ are plotted in
@fig-aids-quantile-plot. Low $q$ values represent individuals who progressed to AIDS or died quickly, while high $q$ values
correspond to individuals with longer survival times.

#margin[
  Check @koenker2001quantile for more examples.
]

== Baseline estimate
Baseline survival time is estimated by the model intercept (@fig-aids-intercept), e.g., median intercept
$beta_"intercept" (q = 1\/2)$ is approximately 240 days. Note that the intercept would be the median
survival time if all other predictors were zero, in our case, they are not.

== Reliability of coefficients
For $q approx 0.5$, the estimates are most reliable and often close to the OLS estimates. Extreme
quantiles are estimated at tails where data is sparse, leading to higher variance $Var [hat(beta)_j]$ and
less reliable estimates, as seen in the fluctuations in @fig-aids-race-hispanic at both tails.
Quantiles $q < 0.1$ and $q > 0.9$ were not estimated at all.


== Sign of quantile regression coefficients $beta_j (q)$
The sign of $beta_j (q)$ reflects the predictor $bold(x)^j$'s impact on survival time $y$ at the $q$-quantile, i.e.,
$QQ_q [Y] prop beta_j (q) dot Delta bold(x)^j$ at the $q$-quantile.

Consistently positive $beta_j (q)$ across all $q$ suggest that the predictor $bold(x)^j$ has only
positive contributions to survival time $y$ for all individuals. For the indinavir group
(@fig-aids-group-indinavir), the positive impact (in days) is greatest for short-survived patients
(low $q$) and decreases for long-lived patients (high $q$).

Likewise, consistently negative $beta_j (q)$ across all $q$ suggest that the predictor $bold(x)^j$ has only
negative contributions to survival time $y$ for all quantiles. AIDS patients generally have lower
CD4 cell counts than healthy individuals, and the lower the CD4 cell count, the more pronounced its
negative contribution (@fig-aids-cd4-cell-count) to survival time $y$.

= Goodness-of-fit

== Bad metrics
Classical metrics (e.g., MAE, MSE, $R^2$) evaluate predictions based on their distribution around the mean $Ex[Y]$. However, quantile regression focuses on other distribution properties, intentionally ignoring the mean. As a result, classical metrics are not suitable for evaluating quantile regression models.

#margin[
  In fact, MAE is equivalent to the mean quantile loss for $q = 1\/2$, making it suitable for *median regression specifically*
]

#margin[
  For a model $a(bold(x)) equiv hat(y)(bold(x))$:

  - The _total sum of squares_ is:

  $
    TSS := sum_(bold(x) in X^ell) (y(bold(x)) - Ex[Y|bold(x)])^2
  $

  - The _explained sum of squares_ is:

  $
    ESS := sum_(bold(x) in X^ell) (hat(y)(bold(x)) - Ex[Y|bold(x)])^2
  $

  - The _residual sum of squares_ is:

  $
    RSS := sum_(bold(x) in X^ell) (y(bold(x)) - hat(y)(bold(x)))^2
  $

  For unbiased models, $TSS = ESS + RSS$, which is used to derive @eq-r2-quantile.
]

== Mean quantile loss
The simplest approach to evaluate quantile regression models is to use the quantile loss $cal(L)_q$ directly:

$
  bra cal(L)_q ket := 1 / ell dot sum_((bold(x), y) in (X, Y)^ell) cal(L)_q (y - hat(y)_q (bold(x))),
$ <eq-mean-quantile-loss>

where $hat(y)_q (bold(x))$ is the quantile regression model.

For two quantile regression models $hat(y)_q$ and $hat(y)'_q'$ (e.g., for different quantiles, regularization, or features), the model with the lower quantile loss better fits the data and is preferred. In `sklearn`, this metric is implemented as `sklearn.metrics.mean_pinball_loss`.

== $R^1$ metric
Another approach involves metrics specifically designed for quantile regression. Classical $R^2$ measures the proportion of variance explained by the model:

$
  R^2 = ESS / TSS :> 1 - RSS / TSS,
$ <eq-r2-quantile>

where $RSS$ is the sum of squared residuals between the predicted and actual values, and $TSS$ is the squared difference between the actual values $y(bold(x))$ and the mean $macron(y) = Ex[Y]$. $TSS$ can be viewed as $RSS$ for a very simple constant model $hat(y) (bold(x)) = macron(y)$:

$ R^2 = 1 - (RSS[hat(y)]) / (RSS[macron(y)]), $ <eq-r2-baseline-model>

where $RSS[hat(y)]$ and $RSS[macron(y)]$ are the residual sum of squares for the actual (proposed) model $hat(y)$ and the mean constant (baseline) model $macron(y)$, respectively.

#note[
  The choice of the baseline model is *arbitrary*, so a pseudo-$R^2$ metric can be used to compare $RSS$ of any two arbitrary models $hat(y)$ and $hat(y)'$.
]

For quantile regression, a similar metric can be defined (@koenker1999goodness, eq. 7). For two quantile regression models $hat(y)_q$ and $hat(y)'_q'$ and corresponding mean quantile losses $bra cal(L)_q [hat(y)_q] ket$ and $bra cal(L)'_q' [hat(y)'_q'] ket$ computed via @eq-mean-quantile-loss, the analog of $R^2$ is:

$
  R^1 := 1 - ( bra cal(L)_q [hat(y)_q] ket ) / ( bra cal(L)'_q' [hat(y)'_q'] ket ),
$ <eq-r1-quantile>

where $hat(y)_q$ and $hat(y)'_q'$ are the proposed and baseline models, respectively. Usually, models are compared for the same quantile $q$.

A straightforward choice for the baseline model $hat(y)'_q$ is the empirical quantile value $QQ_q [Y]$ calculated from the training set. The difference in the upper index arises because in $R^2$, quadratic units ($TSS$, $RSS$, and $ESS$) are used, while in $R^1$, linear units (quantile loss) are used, as seen in @eq-check-loss.

Like the general definition of $R^2$, which is not bound to $[0, 1]$ and can be negative, the $R^1$ metric can also be negative if the model $hat(y)_q$ is worse than the baseline model $hat(y)'_q$. In `sklearn`, this metric is implemented as `sklearn.metrics.d2_pinball_score`.

== Ordered metrics
A quantile model can be evaluated on how well it preserves the order. This is particularly important for risk modeling applications and ranking.
For example, if a patient $bold(x)$ died at $y(bold(x))$ and another at $y(bold(x)')$, where $y(bold(x)) < y(bold(x)')$, the model should predict $hat(y)_q (bold(x)) < hat(y)_q (bold(x)')$.

#margin[
  In @eq-c-index, the numerator counts the number of observation pairs where the model predicts the same order as the actual order.
  The denominator counts the total number of comparable observation pairs. Equation @eq-c-index can also be extended to handle censored data.
]

#margin[
  The calculation can be optimized by summing over unique pairs $i = 1..ell$ and $j = i+1..ell$, reducing redundancy.
]

The proportion of correctly ordered pairs is measured by the _concordance index_ (C-index):
$
  C := (sum_(i=1)^ell sum_(j=1)^ell Ind(y(bold(x)_i) < y(bold(x)_j)) dot Ind(hat(y)_q (bold(x)_i) < hat(y)_q (bold(x)_j))) / (sum_(i=1)^ell sum_(j=1)^ell Ind(y(bold(x)_i) < y(bold(x)_j))),
$ <eq-c-index>

The C-index ranges from 0.5 (random predictions) to 1 (perfect predictions).


= Practical considerations

== Targets
Median is sometimes more interpretable and a better measure of centrality than
the mean, particularly for skewed or multimodal data:

- Median salary or house price characterizes the central tendency of a distribution better than
  the mean, which can be skewed by extreme values.


== Parameters

- Coefficients $beta$ in linear quantile regression are noisier than in OLS and depend on quantile $q$,
  making them harder to interpret. The Gauss-Markov theorem ensuring convergence and variance in OLS
  does not apply to quantile regression.

- Exact values of $beta$ in OLS are interpretable, but in quantile regression, they are generally not.
  In simple cases, they can be close to OLS coefficients and interpretable. However, when quantile
  regression is applied to transformed data (e.g., $log(y)$), coefficients remain invariant, but their
  contribution to $y' = log(y)$ becomes less obvious. For skewed data where OLS fails, quantile
  regression coefficients differ significantly from OLS but may still be interpretable.

== Computational complexity
Quantile regression lacks a universal analytical solution and is typically solved numerically.
The quantile loss function @eq-check-loss combines two linear functions separated at $epsilon = 0$.
Residuals can be decomposed into positive and negative parts:

$
  epsilon = epsilon^+ - epsilon^-, quad "where" quad cases(epsilon^+ := max {0, epsilon}, epsilon^- := -min{0, epsilon}),
$

Using this decomposition, the quantile loss can be expressed as:

$
  cal(L)_q (epsilon) = q dot epsilon^+ + (1-q) dot epsilon^-.
$

This formulation leads to a constrained linear programming problem [@Koenker2018handbook, p.282]:

$
  & 1 / ell sum_(i=1)^ell {q dot epsilon_i^+ + (1-q) dot epsilon_i^-} -> min_(bold(epsilon)^+, bold(epsilon)^-) \
  "s.t." quad & y_i - hat(y)_i = epsilon_i^+ - epsilon_i^-, quad i = 1..ell, \
  & epsilon_i^+ >= 0, quad epsilon_i^- >= 0, quad i = 1..ell.
$

Solving this optimization problem is computationally more intensive than OLS's closed-form solution, particularly for large datasets or when estimating multiple quantiles simultaneously.

== Extreme quantiles
Estimates for extreme quantiles (e.g., $q = 0.01$ or $q = 0.99$) are often less reliable due to
sparse data in distribution tails, resulting in higher variance as shown in the parameter
convergence section.

== Complete picture of conditional distributions
Quantile regression allows modeling multiple quantiles, providing a comprehensive view of how
predictors affect the entire conditional distribution of the response, not just its center. This
reveals heterogeneous effects that OLS cannot capture.

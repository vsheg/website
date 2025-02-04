#import "../../defs.typ": *

Conditional probability is a foundational concept in probability theory that can be
challenging to visualize. This post uses a geometric example of a satellite’s position to
illustrate both unconditional and conditional probability densities.

== Statistical model: geostationary satellite position

Consider a satellite with mean Earth coordinates $(macron(x), macron(y))$. A radio station
at the origin $(0, 0)$ measures the distance $d$ to the satellite.

The satellite's location is modeled by an _uncorrelated bivariate Gaussian distribution_:

$
  f_(X,Y) (x, y) = 1/(2pi sigma_X sigma_Y) exp{-(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y)^2)^2 / (2 sigma_Y^2)}
$

The measured distance $d$ follows a _conditional univariate distribution_:

$
  f(D = d | X = x, Y = y) = 1 / (sqrt(2 pi) sigma_D) exp {-(d - sqrt(x^2 + y^2))^2 / (2 sigma^2_D)}
$

*N.B.* The unconditional distribution $f(D=d)$ can be obtained by substituting $x=macron(x)$ and $y=macron(y)$

== Conditional probability density

Using Bayes theorem:

$
  Pr[A|B] = (Pr[A] dot Pr[B|A]) / Pr[B],
$

the conditional probability density for the satellite being at $(x, y)$ given a measured
distance $d^*$ is

$
  f(X = x, Y = y | D = d^*)
    &= (f(X = x, Y = y) dot f(D = d^*|X = x, Y = y)) / f(D = d^*) \
    &= underbrace(1 / f(D = d^*), const) f(X = x, Y = y) dot f(D = d^*|X = x, Y = y).
$

By substituting the bivariate and univariate Gaussian distributions, we obtain:

$   &= const dot 1/(2pi sigma_X sigma_Y) exp{-(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y)^2)^2 / (2 sigma_Y^2)} dot 1 / (sqrt(2 pi) sigma_D) exp {-(d^* - sqrt(x^2 + y^2))^2 / (2 sigma^2_D)} \
  &= underbrace(const/(sqrt(2pi)^3 sigma_X sigma_Y sigma_D), const_2) exp{ -(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y)^2)^2 / (2 sigma_Y^2) -(d - sqrt(x^2 + y^2))^2 / (2 sigma^2_D) }. $

== Marginalization

The marginal probability constant $const(d)$ can be obtained by integrating the
conditional probability density:

$
  const(d) = f(D = d)
    &= integral_(-oo)^(+oo) integral_(-oo)^(+oo) f(D = d, X=x, Y=y) dd(x) dd(y) \
    &= integral_(-oo)^(+oo) integral_(-oo)^(+oo) f(D = d|X=x, Y=y) dot f_(X,Y)(x, y) dd(x) dd(y)
$

This integral contains multiple exponential terms that make it challenging to solve
analytically or compute numerically due to rapid growth and potential numerical overflow.

However, the pre-exponential constants can be omitted because:

- For finding the most probable location through density maximization, the constant only
  affects the scale but not the location of the maximum.

- For visualizing the probability density shape, exact values are not required, so constants
  can be omitted.

== Visualization

Now both the unconditional $f(X, Y)$ and conditional $f(X, Y | D = d^*)$ probability
densities can be visualized using the derived expressions.

// embed code
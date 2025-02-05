#import "../../defs.typ": *

Conditional probability is a fundamental concept in probability theory; however, its
interpretation is not always intuitive. Here, an easy-to-visualize example is used to
illustrate the concept.

== Model: radar tracking UFO

Consider a UFO with mean Earth coordinates $(macron(x), macron(y))$. A radar at the origin $(0, 0)$ measures
the distance $d$ to the UFO.

The UFO's location is modeled by an _uncorrelated bivariate Gaussian distribution_:

$
  f_(X,Y) (x, y) &equiv f_(X,Y) (X = x,Y = y) \
                 &= 1/(2pi sigma_X sigma_Y) exp{-(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y))^2 / (2 sigma_Y^2)}.
$

The measured distance $d$ follows a _conditional univariate distribution_:

$
  f_D (d | x^*, y^*) & equiv f_D (D = d | X = x^*, Y = y^*) \
                     &= 1 / (sqrt(2 pi) sigma_D) exp {-(d - sqrt((x^*)^2 + (y^*)^2))^2 / (2 sigma^2_D)},
$

where $(x^*, y^*)$ is the exact UFO position and $sigma_D$ is the standard deviation (with
variance $sigma_D^2$) of the distance measurement.

*N.B.* The unconditional distribution $f_D (D=d)$ can be obtained by substituting $x^*=macron(x)$ and $y^*=macron(y)$

== Conditional probability density

Using Bayes theorem:

$
  Pr[A | B] = (Pr[A] dot Pr[B | A]) / (Pr[B]),
$

the conditional probability density of the UFO being at $(x, y)$ given a measured distance $d^*$ is

$
  f_(X,Y) (X = x, Y = y | D = d^*)
    &= (f_(X,Y) (X = x, Y = y) dot f_D ( D = d^* | X = x, Y = y)) / (f_D ( D = d^*)) \
    &= underbrace(1 / (f_D ( D = d^*)), const) dot f_(X,Y) (x, y) dot f_D (d^* | x, y),
$

where $const equiv const(d^*)$ is the normalization constant for fixed $d^*$. By
substituting the bivariate and univariate Gaussian distributions:

$   &= const dot 1/(2pi sigma_X sigma_Y) exp{-(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y))^2 / (2 sigma_Y^2)} dot 1 / (sqrt(2 pi) sigma_D) exp {-(d^* - sqrt(x^2 + y^2))^2 / (2 sigma^2_D)} \
  &= underbrace(const/(sqrt(2pi)^3 sigma_X sigma_Y sigma_D), const') exp{ -(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y))^2 / (2 sigma_Y^2) -(d^* - sqrt(x^2 + y^2))^2 / (2 sigma^2_D) }, $

where $const' equiv const/(sqrt(2pi)^3 sigma_X sigma_Y sigma_D)$ is a new normalization
constant for given distribution parameters $sigma_X$, $sigma_Y$, and $sigma_D$.

== Marginalization

The normalization constant $const(d^*)$ can be calculated by integrating the conditional
probability density:

$
  1 / (const(d^*))
    &= f_D (D = d^*) \
    &= integral_(-oo)^(+oo) integral_(-oo)^(+oo) f_(X, Y, D)(X=x, Y=y, D = d^*) dd(x) dd(y) \
    &= integral_(-oo)^(+oo) integral_(-oo)^(+oo) f_D (d^* | x, y) dot f_(X,Y)(x, y) dd(x) dd(y).
$

This integral contains different exponential terms, making it challenging to solve
analytically or compute numerically due to rapid growth and potential numerical overflow.

However, the pre-exponential constant $const'$ can be omitted for visualization since it
doesn't affect density shape or maximum position.

== Visualization

Now both the unconditional $f_(X,Y) (x, y)$ and conditional $f_(X,Y) (x, y | d^*)$
densities can be visualized using the derived expressions.

The radar at position $(0, 0)$ measures distance $d$ (depicted as a circle) to the UFO.
The UFO is hovering around the mean position $(4, 4)$ with s.d. $sigma_X = sigma_Y = 3$,
corresponding probability density is shown by the color gradient. The measured distance $d$ constrains
the probable UFO location to a circular arc in the conditional density.

// embed code
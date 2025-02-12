#import "../../defs.typ": *

Conditional probability is a fundamental concept in probability theory; however, its
interpretation is not always intuitive. Here, an easy-to-visualize example is used to
illustrate the concept.

== Model: non-directional radar tracking UFO

A UFO hovers above a field with average position $(macron(x), macron(y))$ and standard
deviations $sigma_X$, $sigma_Y$. At the field's center $(0, 0)$, a #link("https://en.wikipedia.org/wiki/Non-directional_beacon", "non-directional radar") measures
only the distance $d$ to the UFO with standard deviation $sigma_D$, but cannot determine
its direction or exact position.

#image("image.jpg", width: 30%)

The UFO's location is modeled by an _uncorrelated bivariate Gaussian distribution_ of two
random variables $X$ and $Y$:

$
  f_(X,Y) (x, y) &equiv f_(X,Y) (X = x,Y = y) \
                 &= 1/(2pi sigma_X sigma_Y) exp{-(x - macron(x))^2 / (2 sigma_X^2) -(y - macron(y))^2 / (2 sigma_Y^2)}.
$

The distance $d$ from the radar antenna to the UFO follows a _conditional univariate Gaussian distribution_ of
a single random variable $D$ given the UFO's position $(X, Y)$:

$
  f_D (d | x^*, y^*) & equiv f_D (D = d | X = x^*, Y = y^*) \
                     &= 1 / (sqrt(2 pi) sigma_D) exp {-(d - sqrt((x^*)^2 + (y^*)^2))^2 / (2 sigma^2_D)},
$

where $(x^*, y^*)$ is a specific UFO position, and $sqrt((x^*)^2 + (y^*)^2)$ is the exact
distance (unknown) from the radar to the UFO.

*N.B.* The unconditional distribution $f_D (D=d)$ can be obtained by substituting $x^*=macron(x)$ and $y^*=macron(y)$,
it will model measurements of the distance to the UFO without knowing its exact position,
only the average.

Our goal is to find and visualize the conditional pdf $f_(X,Y) (x, y | d^*)$ of the UFO
position given a measured distance $d^*$ and compare it with the unconditional pdf $f_(X,Y) (x, y)$.

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

Now the unconditional $f_(X,Y) (x, y)$ and conditional $f_(X,Y) (x, y | d^*)$
densities can be visualized and compared.

The non-directional radar at position $(0, 0)$ measures distance $d$ (depicted as a
circle) to the UFO. The UFO is hovering around the mean position $(4, 4)$ with s.d. $sigma_X = sigma_Y = 3$.
The corresponding probability density is shown by the color gradient. The measured
distance $d$ constrains the probable UFO location to a circular arc in the conditional
density. The s.d. $sigma_D$ determines the width of the arc and was chosen to be $sigma_D = 0.5$ (relatively
small compared to fluctuations in the UFO position).

// embed code
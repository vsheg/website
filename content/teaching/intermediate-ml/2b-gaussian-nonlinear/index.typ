#import "@preview/physica:0.9.3": *
#import "../defs.typ": *
#import "../template.typ": *

#show: template

= Non-linear Optimization: Newton --- Gauss Method

The Newton–Gauss method is a second-order optimization technique for quadratic functions,
utilizing a linear approximation of the optimized function at each step. It is applied to
solve nonlinear least squares problems, effectively reducing them to a sequence of linear
least squares problems.

== Gradient and Hessian of the Loss Function

Given the quadratic loss function

$
  Q(xb) = sum_(xb in X^ell) (a(xb, tb) - y(xb))^2
$

we can express the gradient and Hessian of the function in terms of the model's
parameters:

#margin[
  gradient is the column vector:
  $ nabla f(bold(x)) := vec(pdv(f(xb), x_1), dots.v, pdv(f(xb), x_k)), $
  and $f'_j$ denotes $j$th component of the column
]

1. The gradient components are

  $
    Q'_j
      &= pdv(Q, theta_j) \
      &= 2 sum_(xb in X^ell) (a(xb, tb) - y(xb)) dot pdv(a(xb, tb), theta_j)
  $

2. The Hessian components are

  $
    Q''_(i,j) &= pdv(Q, theta_i, theta_j) \
              &= 2 sum_(xb in X^ell) pdv(a(xb, tb), theta_i) pdv(a(xb, tb), theta_j) - 2
    sum_(xb in X^ell) (a(xb, tb) - y(xb)) dot pdv(a(xb,
    tb), theta_i, theta_j).
  $

== Linear Approximation of the Algorithm

Apply a Taylor series expansion of the algorithm up to the linear term near the current
approximation of the parameter vector $hat(tb)$:

$
  a(xb, tb) = ub(a(xb, hat(tb)), const) + sum_j ub(pdv(a(xb, hat(tb)), theta_j), const_j)
  ub((theta_j - hat(theta)_j), delta theta_j) + O(norm(tb - hat(tb))^2),
$

$a(xb, hat(tb))$ is constant, and the linear term is the sum of the partial derivatives of
$a(xb, hat(tb))$ with respect to the parameters $theta_j$. The higher-order terms are
negligible and will be omitted below.

Differentiate the linear approximation of the algorithm:

$
  pdv(, theta_j) a(xb, tb)
    &approx 0 + ub(pdv(a(xb, hat(tb)), theta_j), const_k) dot 1 + cancel(O(norm(tb - hat(tb))^2)) \
    &= const_j
$

The components of the sum depending on $theta_(j!=k)$ was zeroed out in the
differentiation over $theta_k$.

Substitute the obtained derivative into the expression for the Hessian:

$
  Q''_(i,j)
    & approx 2 sum_(xb in X^ell) ub(pdv(a(xb, hat(tb)), theta_i), const_i)
  ub(pdv(a(xb, hat(tb)), theta_j), const_j) - cancel(2 sum_(xb in
  X^ell) (a(xb, tb) - y(xb)) dot 0)
$

The linear term will be zeroed out in the second differentiation and will not enter the
Hessian.

== Matrix Formulation of the Optimization Step

Introduce the matrix of first partial derivatives and the algorithm's response vector at
the current approximation of the parameters $hat(tb)$:

$
  D := {pdv(a(xb_i, hat(tb)), theta_j)}_(i,j), quad bold(a) :=
  vec(a(xb_1, hat(tb)), dots.v, a(xb_ell, hat(tb)))
$

matrix $D$ and vector $bold(a)$ depend on the point of expansion $hat(tb)$ and are
recalculated at each optimization step.

The gradient and Hessian (at each step) are calculated using the matrix $D$:

$
  Q' = D^Tr (bold(a) - bold(y)), quad Q'' = D^Tr D (bold(a) - bold(y))
$

#margin[Newton --- Rafson method is a second-order optimization technique that provides fast
  convergence. Newton–Gauss method is an approximate second-order method that uses a linear
  approximation of the optimized function at each step.]

The optimization step of the Newton --- Rafson method is also expressed in terms of the
matrix $D$:

$
  tb<- tb - gamma dot overbrace(ub((D^Tr D)^(-1) D^Tr, D^+)
  ub((bold(a) - bold(y)), bold(epsilon)), delta tb)
$

#margin[The nonlinear optimization problem is reduced to a sequence of linear problems: at each
  iteration, a linear expansion of the function is made, matrices are calculated, and a
  (new) system of linear equations is solved.]

The optimization step vector at each iteration can be determined from the linear system in
any of these formulations:

$
  ub(bold(epsilon), bold(y)) = D dot ub(delta tb, bold(beta))
  quad <=> quad delta tb = D^+ bold(epsilon) quad <=> quad norm(D dot delta
  tb - bold(epsilon))^2 -> min_bold(beta)
$

#margin[The method is a second-order approximation method, providing fast convergence and slightly
  inferior accuracy compared to the Newton–Raphson method.]

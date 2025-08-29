#import "@preview/physica:0.9.3": *
#import "../defs.typ": *
#import "../template.typ": *

#show: template

= Principal Component Analysis (PCA)

== Problem
#margin(
  title: [Сrumbs on the floor],
)[
  Each data point is represented by three coordinates $x, y, z$, but $z$ is always $0$.
  Therefore, the data can be represented by just two coordinates:

  $ bold(f) = vec(x, y, 0) quad ->^A quad bold(p) = vec(x, y) quad ->^B quad hat(bold(f)) = vec(x, y, 0). $

  It is straightforward to find the linear transformations $A$ and $B$:

  $ ub(mat(1, 0, ?;0, 1, ?), A) vec(x, y, 0) = vec(x, y), quad ub(mat(1, 0;0, 1;0, 0), B) vec(x, y) = vec(x, y, 0) $
]
#margin(
  title: [NB],
)[
  In the example above:

  - The last column of $A$ is arbitrary, so the choice of transformations is not unique.
  - $A$ and $B$ are related: $hat(bold(f)) = B A bold(f)$, thus $B A = I$.
  - Since $A$ and $B$ are non-square, they are non-invertible, so $A = B^(-1)$ does *not*
    hold.
]
- Principal Component Analysis (PCA) is a feature transformation method that converts the
  original features $bold(f)$ into a new set of transformed features $bold(p)$, ensuring
  their linear independence:
  $
    bold(f) = vec(f_1, dots.v, f_k) quad &-> quad bold(p) = vec(p_1, dots.v, p_m),
  $

  If the original features are linearly dependent, the data resides in a lower-dimensional
  space, meaning $m < k$. For clarity, we will assume $m < k$ explicitly.

- The new representation $p_1, ..., p_m$ is constructed as a linear combination of the
  original features $f_1, ..., f_k$:
  $
    p_s = sum_(j=1)^k alpha_(s,j) dot f_j,
  $

  the coefficients $alpha_(s,j)$ form the matrix $A$, which defines the linear
  transformation from $bold(f)$ to $bold(p)$.

- The new, usually lower-dimensional, representation $bold(p)$ must still be informative.
  This is achieved by ensuring that $bold(p)$ can approximately restore the original
  features $bold(f)$ linearly and with minimal error:
  $
    hat(f)_j = sum_(s=1)^m beta_(j,s) dot p_s approx f_j,
  $

  the coefficients $beta_(j,s)$ form the matrix $B$, which defines the linear transformation
  from $bold(p)$ back to $bold(f)$.

- The objective of PCA is to minimize the reconstruction error $bold(hat(f)) - bold(f)$ by
  finding the optimal linear transformations $A: bold(f) -> bold(p)$ and $B: bold(p) -> bold(f)$:

  $
    R = sum_(bold(x) in X^ell) norm(hat(bold(f)) - bold(f))^2 = sum_(bold(x) in X^ell) norm(B A bold(f) - bold(f))^2 -> min_(A, B).
  $

#margin(
  title: [Сrumbs on the table.],
)[
  Now, the third coordinate equals the table height $h = 1$:

  $ vec(x, y, 1) ->^A vec(x, y), quad vec(x, y) ->^B vec(x, y, 1) $

  Here, $A$ is the same as before, but no $B$ can restore the original vector exactly.

  Formally, if $B$ exists, we could write the system of equations:

  $ mat(beta_(1,1), beta_(1,2);beta_(2,1), beta_(2,2);beta_(3,1), beta_(3,2)) vec(x, y) = vec(x, y, 1) => cases(1 x + 0 y = x, 0 x + 1 y = y, beta_(3,1) x + beta_(3,2) y = 1). $

  - The coefficients in the first two equations are determined by the identities $x = x$ and $y = y$.
  - The third equation cannot yield $1$ for all $x, y$ since it lacks a bias term.
]

#margin(
  title: [Approximate solution.],
)[
  In the example above, we could find $B$ as the pseudoinverse $B = A^+ = (A^Tr A)^(-1) A^Tr$,
  but:

  - The original vector will only be restored approximately, so $A B approx I$.
  - Since the choice of $A$ is arbitrary, the choice of $B$ is also arbitrary. This freedom
    allows us to impose additional constraints on the transformations.
]

== Linear Maps
Matrices $A$ (dimension reducer) and $B$ (dimension adder) are linear maps that work
oppositely: $A$ reduces the dimension of the original features $bold(f)$ to the dimension
of the principal components $bold(p)$, and $B$ restores, as closely as possible, the
original features from the principal components.

$
  bold(f) &= vec(f_1, dots.v, f_k) quad &->^A quad bold(p) &= vec(p_1, dots.v, p_m) quad ->^B quad bold(hat(f)) &= vec(hat(f)_1, dots.v, hat(f)_k)
$

This can be written as:

$
  bold(p) = A bold(f), wide
  hat(bold(f)) = B bold(p).
$

== Matrix Formulation
The feature matrix $F$ and the principal component matrix $P$ are formed by stacking the
row vectors $bold(f)^Tr = (f_1, ..., f_k)$ and $bold(p)^Tr = (p_1, ..., p_m)$:

$
  F := vec(bold(f)_1^Tr, dots.v, bold(f)_ell^Tr), quad P :=
  vec(bold(p)_1^Tr, dots.v, bold(p)_ell^Tr)
$

In matrix form, the linear maps $A$ and $B$ are applied as follows:

$
  P^Tr = A F^Tr, wide hat(F)^Tr = B P^Tr,
$

or equivalently, by transposing:
$ P = F A^Tr, quad hat(F) = P B^Tr. $

Substituting $P$ into $hat(F)$ yields the following equation:

$
  hat(F) = F A^Tr B^Tr = F (A B)^Tr,
$

The approximation $hat(F)$ equals $F$ exactly if $A B = I$. Ideally, $A$ would equal $B^(-1)$,
but in general, $A$ and $B$ are non-square and therefore non-invertible.

== Pseudoinverse matrix
$A B = I$ holds if $B$ is the pseudoinverse of $A$:
#margin[
  $B A = A^+ A = (A^Tr A)^(-1) (A^Tr A) = I$
]
$
  B = A^+ = (A^Tr A)^(-1) A^Tr.
$

$A^+$ is exact if $A$ has full rank, but in general, it does not, so the solution is only
approximate:

$ A B approx I. $

== Geometric Interpretation
#margin(
  title: [Basis Transition Matrix.],
)[
  If in vector space $V$, there are two bases: the old one $cal(O): bold(omega)_1, ..., bold(omega)_n$ and
  the new one $cal(N): bold(nu)_1, ..., bold(nu)_n$, the vectors of the new basis can be
  represented as linear combinations of the old basis vectors:

  $ cases(
    bold(nu)_1 = focus(alpha_(1,1)) bold(omega)_1 + ... + focus(alpha_(1,n)) bold(omega)_n,
    dots.v,
    bold(nu)_n = alpha_(n,1) bold(omega)_1 + ... + alpha_(n,n) bold(omega)_n,

  ) $

  The coefficients $alpha_(s,j)$ are the coordinates of the new basis vectors in the
  coordinate system of the old basis. These coefficients form the basis transition matrix
  (by columns!):

  $ A = mat(
    focus(alpha_(1,1)), ..., alpha_(n,1);dots.v, dots.down, dots.v;focus(alpha_(1,n)), ..., alpha_(n,n)
  ) $

  This matrix transforms coordinates between bases:

  $ {bold(nu)_1}_cal(O) = vec(focus(alpha_(1,1)), dots.v, focus(alpha_(1,n)))_cal(O) = A vec(1, 0, dots.v)_cal(N) = A{bold(nu)_1}_cal(N) $

  $ {bold(v)}_cal(O) = A {bold(v)}_cal(N), quad {bold(v)}_cal(N) = A^(-1) {bold(v)}_cal(O) $
]
Matrices $A$ and $B$ resemble transition matrices between bases:

- $A$ transforms vectors from the original basis of features $f_1, ..., f_k$ into a new
  space with the basis of principal components $p_1, ..., p_m$. However, since these bases
  are in different dimensional spaces, this is only an analogy.

- $B$ performs the reverse transformation, converting from the principal component basis
  back to the original basis (approximately).

Since $A$ and $B$ are related by the pseudoinverse operation and perform inverse
transformations, we can focus on one of the matrices. Let it be $B$.

The basis transition matrix stores the vectors of the new basis in the coordinates of the
old basis. As the linear map $B$ transforms principal components into the original
features (approximately):

$ bold(f) approx B bold(p), $

it acts similarly to a basis transition matrix from $bold(f)$ to $bold(p)$, storing the
orthogonal basis of principal axes in the coordinates of the original space.

#margin[
  The choice of matrix $B$ is flexible, allowing us to impose additional constraints. For
  example, we can require that $B^T B$ be diagonal or even the identity matrix:

  $ B^T B = mat(1, 0, 0;0, 1, 0) mat(1, 0;0, 1;0, 0) = mat(1, 0;0, 1) $
]

Any basis consists of linearly independent, or orthogonal, vectors, meaning that $B$ stores
orthogonal vectors, and $B^Tr B = Lambda$ is diagonal.

Since the choice of $B$ is not unique, we can use this freedom to demand that $B^Tr B$ be
not just diagonal $Lambda$, but the identity matrix $I$:

$ exists B: B^Tr B = I, $

This implies that $B$ stores not just orthogonal vectors but an *orthonormal* basis of
principal components.

== Risk Minimization
The objective of PCA is to minimize the restoration error. In this notation, the empirical
risk depends on $A$ and $B$:

$
  R :&= norm(hat(F) - F)^2 \
     &= norm(F A^Tr B^Tr - F)^2 -> min_(A, B).
$

We can reformulate the objective in terms of the new coordinates $P$ and the transition
matrix $B$ by substituting $P = F A^Tr$, which at least reduces one matrix multiplication:

$
  R = norm(P B^Tr - F)^2 -> min_(P, B).
$

By differentiating $R$ with respect to $P$ and $B$, we can find the values of $P$ and $B$ at
the extremum:

#columns(2)[

  $ pdv(R, P) = 2 (P B^Tr - F) B = 0 \
  arrow.b.double \
  P = F B (B^Tr B)^(-1) $

  #colbreak()

  $ pdv(R, B) = 2 P^Tr (P B^Tr - F) = 0 \
  arrow.b.double \
  B^Tr = (P^Tr P)^(-1)P^Tr F $

  $
    B
      &= F^Tr P ((P^Tr P)^(-1))^Tr \
      &= F^Tr P ((P^Tr P)^Tr)^(-1) \
      &= F^Tr P (P^Tr P)^(-1)
  $

]

#margin[
  $S = P^Tr P$ is symmetric, i.e. $S^T = S$
]

The objective $R$ depends only on the product $P B^T$, which can result from multiplying
any number of different pairs of matrices:

$ P B^Tr = P I B^Tr = ub((P^* R), P) ub((R^(-1) B^*^Tr), B^Tr) $

#margin[
  Earlier, we showed that $B$ could be chosen to store an orthonormal basis, but this wasn't
  strictly necessary.

  It can be demonstrated analytically that it is sufficient to choose $R$ such that $B^T B$ is
  diagonal, which is enough to ensure $B^T B = I$. This will determine the form of $B$,
  which can then be interpreted as a matrix storing an orthonormal basis.

  As the proof involves boring linear algebra, we relied on geometric intuition instead
  (though formal proof is possible!). //TODO: add reference.
]

We will use the freedom in choosing $R$ and let $P^Tr P$ and $B^Tr B$ be diagonal:

- $P$ stores the principal components in their respective coordinates.
- $B$ stores the orthonormal "basis" of principal components in the coordinates of the
  original space, so $B^Tr B = I$.

$
  cases(P^Tr P = Lambda, B^Tr B = I)
$

Now, we can further simplify the expressions for $P$ and $B$:

$
  P = F B (B^Tr B)^(-1) = F B I, \
  B = F^Tr P (P^Tr P)^(-1) = F^Tr P Lambda^(-1).
$

#columns(2)[
  Eliminate $P$:
  $ B Lambda = F^Tr F B $

  This means that the columns of $B$ are eigenvectors of $F^Tr F$:
  $ bold(b)_j dot lambda_j = (F^Tr F) bold(b)_j. $

  #colbreak()

  Eliminate $B$:
  $ P Lambda = F F^Tr P $

  This means that the columns of $P$ are eigenvectors of $F F^Tr$:
  $ bold(p)_j dot lambda_j = (F F^Tr) bold(p)_j. $
]

#import "../template.typ": *
#show: template

= Diagonal matrices
Matrix $Lambda$ is diagonal if all its off-diagonal elements are zero:

#let Lambda-full = $mat(lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda_k)$
#let Lambda-short = $dmat(lambda_1, dots.down, lambda_k)$

$ Lambda = diag(lambda_1, ..., lambda_k) = #Lambda-full equiv #Lambda-short $

where $lambda_1, dots, lambda_k$ are the diagonal elements of $Lambda$. For simplicity,
off-diagonal elements are often omitted.

#margin[
  / Zero matrix $O$: A diagonal matrix where all diagonal elements are zero:
    $ O = diag(0, dots, 0) = mat(0, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, 0) $

  / Identity matrix $I$: A diagonal matrix where all diagonal elements are one:
    $ I = diag(1, dots, 1) = mat(1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, 1) $

  / Scalar matrix: A diagonal matrix with constant diagonal elements:
    $ lambda I = diag(lambda, dots, lambda) = mat(lambda, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda) $
]
Diagonal matrices have several important properties:

== Linear independence
The fundamental property of diagonal matrices is linear independence of their columns, which 
holds if and only if all diagonal elements are non-zero. This directly follows from the fact 
that any linear combination of columns equaling zero requires all coefficients to be zero.
#note[
  For any two distinct columns $i != j$ of a diagonal matrix $Lambda$, their dot product is
  zero since they have non-overlapping non-zero elements:

  $ vec(dots.v, focus(lambda_i), 0, dots.v)^Tr dot vec(dots.v, 0, focus(lambda_j), dots.v) = 0 $
]

== Basis set
For a diagonal matrix with non-zero diagonal elements, the columns form an orthogonal basis. 
Each column $bold(b)_i$ contains exactly one non-zero element $lambda_i$:

$ cal(B): wide bold(b)_1 = vec(lambda_1, 0, dots.v, 0), quad bold(b)_2 = vec(0, lambda_2, dots.v, 0), quad ... , quad bold(b)_k = vec(0, 0, dots.v, lambda_k) $

#margin[
  Diagonal matrices are well suited for storing bases. If all diagonal elements are equal to
  one, the matrix is an identity matrix $I$ and it stores orthonormal basis vectors of the
  standard basis.

  In 3D space, the identity matrix $I$ stores the standard basis vectors:
  $ bold(i) = vec(1, 0, 0), quad bold(j) = vec(0, 1, 0), quad bold(k) = vec(0, 0, 1) $

]

This orthogonal basis has a clear geometric interpretation --- each basis vector aligns with 
a coordinate axis and has magnitude $|lambda_i|$. When all $|lambda_i| = 1$, the basis 
becomes orthonormal.

== Scaling
A diagonal matrix $Lambda$ performs scaling transformations by independently scaling each
coordinate by its corresponding diagonal element:

$ Lambda bold(v) = vec(lambda_1 v_1, dots.v, lambda_k v_k) $

This represents stretching or compressing the space along each coordinate axis by factors $lambda_1,dots,lambda_k$.

#note[
  $
    Lambda bold(v) = #Lambda-full dot vec(v_1, dots.v, v_k) = vec(lambda_1 v_1 + 0, dots.v, 0 + lambda_k v_k)
  $
]

== Inverse matrix
For a diagonal matrix, its inverse is obtained by taking the reciprocal of each diagonal
element, provided all diagonal elements are non-zero.
$ diag(lambda_1, ..., lambda_k)^(-1) = diag(1/lambda_1, dots, 1/lambda_k) $

#note[
  Assuming, that the inverse of a diagonal matrix $Lambda = diag(lambda_1, ..., lambda_k)$ is $Lambda^(-1) = diag(1/lambda_1, ..., 1/lambda_k)$,
  then:
  $
    #Lambda-full mat(1/lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, 1/lambda_k) = mat(
      lambda_1 dot 1/lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda_k dot 1/lambda_k


    ) = I
  $
]

== Commutativity
#margin[
  Moreover, diagonal matrices form a commutative group under multiplication.
]
Diagonal matrices have the special property that they commute under multiplication:

$ Lambda_1 Lambda_2 = Lambda_2 Lambda_1 $
#note[
  For two diagonal matrices $Lambda$ and $Mu$:
  $
    Lambda Mu = #Lambda-full mat(mu_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, mu_k) &= mat(lambda_1 mu_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda_k mu_k) \
                                                                                            &= mat(mu_1 lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, mu_k lambda_k) = mat(mu_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, mu_k) mat(lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda_k) = Mu Lambda
  $
]

#margin[
  Any diagonal matrix can be decomposed into a product of diagonal matrices, at least:
  $ Lambda = Lambda I ... I $
]

Therefore, any order of multiplication of diagonal matrices results in the same diagonal
matrix:
$ Lambda equiv Lambda_1 Lambda_2 ... Lambda_k = Lambda_k ... Lambda_2 Lambda_1 $

== Eigenvalues and eigenvectors
For a diagonal matrix, the eigenvalues are precisely its diagonal elements, while the
eigenvectors are the standard basis vectors of the space.

#margin[
  Eigenvalues and eigenvectors arise from the matrix equation $A bold(v) = lambda dot bold(v)$,
  where $A$ is a square matrix, $lambda in RR$ is an eigenvalue, and $bold(v) in RR^k$ is an
  eigenvector.

  When viewing $A$ as an operator, eigenvectors represent directions that maintain their
  orientation under the operation, being only scaled by the factor $lambda$.
]

#note[
  For a standard basis vector $bold(e)_i$:

  $ Lambda bold(e)_i = #Lambda-full dot bold(e)_i = vec(0, dots.v, lambda_i, dots.v, 0) = lambda_i vec(0, dots.v, 1, dots.v, 0) = lambda_i bold(e)_i $
  so, $bold(e)_i$ is an eigenvector of $Lambda$ with eigenvalue $lambda_i$.
]
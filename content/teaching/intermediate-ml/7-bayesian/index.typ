#import "../template.typ": *
#show: template

= Foundations of Bayesian Statistics

== Bayes' Theorem for Probability Densities

The conditional probability mass function of a random variable $xi$ given another random variable $eta$ can be expressed as:

$
  f_(xi|eta)(x | y) := (f_(xi|eta)(x, y)) / (f_eta (y)) = (f_(xi, eta)(y | x) dot f_xi (x)) / (f_eta (y))
$

Where:
$f_(xi|eta)$ is the conditional pmf \
$f(x, y) = f(y|x) dot f(x)$ is the joint distribution \
$x in supp xi$ \
$y in supp eta$

== Likelihood

The joint probability of an object $bold(x)$ and its class $y$ can be decomposed in two equivalent ways:

$
  p(bold(x), y)
  &= p(bold(x)) dot p(y|bold(x)) \
  &= p(y) dot p(bold(x)|y)
$

Where:
$p(bold(x)) equiv Pr[X = bold(x)]$ is the distribution of $bold(x)$ \
$p(y) equiv Pr[Y = y]$ is the probability of a class $y$ \
$p(bold(x) | y) equiv Pr[X = bold(x) | y(bold(x)) = y]$ is the likelihood function of a class $y$ \
$p(y | bold(x)) equiv Pr[Y = y | X = bold(x)]$ is the a posterior probability of a class $y$ \
$p(x|y) := p(x, y) / p(y)$ is the Bayes' formula

Objects and class labels are produced by a joint distribution:
$ X^ell, Y = (bold(x)_i, y_i)_(i=1)^ell tilde p(bold(x), y) $

The term $p(x | y)$ is called the *likelihood* as it assesses how likely the data (objects $x$) comes from a particular class $y$. For example, if $x$ is weight and $y$ is one of two classes, 'cat' or 'dog', $p(x | y = "cat")$ is the probability for a particular animal to have weight $x$ assuming it is a cat. If this probability is low, the animal is unlikely to be a cat.

== Maximum Likelihood Classifier (Bayes Optimal Classifier)

$
  a^*(bold(x))
  &= arg max_(y in Y) lambda_y dot p(y|bold(x)) \
  &= arg max_(y in Y) lambda_y dot p(y) dot p(bold(x)|y)
$

Where:
$lambda_y$ is the error cost for class $y$ \
$p(bold(x))$ is the distribution of $bold(x)$ \
$p(y)$ is the probability of class $y$ \
$p(bold(x) | y)$ is the likelihood function of class $y$ \
$p(y | bold(x))$ is the posterior probability of class $y$ \

This classifier is considered "optimal" because it minimizes the probability of classification error:

$
  a = a^* colon quad R(a) = sum_(y in Y) lambda_y integral Ind(a(bold(x)) != y) dot f(bold(x), y) dd(bold(x)) -> min_a
$

by selecting the hypothesis that maximizes the posterior probability $p(y | x)$ given the data.

In practice, empirical probability densities $hat(p)$ are used, so the classifier is not truly optimal. The $p(x)$ term from the Bayes formula can be eliminated from argmax over $y in Y$ as it depends only on $x$:

$
  a(bold(x))
  &= arg max_(y in Y) p(y|bold(x)) \
  &= arg max_(y in Y) (p(y) dot p(bold(x) | y)) / cancel(p(bold(x)))
$

If the probabilities of all classes are equal $p(y_i) = p(y_j), forall i,j$, then $p(y)$ can also be eliminated from argmax:

$
  a(bold(x)) = arg max_(y in Y) p(bold(x) | y)
$

The main idea of this method is to assign the object to the class whose probability density is maximized in a given region.

However, it's optimal only if we know the true distributions $p(y | x)$ or $p(y), p(x|y)$, which are unknown in practice. Overfitting occurs when the empirical estimations $hat(p)(y | x)$ or $hat(p)(y), hat(p)(x|y)$ are used.

= Discriminative vs Generative Models

== Key Characteristics

#margin[
  - A **discriminative classification model** focuses on learning the **decision boundary between classes** by modeling the conditional probability $P(y|x)$, where $y$ is the class label and $x$ is the input data. These models do not attempt to model the distribution of the data itself, but rather directly predict the label given the features.

  - A **generative classification model** aims to model the **joint probability distribution** $P(x, y)$, meaning they model **how the data is generated for each class**. From this, they can derive the conditional probability $P(y|x)$.
]

#grid(
  columns: (auto, 1fr, 1fr),
  [**Aspect**], [**Discriminative Models**], [**Generative Models**],
  [**Goal**],
  [Learn the decision boundary between classes],
  [Model the joint probability $P(x, y)$],

  [**Probability Modeled**],
  [$P(y | x)$ (conditional probability)],
  [$P(x, y)$ (joint probability)],

  [**Example Algorithms**],
  [Logistic regression, GLM, SVMs, neural networks],
  [Naive Bayes, GMMs, Fisher's Linear Discriminant, HMMs],

  [**Accuracy**],
  [Often more accurate for classification tasks],
  [Can be less accurate for classification],

  [**Data Generation**],
  [Cannot generate new data],
  [Can generate new data (e.g., by sampling from $P(x, y)$)],

  [**Insight into Data**],
  [Focuses on boundaries between classes],
  [Models the distribution of the data],

  [**Complexity**],
  [Simpler, as only the boundary is learned],
  [More complex, as the full data distribution is modeled. Require more data.],

  [**Use Case**],
  [When classification accuracy is the priority],
  [When understanding data distribution or data synthesis is needed],
)

In a discriminative approach, it's important to accurately describe only the points near the decision boundary, while other points can be ignored. Methods like SVM are sensitive to outliers since they build their decision boundary based on these points.

In a generative approach, the probability distribution that generated the observed data is recovered. These methods are more robust to outliers that are far from the decision boundaries.

Generative approaches should be used when density distribution recovery is required, such as for data interpretation. For simple classification tasks, discriminative approaches will work better, though they are more complex and require more data.

== Error and Empirical Risk

The error (empirical risk) of the optimal Bayesian classifier is:

$
  R(a) = sum_(y in Y) lambda_y integral Ind(a(bold(x)) != y) dot f(bold(x), y) dd(bold(x)) -> min_a
$

Where:
$lambda_y$ is the error cost for class $y$ \
$p(y) equiv Pr[y] = integral f(bold(x), y) dd(bold(x))$ is the probability of class $y$ (marginal dist.) \
$Ind(a(bold(x)) != y)$ defines the region of space where the algorithm made an error

The integral exactly equals the total probability of error: the probability $f(x, y)$ is integrated over the region of space where the algorithm makes an error.

The form of the Bayesian classification algorithm (maximum likelihood classifier) follows from minimizing this empirical risk:

$
  a^*(bold(x))
  &= arg max_(y in Y) lambda_y dot p(y|bold(x)) \
  &= arg max_(y in Y) lambda_y dot p(y) dot p(bold(x)|y)
$

The Bayesian classifier $a(x)$ is optimal in the sense that it minimizes the error $R(a) -> min$. This optimality is proven when the density distributions $p(y|x)$ or $p(y), p(x|y)$ are exactly known, but in reality, they are unknown. Overfitting occurs when empirical estimates of distributions $hat(p)(y|x)$ or $hat(p)(y), hat(p)(x|y)$ are used to build the classifier.

If the error costs $lambda_y$ and/or class probabilities $p(y) equiv Pr[y]$ are equal for all classes, they can be removed from the $arg max$ operator.

A generalization of the optimal Bayesian classifier formula for when the error cost $lambda = lambda_(y|y')$ depends not only on the true class $y$ but also on the erroneous class $y'$ predicted by the algorithm:

$
  R(a) = sum_(y in Y) sum_(y' in Y \ y) lambda_(y|y') integral_(supp bold(x)) Ind(a(bold(x)) = y') dot f(bold(x), y) dd(bold(x)) -> min_a
$

$
  a^*(bold(x)) = arg min_(y' in Y) sum_(y in Y) lambda_(y|y') dot p(y) dot p(bold(x)|y)
$

This records the total probability that the algorithm will confuse the true class with another class, taking penalties into account. The original formulation treats the error cost as a class weight; the more dangerous it is to make a mistake on a class, the greater its weight.

The minimal error value $R(a)$ is achieved if the class $y'$ is chosen for which the probability of error is **minimal**:

$
  a^*(bold(x)) = arg min_(y' in Y) sum_(y in Y) lambda_(y|y') dot p(y) dot p(bold(x)|y)
$

In the original formulation, the posterior probability of class $y$ is **maximized**, and $arg max$ is used over the true classes. In this generalization, the class $y'$ is chosen for which the prediction of error probability is lowest.

= Naïve Bayes Classifier

== General Naïve Bayes

The Naïve Bayes classifier is expressed as:

$ a(bold(x)) = arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k ln hat(p)_j (bold(x)^j|y) } $

Where:
$p(bold(x)|y) := p_1 (f_1 (bold(x)) | y) ... p_k (f_k (bold(x)) | y)$, meaning features are independent \
$hat(p)(y)$ is the empirical probability density \
$hat(p)_j (bold(x)^j)$ is the 1D empirical distribution for the $j$th feature

From the optimal Bayes classifier:
$
  a^*(bold(x)) = arg max_(y in Y) lambda_y dot p(y) dot p(bold(x)|y)
$

Restoring $n$ one-dimensional densities is a much simpler task than restoring an $n$-dimensional one. The method is called "naive" because features are assumed to be independent. Density estimation can be modeled in any framework:
- parametric density estimation
- non-parametric density estimation
- mixture of distributions

== Training Process

The Naïve Bayes model can be expressed as:

$
  a(bold(x))
  &= arg max_(y in Y) lambda_y dot hat(p)(y) dot product_(j=1)^k hat(p)_j (bold(x)|y) \
  &= arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k ln hat(p)_j (f_j (bold(x))|y) }
$

Naïve Bayes assumes independence of all features. We need to train a 1D empirical probability distribution per feature. Using the exponential family, which encompasses many distributions:

$
  hat(p)(bold(x)^j | theta_(j|y)) = h(bold(x)^j) / Z(theta_(j|y)) dot e^(theta_(j|y) dot T(bold(x)^j)), quad T(x) equiv x
$

The empirical risk for all objects, features, and classes can be split per class and per feature, resulting in $|Y| dot k$ separate 1D optimization problems:

$
  Q(bold(theta))
  &= sum_(bold(x) in X^ell) sum_(j=1)^k sum_(y in Y) ln hat(p)(bold(x)^j | theta_(j|y)) \
  &= sum_(y in Y) sum_(j=1)^k { sum_(bold(x) in X_y) ln p(bold(x)^j | theta_(j|y)) } -> max_bold(theta)
$

The empirical risk per separate problem:

$
  Q_(j|y) = sum_(bold(x) in X_y) { theta_(j|y) dot bold(x)^j - ln Z(theta_(j|y)) + ln h(bold(x)^j) } -> max_(theta_(j|y))
$

Finding the extremum of the empirical risk:

$
  pdv(Q_(j|y), theta_(j|y)) = sum_(bold(x) in X_y) { bold(x)^j - pdv(,theta_(j|y)) A(theta_(j|y)) } = 0
$

Thus, the mean parameter equals the average feature value per class:

$hat(mu)_(j|y) = 1 / (|X_y|) sum_(bold(x) in X_y) bold(x)^j = macron(bold(x))^j_y$

And the canonical parameters can be found with the canonical link function depending on the specific distribution:

$hat(theta)_(j|y) = psi(hat(mu)_(j|y)) = psi(macron(bold(x))^j_y)$

It's quite easy to train Naïve Bayes, as you only need to compute the average feature value per class and then use it to compute the canonical parameters. After that, the classifier can be used immediately.

== Linear vs. Non-linear Naïve Bayes

When using some distributions (like Bernoulli), Naïve Bayes becomes an ensemble of per-class linear models:

$
  a(bold(x))
  &= arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k ln hat(p)_j (bold(x)^j|y) } \
  &= arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k [ theta_(j|y) dot bold(x)^j - ln Z(theta_(j|y)) + ln h(bold(x)^j|y)] } \
  &= arg max_(y in Y) { hat(bold(theta))_(|y) dot bold(x) + ln lambda_y hat(p)(y) - sum_(j=1)^k ln Z(theta_(j|y))}
$

For other distributions, the algorithm is non-linear. For example, with the commonly used Gaussian distribution, the non-linear term is quadratic. Also, a dispersion parameter $phi$ can occur.

The Naïve Bayes classifier becomes a linear model when:

$
  a(bold(x))
  &= arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k ln hat(p)_j (bold(x)^j|y) } \
  &= arg max_(y in Y) { ln lambda_y hat(p)(y) + sum_(j=1)^k [ (theta_(j|y) dot bold(x)^j - A(theta_(j|y))) / phi_(j|y) + ln h(bold(x)^j, phi_(j|y))] }
$

When the dispersion parameter $phi$ does not depend on the class, we can eliminate $h(x)$ under $arg max_(y in Y)$ as it does not depend on $y$:

$a(bold(x)) = arg max_(y in Y) { hat(bold(theta))_(|y) dot bold(x) + ln lambda_y hat(p)(y) - sum_(j=1)^k ln Z(theta_(j|y)) + 0}$

- For most distributions from the exponential family, $phi equiv 1$, so the classifier is linear.
- For non-standard Gaussian distributions described by a generalized exponential family, if the data is normalized, again $phi equiv 1$ and $h(x)$ can be canceled. If the data is not normalized, but $sigma$ does not depend on class $y$, $h(x)$ can still be eliminated and the classifier remains linear.
- **For Gaussian distributions with heteroscedasticity** (variance depends on class), the Naïve Bayes classifier is **non-linear**. It will have a quadratic term $x^2 / sigma_y$.

Different features may be described by different distributions (e.g., feature #1 by Poisson, feature #2 by Gaussian, feature #3 by Bernoulli), and the classifier can still be linear.

== Text Classification with Naïve Bayes

In text classification, Poisson distribution models word $w_j$ occurrence in a document:

$
  Pr(K_j = k|nu)
  &= e^(-nu) dot nu^k / k! \
  &= 1 / k! dot exp { k dot ln nu - nu }
$

The Naïve Bayes classifier depends on the average document length:

$
  a(bold(x))
  &= arg max_(y in Y) { hat(bold(theta))_(|y) dot bold(x) + ln lambda_y Pr[y] - sum_(j=1)^k A(theta_(j|y))} \
  &= arg max_(y in Y) { hat(bold(theta))_(|y) dot bold(x) + ln lambda_y Pr[y] - macron(N)_y}
$

Poisson distribution models the probability of an event occurring exactly $k$ times with a given parameter $nu = e^theta$ (the average/expected number of events in the modeled time interval).

Poisson distribution is used in text classification models. Any document is represented as an object with occurrences of specific words from an ordered dictionary of words $w_1, ..., w_W$:

$ bold(x) = vec(k_1, dots.v, k_W) = vec(\# "fat", dots.v, \# "cat") = vec(10, dots.v, 15) $

The parameter $nu_(j|y) = e^(theta_(j|y))$ represents the average occurrence of the word $w_j$ in class $y$.

The expected occurrence of word $w_j$ in documents of class $y$:

$
  nu_(j|y) equiv bra K_(j) ket_y equiv Ex [K_j (bold(x)) | bold(x) in X_y] = pdv(,theta_(j|y)) A(theta_(j|y)) equiv e^(theta_(j|y)) = A(theta_(j|y))
$

The average length of documents in class $y$ is:

$
  macron(N)_y = bra N ket_y
  &= sum_(w_j in W) nu_(j|y) = sum_(j=1)^W A(theta_(j|y))
$

Finally:

$
  a(bold(x)) = arg max_(y in Y) { hat(bold(theta))_(|y) dot bold(x) + ln lambda_y Pr[y] - sum_(j=1)^k A(theta_(j|y))}
$

The more words in a document, the more non-zero elements in $bold(x)$ and the larger the linear part $theta dot bold(x)$. This is compensated by subtracting the average document length $N_y$. Long documents with many words don't have priority over short ones.

If the average occurrence of a word $f_(j|y)$ doesn't depend on class $y$, it's part of the common vocabulary and doesn't affect classification. This means Naïve Bayes has a built-in feature selection mechanism.

For text representation in Naïve Bayes classifiers:
- Text is a sequence of random events, where each word is a random event
- Each word in a document occurs independently of other words
- Each word is generated by a Poisson process (distribution)
- To obtain an object representation of text $bold(x)$, it's treated as a bag of words with integer frequencies $f_j$, where $j=1..W$ is the word's position in the dictionary, and these frequencies are recorded in vector $bold(x) = (f_1, ..., f_W)$

== Multinomial Bayes Classifier

From the optimal Bayes classifier, we can derive the multinomial Bayes classifier:

$
  a(bold(x)) = arg max_(y in Y) { ln hat(p)(bold(x)|y) + ln lambda_y hat(p)(y) }
$

A word $w$ is represented by its index in a dictionary of size $W$. Each word in a document is represented as number $w$, and the document is represented by a sequence (order not important) of all its words. Objects may have different dimensions and are not represented as a matrix $X$.

$
  w = 1..W, quad bold(x) = vec(w_1, dots.v, w_N_(bold(x))), quad |bold(x)| = N_bold(x)
$

Let's define counters for a word $w$ in document $bold(x)$ and in corpus of documents $X = { bold(x)_1, bold(x)_2, ... }$:

$nu_w (bold(x)) := sum_(w' in bold(x)) Ind(w' = w)$

$nu_w (X) := sum_(bold(x) in X) nu_(w) (bold(x))$

Also define empirical probability for each word $w$ for a given class $y$, i.e., the frequency of $w$ in documents of class $y$:

$
  pi_(w|y) :&= Pr["a random word from a document of class" y "is" w] \
  &= (nu_w (X_y)) / (sum_(w' in W) nu_(w') (X_y)) \
  &= (bra nu_w ket_(y)) / (bra N_bold(x) ket_(y)) = ("average occurance of" w "in docs of class" y) / ("average document size in" y)
$

Write the logarithm of empirical probability $ln hat p (x|y)$ using multinomial distribution:

$
  ln hat(p)(bold(x)|y)
  &= ln product_(w in bold(x)) (pi_(w|y))^(nu_w (bold(x))) \
  &= sum_(w in bold(x)) nu_w (bold(x)) dot ln pi_(w|y) = *
$

$
  * &= sum_(w in bold(x)) { nu_w (bold(x)) dot ln bra nu_w ket_y - nu_w (bold(x)) dot ln bra N_bold(x) ket_y } \
  &= sum_(w in bold(x)) nu_w (bold(x)) dot ln bra nu_w ket_y - sum_(w in bold(x)) nu_w (bold(x)) dot ln bra N_bold(x) ket_y
$

Finally:

$
  a(bold(x)) = arg max_(y in Y) { sum_(w in bold(x)) nu_w (bold(x)) dot ln bra nu_w ket_y - N_bold(x) dot ln bra N_bold(x) ket_y + ln lambda_y hat(p)(y) }
$

The linear part is a weighted logarithm of each word's (non-unique) occurrences. The classifier depends on the document's size (in contrast to Poisson Naïve Bayes).

== Advantages of Naïve Bayes

Naïve Bayes has several advantages:

- It trains in linear time, as it only requires computing average feature estimates by class and the dispersion/spread parameter (for some distributions)
- It rarely overfits, as the solution depends on average feature values, and these estimates are reliable, stable, and controlled by the law of large numbers
- It can uniformly handle features of different types, as different distributions can be used to describe them
- It has built-in feature importance through maximizing posterior probabilities $max_j p(y|f_j(x))$
- It has built-in feature selection through equal posterior probabilities $p(y|f_j(x)) approx p(y'|f_j(x))$ — if a feature equally affects the probability of all classes, it's uninformative, allowing for filtering of stop words and common vocabulary in text classification
- It can be used as a strong baseline model, but the independence assumption of features is too strong for more complex applications

= Discriminant Analysis

== Connection Between Metric and Bayesian Classifiers

The optimal Bayesian classifier (without logarithm of density):

$
  a(bold(x)) = arg max_(y in Y) lambda_y dot p(y) dot p(bold(x)|y)
$

Non-parametric Parzen-Rosenblatt density estimation:

$
  hat(p)(bold(x)|y) = 1 / (\# X_y dot V_h) sum_(bold(x)' in X_y) K( rho(bold(x), bold(x)') / h )
$

Substituting the density estimate into the Bayesian classifier, assuming **the same window width is used for all classes**:

$
  a(bold(x)) = arg max_(y in Y) { (lambda_y dot p_y) / (\# X_y) dot sum_(bold(x)' in X_y) K( rho(bold(x), bold(x)') / h ) }
$

If the window width doesn't depend on the class, we can exclude normalization by window width $V_h$ — it ensures that the empirical estimate is actually a density (normalization by the number of objects is introduced separately).

However, if different window widths are required for different classes, we need to calculate:
$ V(h_y) = integral_X K(rho(bold(x), bold(x)') / h_y) dd(bold(x)) $

Thus, the metric classifier (generalization of kNN) can be derived from more general principles — Bayes' theorem and kernel density estimation.

== Quadratic Discriminant

$
  a(bold(x)) = arg max_(y in Y) {ln lambda_y p(y) - 1 / 2 (bold(x) - hat(bold(mu))_y)^Tr hat(Sigma)_y^(-1) (bold(x) - bold(hat(mu))_y) -1 / 2 ln det hat(Sigma)_y }
$

Where:
$mu_y, Sigma_y$ are distribution parameters for class $y$ \
$Q(x^1, ..., x^k) = sum_i sum_j a_(i,j) x^i x^j = bold(x)^Tr A bold(x)$ is a quadratic form of features

Substituting parametric density estimates for each class into the optimal Bayesian classifier:

$
  a^*(bold(x))
  &= arg max_(y in Y) ln {lambda_y dot p(y|bold(x))} \
  &= arg max_(y in Y) ln {lambda_y dot p(y) dot p(bold(x)|y)}
$

$
  p(bold(x)|y)
  = exp(-1/2 (bold(x) - bold(mu)_y)^Tr Sigma_y^(-1) (bold(x) - bold(mu)_y)) / sqrt((2pi)^k det Sigma_y)
  tilde cal(N)(bold(mu)_y, Sigma_y)
$

The expression under $arg max$ is a quadratic form, creating a quadratic separating surface between classes. If covariance matrices are identical, the quadratic discriminant degenerates into a linear one.

If classes have equal prior probabilities and equal error costs, the first term disappears and the separating surface is a linear hyperplane passing exactly between the densities. If the error cost for a class is higher, it draws the separating surface closer.

In some cases, the density of one class can outweigh that of another at a distance, then the separating surface will have two regions. In the example, the left region would belong to the gray class, although it would be more logical to assign it to the red; this is a disadvantage of the generative approach.

If two Gaussians overlap (classes are inseparable), the separating surface will be a doughnut, and the density separation approach won't work.

When the covariance matrices for different classes are equal, the separating surface between them is linear. This can be shown by starting with the equation for the surface separating classes $y$ and $y'$:

$ lambda_y p(y) p(bold(x)|y) = lambda_y' p(y') p(bold(x)|y') $

After logarithmic transformation and substituting density estimates with equal covariance matrices, the quadratic terms cancel out, resulting in a linear form in terms of $x$:

$
  2 bold(x)^Tr Sigma^(-1) delta bold(mu) + Q(bold(mu)_y') - Q(bold(mu)_y) = 2 ln lambda_y' p(y') - 2 ln lambda_y p(y)
$

== Linear Discriminant Analysis (LDA)

LDA is a supervised learning method for classification and dimensionality reduction. It works by finding a linear combination of features that best separates two or more classes.

LDA assumes different classes generate data based on Gaussian distributions with the same covariance matrix but different means. The method maximizes the ratio of between-class variance to within-class variance to ensure maximum class separability. The resulting decision boundary is linear, and LDA can also reduce feature space dimension by projecting data onto a lower-dimensional subspace that preserves class separability.

== Fisher's Linear Discriminant

Fisher's linear discriminant (FLD) is a supervised linear classification method used to separate two or more classes of data. The technique aims to find a linear combination of features that maximizes the separation between classes while minimizing the within-class scatter. It is often applied in dimensionality reduction and classification problems.

The method assumes all classes have the same covariance matrix (i.e., all class distributions have the same shape but different bias), so it can be used to estimate the covariance matrix for small datasets. It seeks to project the data onto a line such that the separation between different class means is maximized while the variance within each class is minimized.

1. The whole dataset scatter matrix can be divided into two separate terms:

$
  hat(Sigma)
  &= 1 / ell X^Tr H X \
  &=
  1 / ell sum_(y in Y) sum_(bold(x) in X_y) (bold(x) - macron(bold(x))_y)(bold(x) - macron(bold(x))_y)^Tr
  + 1 / ell sum_(y in Y) ell_y (macron(bold(x))_y - macron(bold(x)))(macron(bold(x))_y - macron(bold(x)))^Tr
$

- **Within-class scatter matrix** is estimated as the mean covariance matrix of all classes.
- **Between-class scatter matrix** is estimated as the covariance matrix of the classes themselves.

2. The idea is to find a projection of a data vector onto a 1D line:
$z = P(bold(x)) = bold(x)^Tr bold(beta)$

3. The variance of this projection is decomposed into 2 terms:
$
  var z
  &= bold(beta)^Tr Sigma bold(beta) \
  &= bold(beta)^Tr Sigma_w bold(beta) + bold(beta)^Tr Sigma_b bold(beta)
$

4. That projection must maximize the ratio of the between-class variance to within-class variance:

$ (bold(beta)^Tr Sigma_b bold(beta)) / (bold(beta)^Tr Sigma_w bold(beta)) -> max_bold(beta) $

5. Linear discriminant function represents the distance from $x$ to a class center in the new projected space:

$
  delta_y (bold(x))
  &= bold(x)^Tr bold(beta) - bold(beta)^Tr macron(bold(x))_y \
  &= bold(x)^Tr bold(beta) - const_y
$

6. Finally, we use this distance (discriminant function) in the discriminant rule to choose the closest class:
$
  a(bold(x)) = arg min_(y in Y) abs(delta_y (bold(x)))
$

== Linear Discriminant Function

A linear discriminant function is defined as:

$ delta_y (bold(x)) = bold(beta)_y^Tr bold(x) + const_y $

This is a mapping from the feature space $RR^k$ to a set of categories $Y = {1, dots, N}$, defined by partitioning $RR^p$ into disjoint regions $cal(R)_1, dots, cal(R)_N$, where each region corresponds to a different category in $Y$.

This is achieved by maximizing **discriminant functions** $delta_y(bold(x))$ **for each category**, so $bold(x)$ is classified into $y$ when $delta_y(bold(x))$ exceeds the discriminant values for all other categories:
$hat y(bold(x)) = arg max_(y in Y) delta_y(bold(x))$,
where $y$ maps from $RR^k$ to $Y = {1, dots, N}$.

== Multiclass LDA

For multiclass problems, LDA can be applied in two different settings:

1. **One-vs-one**: Train $C_K^2$ pairwise models
$delta_(y,y') (bold(x)) = bold(beta)_(y,y')^Tr bold(x) + const_(y,y')$
And aggregate them using some kind of voting, e.g.,
$ delta_y (bold(x)) = sum_(y') delta_(y,y') $

The result is a decision boundary combined of multiple one-vs-one lines, which may be smooth depending on the voting approach used.

2. **One-vs-all**: Train exactly $K$ models and determine which class maximizes:
$hat(y) (bold(x)) = arg max_(y in Y) delta_y (bold(x))$

This effectively selects the hyperplane with the maximum distance to $x$. The final decision boundary is defined by distance; blue lines lie between hyperplanes at equal distances, and the decision is based on which side of the blue line the object $x$ lies.

= Gaussian Mixture Models

== Model Structure

In a Gaussian Mixture Model (GMM), the density of each class $y$ is described by a weighted sum of $N_y$ multivariate Gaussian densities:

$
  f(bold(x)|y, Theta) := sum_(n=1)^N_y w_(n|y) dot f_n (bold(x) | bold(mu)_(n|y), Sigma_(n|y))
$

All features are assumed independent, so all covariance matrices are diagonal. Multivariate Gaussian densities can be represented as products of one-dimensional densities:

$
  f_(n|y) (bold(x)) equiv f_n (bold(x) | bold(mu)_(n|y), Sigma_(n|y)) := product_(j=1)^k 1 / sqrt(2 pi (sigma_(n|y))_j^2) exp { - (bold(x)^j - bold(mu)^j)^2 / (2 (sigma_(n|y))_j^2) }
$

Each class is described by a mixture of multivariate Gaussians, each of which decomposes into a product of one-dimensional Gaussians due to feature independence. However, this doesn't mean the model assumes feature independence overall: a sum of "orthogonal" Gaussians can describe any density, though it might require more components than "non-orthogonal" Gaussians. Nevertheless, their mixture is a universal approximator.

== EM Algorithm for GMM

The Expectation-Maximization (EM) algorithm is used to find the parameters $w_(n|y)$, $bold(mu)_(n|y)$, $Sigma_(n|y)$ for $n=1..N$, $y in Y$:

1. **Initialization**: Set $w_(n|y) <- 1/N_y$, $bold(mu)_(n|y) <- ...$, $Sigma_(n|y) <- ...$

2. **Repeat** iteration $t$ to optimize distribution parameters $w_(n|y) = w_(n|y)^{(t)}$, $bold(mu)_(n|y) = bold(mu)_(n|y)^{(t)}$, $Sigma_(n|y) = Sigma_(n|y)^{(t)}$:

3. **Expectation**: For all $bold(x) in X^l$, estimate _a posteriori_ probability (weights) to come from $f_(n|y)$:
  $acute(w)'_(n|y) (bold(x)) <- ( w_(n|y) dot f_(n|y) (bold(x)) ) / (sum_(m=1)^N w_(m|y) dot f_(m|y)(bold(x)))$

4. **Maximization**: For all mixture components $n=1..N$ and classes $y in Y$, reestimate parameters:
  $w_(n|y) <- 1 / ell_y sum_(bold(x) in X^{ell}) acute(w)'_(n|y)(bold(x))$
  $bold(mu)_(n|y) <- 1 / (ell_y dot w_(n|y)) dot sum_(bold(x) in X_y) acute(w)'_(n|y)(bold(x)) dot bold(x)$
  $(Sigma_(n|y))_(j,j) <- 1 / (ell_y dot w_(n|y)) dot sum_(bold(x) in X_y) acute(w)'_(n|y)(bold(x)) dot (bold(x)^j - bold(mu)_(n|y)^j)^2$

5. **Stop** if $w_n$, $bold(theta)_n$, or $w'_n$ do not change significantly

== Connection to Metric Classifiers

The density of data can be described by a mixture of simple Gaussian distributions (features are considered independent, i.e., covariance matrices are diagonal):

$
  f(bold(x)) = sum_(n=1)^n f_n (bold(x) | bold(mu)_n, Sigma_n) \
  f_n (bold(x) | bold(mu)_n, Sigma_n) = product_(j=1)^k 1 / sqrt(2 pi sigma_(n,j)^2) exp { - (bold(x)^j - bold(mu)^j)^2 / (2 sigma_(n,j)^2) }
$

Dependency between features is already accounted for in the sense that a mixture of simple Gaussians is a universal approximator; any density can be described by a sum of "orthogonal" Gaussians, though more might be needed than with "non-orthogonal" ones.

Diagonal covariance matrices are easy to invert and have simple determinants, making GMMs quick to train. In fact, GMMs are a generalization of non-parametric kernel density estimation, where each Gaussian has tunable parameters.

The density of each class can be modeled by a mixture of simple Gaussian distributions:

$
  f(bold(x) | y) = sum_(n=1)^n w_(n|y) dot f_n (bold(x) | bold(mu)_(n|y), Sigma_(n|y)) \
  f_n (bold(x) | bold(mu)_(n|y), Sigma_(n|y)) = product_(j=1)^k 1 / sqrt(2 pi sigma_(n|y, j)^2) exp ( - {bold(x)^j - bold(mu)^j}^2 / (2 sigma_(n|y, j)^2) )
$

The estimated density is used in a Bayesian classifier:

$
  a(bold(x))
  &= arg max_(y in Y) { lambda_y dot Pr[y] dot sum_(n=1)^N_y w_(n|y) / ( (2 pi)^(k\/2) sigma_(n|y,1) dots sigma_(n|y,k) ) dot e^( - 1 / 2 {bold(x)^j - bold(mu)_(n|y)^j}^2 \/ sigma_(n|y, j)^2 ) }\
  &= arg max_(y in Y) { sum_(n=1)^N_y const_(n|y) dot e^( - 1 / 2 {bold(x)^j - bold(mu)_(n|y)^j}^2 \/ sigma_(n|y, j)^2 ) }\
  const_(n|y) &= w_(n|y) / ( (2 pi)^(k\/2) sigma_(n|y,1) dots sigma_(n|y,k) )
$

This algorithm explicitly contains an RBF kernel:

$
  a(bold(x)) = arg max_(y in Y) { sum_(n=1)^N_y const_(n|y) dot e^( -1 / 2 rho(bold(x), bold(mu)_(n|y)) \/ sigma^2_(n|y,j) ) }
$

Effectively, the algorithm is a metric algorithm that evaluates the metric proximity of each object to each class through a sum of weighted RBF kernels. Each RBF kernel contains a learned metric that measures the distance from the class to the object, weighing features in an optimal way:

$
  rho^2(bold(x), bold(mu)) = sum_(j=1)^k 1 / (sigma^2_j) { bold(x)^j - bold(mu)^j }^2
$

Key insights:
- The generative Gaussian Mixture Model, when substituted into a Bayesian classifier, also functions as a discriminative (metric) classification method
- GMM is a density estimation method but can be used for clustering
- The algorithm is equivalent to a three-layer neural network, where the 1st layer calculates "one-dimensional distances" from the object to the class center, the 2nd layer weighs these distances, and the 3rd chooses the best class
- GMM = generative + discriminative + neural network

The RBF (radial basis function) kernel can be used in both discriminative and generative approaches to classification:
- In SVM (discriminative), the RBF kernel is used in the kernel trick, and the method identifies support objects to build the separating surface, making it sensitive to outliers
- In EM (generative), the RBF kernel arises from parametric estimation of class densities, primarily identifying objects in cluster centers, making the algorithm robust to noise

The EM algorithm converges quickly but is sensitive to initial approximation. Selecting the number of components (hyperparameter) can be challenging, and simple heuristics may not work.

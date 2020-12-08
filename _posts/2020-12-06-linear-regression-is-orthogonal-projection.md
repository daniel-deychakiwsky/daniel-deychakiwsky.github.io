---
layout: post
title: Ordinary Least Squares is Orthogonal Projection
categories: machine-learning artificial-intelligence
author: Daniel Deychakiwsky
meta: Ordinary Least Squares is Orthogonal Projection
mathjax: true
permalink: /:title
---

This post visualizes the connection between
orthogonal projection (OP) and ordinary least squares (OLS).

* TOC
{:toc}

## Context

In order to get more out of this post, you may want to brush up on:

1. Wikipedias on [ordinary least squares], [vector projection],
$L^2$ [norm], and [mean squared error].
2. [Vladimir Mikulik's post] on "Why Linear Regression
is a projection".
2. [Andrew Chamberlain's post] on
"The Linear Algebra View of Least-Squares Regression".

## OLS & OP

### The Big Picture

[Fig 1.](#fig-1) is a compact and interactive
[visualization] that superimposes the two perspectives on a
[toy dataset](#toy-dataset). A single parameter (no bias term)
linear regression model (OLS) is seen in blue.
The corresponding orthogonal projection is seen in green and orange.
The red parabola (quadratic function) is the distance function of
the difference vector (dotted orange line) that is minimized when
the difference vector is orthogonal to $y_1$'s subspace.
Let's dive in to all of this for a better understanding.

#### Fig. 1

<iframe src="https://www.desmos.com/calculator/eowrcpdore?embed"
        width="500px"
        height="500px"
        style="border: 1px solid #ccc" frameborder=0>
</iframe>

### Toy Dataset

In order to be able to visualize everything in two dimensions
the data was kept as simple as can be, just two data points
$\in \mathbb{R}^2$.

As vectors:

$$
\vec{X_1} = \begin{bmatrix}2 \\ 1\end{bmatrix}
\quad \textrm{and} \quad
\vec{X_2} = \begin{bmatrix}3 \\ 3\end{bmatrix}
\tag{1} \label{1}
$$

As a matrix:

$$
\mathbf{X} = \begin{bmatrix}2 & 3 \\ 1 & 3\end{bmatrix}
\tag{2} \label{2}
$$

#### Fig. 2

![toy_data_x]

[Fig 2.](#fig-2) shows how the vector representations
(\ref{1}) translate to the cartesian plane as the
first dimension represents the canonical $x$
axis while the second dimension represents the
canonical $y$ axis.

### OLS Perspective

The OLS Perspective treats the learning
process as a [supervised learning] problem. Here's a
_really_ hand-wavy overview:

The model is presented inputs for which
it makes corresponding predictions (outputs). Hence the term "supervised",
an error signal is calculated by aggregating the squared difference
of the model's outputs against corresponding "true" values that have been
observed or are known a priori (a.k.a. labels).
The error signal is used to update the model parameters.
This (training) routine loops until the model parameters converge.
By minimizing the error signal, the model learns optimal parameters.

#### Fig. 3

|input|output|label|error|
|:---:|:---:|:---:|:---:|
|$2.0$|$1.2$|$1.0$|$(1.0 - 1.2)^2$|
|$3.0$|$2.1$|$3.0$|$(3.0 - 2.1)^2$|

[Fig 3.](#fig-3) shows a table of
hypothetical model outputs and corresponding
error calculations at some hypothetical
point in time in the training loop when learning
the [toy dataset](#toy-dataset).

#### Fig. 4

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[2], [3]])
y = np.array([[1], [3]])
reg = LinearRegression(fit_intercept=False).fit(X, y)
print('weights/parameters', reg.coef_)
```
```
weights/parameters array([[0.84615385]])
```

[Fig 4.](#fig-4) prints the resulting parameters (in our case, just one)
of an OLS [LinearRegression] implementation written in Python using
a popular machine-learning Python package ([scikit-learn]) after being
fit to our [toy dataset](#toy-dataset) (omitting the intercept term).

#### Fig. 5

![ols_regression_fit]

[Fig 5.](#fig-5) shows the OLS model fit to our
[toy dataset](#toy-dataset). The hyperplane (line of best fit)
has a slope of $0.84615385$ which is what we would expect per the
output of [Fig 4.](#fig-4) If we didn't omit the bias term and we
let the model learn another degree of freedom (another parameter),
the solution would yield a hyperplane that fits the data perfectly,
both data points would lie on that hyperplane.

### OP Perspective

Here’s another hand-wavy :) overview.

The OP Perspective treats the learning process
as solving a system of linear equations so we'll need to frame
our [toy dataset](#toy-dataset) as such.
We can formulate a vectorized linear equation.
The model's outputs will be a linear function of the inputs
and the learned parameter(s). We can represent our model inputs and
labels as vectors $\in \mathbb{R}^2$.

$$
\vec{Y_1} = \begin{bmatrix}2 \\ 3\end{bmatrix}
\quad \textrm{and} \quad
\vec{Y_2} = \begin{bmatrix}1 \\ 3\end{bmatrix}
\tag{3} \label{3}
$$

To disambiguate \ref{3} from \ref{1},
$\vec{Y_1}$ consists of the first dimensions of $\vec{X_1}$
and $\vec{X_2}$ while $\vec{Y_2}$ consists of the second
dimensions of $\vec{X_1}$ and $\vec{X_2}$.

#### Fig. 6

![toy_data_x_y]

[Fig 6.](#fig-6) shows how \ref{1} and \ref{3},
together, translate to the cartesian plane.

The equation we need to solve is:

$$
\vec{Y_1}\vec{\beta} \approx \vec{Y_2}
\tag{4} \label{4}
$$

It's best practice to validate the shapes of the
matrices/vectors when operating on them.
$\vec{Y_1}$ is $2 \times 1$, $\vec{\beta}$ is $1 \times 1$,
(recall that we're omitting the bias/intercept term),
and so $\vec{Y_2}$ checks out to be $2 \times 1$.

Intuitively, \ref{4} tells us that $\vec{Y_2}$ is
approximately equal to a scaled version
of $\vec{Y_1}$ and that scaling factor is our
learned parameter in $\vec{\beta}$.
The approximation is there because
$\vec{Y_1}\vec{\beta} = \vec{Y_2}$
will only be true if $\vec{Y_2}$ can be expressed as a
scaled version of $\vec{Y_1}$. In our example, it can't
(see [Fig 7.](#fig-7)).

#### Fig. 7

![scaled_vector]

[Fig 7.](#fig-7) shows a subset of scaled versions
of $\vec{Y_1}$ with the dashed green line and one randomly
chosen realization with the orange mark. That line
actually extends to infinity in both directions making up
a [vector subspace]. Notice how $\vec{Y_2}$ does not lie
anywhere on that subspace? This is why we need to approximate
a solution to the system and why we used the $\approx$
symbol in \ref{4}.



## Desmos

I built the visualizations for this post with [Desmos],
an incredible graphing web application that enables
users to visualize, study, and learn mathematics.

## References

[Vladimir Mikulik's post]: https://medium.com/@vladimirmikulik/why-linear-regression-is-a-projection-407d89fd9e3a
[Andrew Chamberlain's post]: https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b
[vector projection]: https://en.wikipedia.org/wiki/Vector_projection
[Desmos]: https://www.desmos.com/
[ordinary least squares]: https://en.wikipedia.org/wiki/Ordinary_least_squares
[norm]: https://en.wikipedia.org/wiki/Norm_(mathematics)
[mean squared error]: https://en.wikipedia.org/wiki/Mean_squared_error
[visualization]: https://www.desmos.com/calculator/gpkgalfzho
[scikit-learn]: https://scikit-learn.org/stable/
[LinearRegression]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
[supervised learning]: https://en.wikipedia.org/wiki/Supervised_learning
[vector subspace]: https://en.wikipedia.org/wiki/Linear_subspace

[toy_data_x]: assets/images/linear_regression_is_orthogonal_projection/toy_data_x.png
[toy_data_x_y]: assets/images/linear_regression_is_orthogonal_projection/toy_data_x_y.png
[scaled_vector]: assets/images/linear_regression_is_orthogonal_projection/scaled_vector.png
[ols_regression_fit]: assets/images/linear_regression_is_orthogonal_projection/ols_regression_fit.png

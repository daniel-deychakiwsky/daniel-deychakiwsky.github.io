---
layout: post
title: Matrix Factorization Sparsity
categories: machine-learning artificial-intelligence
author: Daniel Deychakiwsky
meta: Matrix Factorization Sparsity
mathjax: true
permalink: /:title
---

This post explores how implicit user-item interaction sparsity poses
a challenge for matrix factorization (model-based) collaborative filtering
recommender systems.

* TOC
{:toc}

# Primer

This post assumes understanding of the general *recommendations*
machine learning (ML) problem, basic modeling approaches,
and commonly used evaluation metrics.
For a refresher, check out Google's recommender systems mini-[course]
which introduces Content-Based Filtering (CBF),
Collaborative Filtering (CF) with Matrix Factorization (MF),
and Deep Neural Networks (DNN). The remainder of this post 
focuses on MF for **sparse implicit** user-item interactions. 

# Matrix Factorization

CF with MF is a type of model-based (parametric) algorithm that
decomposes $\mathbf{V}$ (user-item interaction matrix)
into a product of two matrices, $\mathbf{W}$ (latent user embeddings)
and $\mathbf{H}$ (latent item embeddings), by minimizing
some flavor of reconstruction error. Although MF is
inherently linear, it is a "tried and true" technique 
that is often reported to produce serendipitous recommendations.

$$
\mathbf{W}\mathbf{H}^\top \approx \mathbf{V}
\tag{1} \label{1}
$$

The algorithm exposes a hyperparameter that serves to expand or
contract the columns of $\mathbf{W}$ and the rows of $\mathbf{H}$ by the same dimension.
It can be set so that $\mathbf{W}$ and $\mathbf{H}$ become low-rank factors
of $\mathbf{V}$. This forces a compressed encoding which captures the most
important information for approximating $\mathbf{V}$ resulting in effective
latent user and item embeddings.

![mf]

# Sparsity

At production grade scale it's common to produce recommendations
for millions users and items from sparse implicit interactions,
e.g., purchases.

Imagine a scenario with $24$ unique items and $24$ unique users.
If each user purchases 3 unique items, on average,
the number of non-zeros in this interaction matrix is $24 * 3 = 72$
while the remaining entries are all zeros. That's a sparsity of
$1 - (72 / (24 * 24)) = 0.875$. In other words, $87.5\%$ of 
the interaction matrix entries are zeros.

$$
\begin{bmatrix}
0 &  0  & \ldots & 0\\
0  &  1 & \ldots & 0\\
\vdots & \vdots & \ddots & \vdots\\
1  &   0       &\ldots & 0
\end{bmatrix}
\tag{2} \label{2}
$$

MF models can handle sparsity to an extent 
but when it spills over roughly $99.5\%$, 
it can become problematic. There are strategies
to reduce sparsity. Researchers tend to subsample users
and items in such a way that some amount of coverage is
guaranteed. Another mechanism is bringing in more interaction
data from other implicit data sources. The authors of 
the famous Neural Collaborative Filtering ([NCF]) model
reported sparsity of $99.73\%$ on their Pinterest dataset,
so they subsampled users based on interactions.

> The original data is very large but highly sparse. 
For example, over 20% of users have only one pin, making it difficult
to evaluate collaborative filtering algorithms. As such, we
filtered the dataset in the same way as the MovieLens data
that retained only users with at least 20 interactions (pins).

## Why is it problematic?

ML models are only as good as the data that powers them
and MF is no exception to that rule. To demonstrate why sparsity
is a problem, let's consider a wild edge case.

Let's continue with the scenario from above, $24$ users and $24$ items, 
but let's add an evil twist so that each user has purchased only one
distinct item that no other user has purchased. We could reorder the
user indices (or rows of the interaction matrix) to arrive at the 
identity matrix, i.e., a canonical orthonormal basis $\in \mathbb{R}^{24\times24}$
that is linearly independent by definition. In other words, there isn't
much to learn as the unit vectors point in orthogonal directions.

$$
\mathbf{I}_{24} =
\begin{bmatrix}
1 &  0  & \ldots & 0\\
0  &  1 & \ldots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0  &   0       &\ldots & 1
\end{bmatrix}
\tag{3} \label{3}
$$

So what does MF learn in this case? Before we answer that, let's get even more evil.
We train the model with 24 `factors` (the hyperparameter discussed earlier),
what do you think the model will learn? It ends up learning exactly what it should;
an approximation of the interaction matrix (identity) by 
**inverting the item factor matrix**.

```python
import implicit.als as mf
import numpy as np
import scipy.sparse as sp

m = mf.AlternatingLeastSquares(factors=24)
m.fit(sp.csr_matrix(np.eye(24)))
np.allclose(m.user_factors, np.linalg.inv(m.item_factors.T), atol=0.01)
# True
```

$$
\mathbf{W}(\mathbf{H}^\top)^{-1} \approx \mathbf{I}
\tag{4} \label{4}
$$

Although the inversion is obvious given $\mathbf{W} \in \mathbb{R}^{24\times24}$, 
$\mathbf{H} \in \mathbb{R}^{24\times24}$, and $\mathbf{I}_{24}$, this degenerate case 
highlights how helpless the model is with no correlational signal within the data.

## Simulation

Speaking of "signals" and "correlations", the user-item interaction matrix
is exactly that. For whatever scenario you may find yourself in, these interactions
follow some natural generative process. If we knew that process we wouldn't
need any models to begin with!

let's generate a simulated interaction matrix by stacking harmonics
(integer multiple increasing frequencies) of a 5Hz [square wave] sampled
at 1000 Hz.

![square_wave]

```python
import numpy as np
import scipy.signal as sl

interactions = np.array([
    sl.square(2 * np.pi * f * np.linspace(0, 1, 25, endpoint=False))
    for f in range(25)
])
interactions = np.delete(interactions, 0, axis=0)
interactions = np.delete(interactions, 0, axis=1)
interactions[interactions == -1] = 0
```

By clipping this signal to the $[0, 1]$ range,
we end up with a bitmap, that is our interaction matrix.

![interactions]

The pattern that emerges from the square waves' auto
and harmonic correlations is jumping out at us because
our brains (and eyes) are pattern recognition
machines. Before we move to simulating machine
reconstruction/approximation of this interaction matrix,
don't you think you could get close to reproducing it
yourself just by looking at for a few seconds? I think so.
Do you see how certain rows are correlated with other rows
and the same across columns? The sparsity of this matrix is
$48.7\%$, more than half of the matrix entries are non-zeros.

Let's run a monte-carlo simulation to investigate the effect of
increasing sparsity on this easy-to-learn interaction matrix.
We'll report performance using standard ranking eval metrics
against a random $80\%-20\%$ train-test split on the data.

Here's the simulation algorithm's pseudocode followed by
a python implementation.

```
For a range of sparsities
    For a range of trials (monte-carlo)
        randomly sparsify interaction matrix to sparsity level
        train and eval MF model
        store eval metrics
    store trials for given sparsity
. . . 
Plot mean and std for every eval metric across sparsities
```

```python
import implicit.als as mf
import implicit.evaluation as ev
import numpy as np
import scipy.sparse as sp

n_users = 24
n_items = 24
n_factors = 24
n_sim = 24
n_el = n_users * n_items
results = []
shuffle = False
sparsities = np.arange(0.0, 0.91, 0.01)

for sparsity in sparsities:
    ranking_metrics = []
    for _ in range(n_sim):
        i = interactions.copy()
        n_z = int(n_el * sparsity)
        r_z = np.random.choice(n_el, replace=False, size=n_z)
        i[np.unravel_index(r_z, (n_users, n_items))] = 0
        trn_i, tst_i = ev.train_test_split(sp.csr_matrix(i))
        m = mf.AlternatingLeastSquares(factors=n_factors)
        m.fit(trn_i, show_progress=False)
        e = ev.ranking_metrics_at_k(m, trn_i, tst_i, show_progress=False)
        ranking_metrics.append(e)
    results.append(ranking_metrics)
```

![sparsity_sim]

[course]: https://developers.google.com/machine-learning/recommendation/collaborative/basics
[NCF]: https://arxiv.org/pdf/1708.05031.pdf
[square wave]: https://en.wikipedia.org/wiki/Square_wave

[mf]: assets/images/matrix_factorization_sparsity/mf.png
[square_wave]: assets/images/matrix_factorization_sparsity/square_wave.png
[interactions]: assets/images/matrix_factorization_sparsity/interactions.png
[sparsity_sim]: assets/images/matrix_factorization_sparsity/sparsity_sim.png

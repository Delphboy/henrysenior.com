---
title: "Oversmoothing in Graph Neural Networks"
collection: demos
permalink: /demos/oversmoothing
excerpt: "An interactive playground and worked example showing why stacking message-passing layers makes GNN node representations indistinguishable."
date: 2026-02-01
mathjax: true
---

# Why Deep GNNs Go Flat

If you've spent any time training Graph Neural Networks (GNNs), you've probably seen the pattern: the first two or three message-passing layers help, but adding more *hurts*. It's tempting to blame vanishing gradients or overfitting — and those play a part — but there's a more fundamental phenomenon at work: **oversmoothing**.

Oversmoothing is what happens when the message-passing layers of a GNN are repeated enough times that the representations of *all* nodes converge towards the same value. Once every node looks identical to the classifier, the model has lost the information it needs to tell nodes apart, and performance collapses.

## Try it yourself

This is an interactive playground: each node's feature vector is rendered directly as its colour (three channels, capped at 0–255). Hit **Forward pass** to run one message-passing layer — every node replaces its vector with the average of its neighbours' (and its own) — then press it again, and again, and watch what happens.

[Open the demo as a standalone page](/assets/demos/oversmoothing-lab.html){:target="_blank" rel="noopener"}

{% include demos/oversmoothing-lab.html %}

## Where does the smoothing come from?

Every message-passing layer does roughly the same thing:

$$h_v^{(k+1)} = \text{UPDATE}\left(h_v^{(k)}, \text{AGG}\left(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)\right)$$

In its simplest form — the one used by GCNs — this is just a *neighbourhood average*:

$$h_v^{(k+1)} = \sigma\left(W^{(k)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{h_u^{(k)}}{\sqrt{\deg(u)\deg(v)}}\right)$$

In matrix form, ignoring the non-linearity and weight matrix, one layer applies the row-normalised adjacency matrix $\hat{A} = D^{-1}A$ to the feature matrix $X$. Stacking $k$ layers applies $\hat{A}^k X$ — i.e. **repeated averaging of a node's features with its neighbours' features** (and their neighbours' features, and so on).

Repeated averaging is a smoothing process: think of heat diffusing through a graph until every node reaches the same temperature. For a connected, non-bipartite graph, $\hat{A}^k X$ converges to a rank-1 matrix as $k \to \infty$ — every row ends up proportional to the stationary distribution of the random walk, and all node representations become (scaled) copies of one another. This convergence is *exponential* in the number of layers, which is why the effect shows up so quickly.

## See it happen

Here's a two-cluster graph: 50 nodes per cluster, sparse intra-cluster connections, and just a single edge linking the clusters together. I initialise each node with a random 16-dimensional feature vector and repeatedly apply $\hat{A}X$ (no weights, no non-linearities — the smoothing alone is enough). After each step I measure the mean pairwise Euclidean distance between node representations, split into within-cluster and between-cluster pairs:

![Pairwise distances between node representations collapse as message-passing layers are stacked.](/images/demos/oversmoothing.png)

Both curves plummet towards zero. The within-cluster distance dies almost immediately (that's the *useful* smoothing — it pools information within a community), but the between-cluster distance follows it down. After ~15 layers the two clusters are indistinguishable: the between-cluster distance has dropped by **98%**. Any classifier looking at these representations would see one giant blob of identical nodes.

## Reproduce it yourself

The figure above is ~30 lines of NumPy. Run it and watch the distances collapse:

```python
import numpy as np

rng = np.random.default_rng(42)

# Two clusters of 50 nodes, sparsely connected within, one edge between.
n_per, n = 50, 100
p_in, p_out = 0.35, 0.05
adj = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        same = (i // n_per) == (j // n_per)
        if rng.random() < (p_in if same else p_out):
            adj[i, j] = adj[j, i] = 1.0
adj[0, n_per] = adj[n_per, 0] = 1.0  # keep the graph connected

# Row-normalised adjacency: one message-passing layer is X <- P @ X.
P = adj / np.clip(adj.sum(axis=1, keepdims=True), 1, None)

# Random initial features, unit norm.
X = rng.normal(size=(n, 16))
X = X / np.linalg.norm(X, axis=1, keepdims=True)

labels = np.arange(n) // n_per
intra = (labels[:, None] == labels[None, :]) & ~np.eye(n, dtype=bool)
inter = labels[:, None] != labels[None, :]

Xk = X.copy()
for k in range(31):
    d = np.linalg.norm(Xk[:, None, :] - Xk[None, :, :], axis=-1)
    print(f"layer {k:2d} | within-cluster distance: {d[intra].mean():.4f}"
          f" | between-cluster distance: {d[inter].mean():.4f}")
    Xk = P @ Xk
```

## Why this matters

Oversmoothing is the main reason deep GNN architectures are hard to train, and it's driven a lot of research into *making GNNs go deeper*:

- **Skip connections & jumping knowledge** (Xu et al., 2018) — let the final layer combine representations from *all* depths, so information doesn't have to survive every layer.
- **PairNorm** (Zhao & Akoglu, 2019) — renormalise node features after each layer so they don't converge.
- **GCNII** (Chen et al., 2020) — identity mapping + initial residual connections, the same trick that lets ResNets train hundreds of layers.
- **DropEdge** (Rong et al., 2020) — randomly drop edges during training to slow down the information flow.

It's also a useful lens for thinking about your own architectures: if your model only has a handful of layers you probably won't hit oversmoothing head-on, but understanding *why* deeper isn't automatically better stops you from fighting the wrong battle when your deep GNN underperforms.

## Further reading

- Li, Han, Wu — [Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning](https://arxiv.org/abs/1801.07606), AAAI 2018 — the paper that named the problem.
- Oono, Suzuki — [Graph Neural Networks Exponentially Lose Expressive Power for Node Classification](https://arxiv.org/abs/1905.10947), ICLR 2020 — formal proof of the exponential convergence.
- Cai, Wang — [A Note on the Iterative Global Convergence of GNNs](https://arxiv.org/abs/2004.11404) — fixed-point characterisation of the collapse.

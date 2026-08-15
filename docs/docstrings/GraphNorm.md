Graph normalization ([Cai et al., 2021](https://proceedings.mlr.press/v139/cai21e.html)).

Normalizes each feature over the nodes of its graph with a learnable mean scale:

\[
y_i = \gamma \odot \frac{x_i - \alpha \odot \mu_{g_i}}
{\sqrt{\operatorname{mean}_{j \in g_i}[(x_j - \alpha \odot \mu_{g_i})^2] + \epsilon}} + \beta.
\]

Parameters
----------
dim : int
    Number of node features.
eps : float, default=1e-5
    Positive constant added to the variance for numerical stability.
use_bias : bool, default=True
    Whether to include a learnable output bias.

Attributes
----------
scale : Param
    Per-feature output scale, initialized to ones.
b : Param | None
    Per-feature output bias, initialized to zeros. `None` when `use_bias=False`.
mean_scale : Param
    Per-feature graph mean scale, initialized to ones.

Example
-------
```python
num_nodes, dim, num_graphs = 5, 16, 2
x = jnp.ones((num_nodes, dim))
graph_ids = jnp.array([0, 0, 1, 1, 1])

norm = gnn.GraphNorm(dim)
y = norm(x)  # one graph: (5, 16) -> (5, 16)
y = norm(x, graph_ids, num_graphs)  # packed batch: (5, 16) -> (5, 16)
```

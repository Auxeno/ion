# GNN

Graph neural network layers and operations, imported from `ion.gnn`. Graphs are plain arrays in COO format (node features plus `senders`/`receivers` edge indices), so the native JAX transforms work directly and there is no custom graph object to learn.

```python
from ion import gnn

gcn = gnn.GCNConv(in_dim=16, out_dim=32, key=key)
y = gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
```

## What's inside

- **[Reference](reference.md)** — the array format, self-loops, shape labels, and batching. Start here; the layers assume it.
- **Layers** — the convolutions: [GCN](layers/gcn.md), [GAT](layers/gat.md) (including GATv2), and [GIN](layers/gin.md).
- **[Ops](ops.md)** — segment reductions, graph-level pooling, and graph-building helpers (`add_self_loops`, `batch_graphs`).

Full worked examples: [Node classification on Cora](https://github.com/auxeno/ion/blob/main/examples/gnn_cora.py) and [Molecular property prediction on BBBP](https://github.com/auxeno/ion/blob/main/examples/gnn_bbbp.ipynb).

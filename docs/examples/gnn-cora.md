# GNN on Cora

Semi-supervised node classification on the Cora citation network: 2,708 papers connected by 10,556 citation edges, each described by a 1,433-dim bag-of-words vector, to be sorted into 7 subfields. The task is transductive, with only 140 labeled nodes for training. The script trains a `GCNConv` model (fixed degree-weighted averaging) and a `GATConv` model (learned neighbour weighting) on the same split.

Points of interest:

- Graphs are passed as `senders`/`receivers` edge-index arrays, not adjacency matrices.
- `gnn.add_self_loops` appends self-edges so each node aggregates its own features alongside its neighbours'.
- The same training loop drives both `GCNConv` and `GATConv`: only the model class changes, which makes the two convolutions directly comparable.
- Loss is masked to the training nodes; accuracy is evaluated on the held-out test mask over the single shared graph.

## Source

[examples/gnn_cora.py](https://github.com/auxeno/ion/blob/main/examples/gnn_cora.py) on GitHub.

```python title="examples/gnn_cora.py" linenums="1"
--8<-- "examples/gnn_cora.py"
```

## Output

```bash
uv run --group examples examples/gnn_cora.py
```

```
GCNConv: 100%|██████████| 10/10 [00:01<00:00, 6.50it/s]
  loss: 1.7586  test accuracy: 77.40%
GATConv: 100%|██████████| 10/10 [00:02<00:00, 3.89it/s]
  loss: 1.6904  test accuracy: 79.10%
```

# Changelog

## 0.5.0

- **State Space Models.** New `S4D` (Gu et al., 2022), `S5` (Smith et al., 2023), and `LRU` (Orvieto et al., 2023) layers with matching cell variants. `LRU` uses `lax.associative_scan` for parallel sequence processing; `S4D` and `S5` use `lax.scan`.
- **Vanilla RNN.** New `RNN` and `RNNCell` layers alongside the existing LSTM and GRU.
- **Type checker compatibility.** `Param` now exposes JAX array type information so static type checkers correctly resolve array operations.
- **Pathfinder demo.** New example notebook training an S4D model on the Pathfinder-128 long-range task.
- **Molecular property prediction demo.** New example notebook training a GNN on the BBBP molecular benchmark.

## 0.4.1

- **GATv2Conv.** New dynamic graph attention layer (Brody et al., 2022) with strictly more expressive attention than GATConv.
- **Edge features.** `GATConv` and `GATv2Conv` support optional per-edge features via `edge_dim` constructor parameter and `x_edge` call argument.
- **Input shape guards.** All GNN layers unpack input shapes at the top of `__call__` to catch rank mismatches early.
- **Renamed** `GraphConv` to `GCNConv` and `GraphAttention` to `GATConv` to match standard GNN naming conventions.

## 0.4.0

- **Graph Neural Networks.** New `ion.gnn` module with `GCNConv` (Kipf & Welling, 2017) and `GATConv` (Velickovic et al., 2018) layers. Graphs are represented as plain arrays (`x`, `senders`, `receivers`) with no custom data structures. Includes `segment_softmax` and `add_self_loops` utilities.
- **Cora demo.** New example training both GCNConv and GATConv on semi-supervised node classification.
- **GNN docs.** New [gnn.md](docs/gnn.md) covering graph representation, shape annotations, weight init, and batching.

## 0.3.0

- **Optimizer.** New `ion.Optimizer` wraps an optax transform with Param-aware updates,
  replacing `apply_updates` as the third core abstraction alongside `Param` and `Module`.
  Frozen params are automatically partitioned so no optimizer memory is wasted on them.
- **Breaking:** Removed `ion.apply_updates`. Use new `ion.Optimizer` instead.
- **Dependency:** `optax` is now a runtime dependency (previously dev-only).
- Fix Treescope not rendering arrays.

## 0.2.7

- **Faster pytree registration.** Module flatten/unflatten are now defined once at class creation and the pytree structure is cached, improving speed through JAX transforms.
- Moved `_Static` into `module.py` (internal cleanup).
- Documentation updates.

## 0.2.6

- **Removed BatchNorm.** Dropped `BatchNorm` from the library.
- Added warning when hydrating models with mismatched parameter dtypes.
- Fixed dropout edge case when `p >= 1`.
- Treescope visualization now defaults to showing only Modules and Params.
- Documentation and README updates.

## 0.2.5

- Added `__call__` method to `Module` so static type checkers see generic modules as callable.
- Added dynamic `__version__` to package root.
- Added Python 3.14 as supported.
- Improved test coverage and added automated pip install test.
- Documentation and README updates.

## 0.2.4

- **Dtype casting.** New `Module.astype(dtype)` method and `ion.tree.cast(pytree, dtype)` utility for casting all parameters in a model or pytree to a target dtype (e.g. `float16`, `bfloat16`).
- **TinyStories GPT demo.** New example notebook training a small GPT on the TinyStories dataset.
- Added NumPy-style docstrings to all public functions.
- Fixed attention mask shape annotations.
- Documentation and README updates.

## 0.2.3

- Added GitHub Actions CI workflow (tests across Python 3.11, 3.12, 3.13 with linting and type checking).
- Added GitHub Actions workflow for automated PyPI publishing on release.
- Switched to git-tag-based versioning via `hatch-vcs`.
- Added CI status badge to README.

## 0.2.2

- Fixed README image path and install command for PyPI.
- Added hatchling wheel build config.

## 0.2.1

- **Strict input shapes for structural layers.** Conv, ConvTranspose, Pool, GroupNorm, Attention, Transformer, LSTM, and GRU now require exactly the right number of dimensions (one batch dim) and error on incorrect rank. Previously these layers silently reshaped arbitrary leading batch dims, masking shape bugs. Use `jax.vmap` for multiple batch dimensions. Pointwise layers (Linear, LayerNorm, RMSNorm, Embedding, etc.) are unaffected.
- Removed `lax.stop_gradient` wrapping in `BatchNorm.update()` running stat updates (unnecessary since running stats are non-Param arrays).
- Expanded test suite with 56 new tests covering documented sharp edges, safety guards under JIT, and transform compositions.

## 0.2.0

- **Native JAX transforms.** Removed `ion.grad` and `ion.value_and_grad`. `Param` now applies `stop_gradient` automatically for frozen params via `__jax_array__`, so `jax.grad`, `jax.value_and_grad`, `jax.jit`, `jax.vmap`, `jax.jacobian`, `jax.hessian`, etc. all work directly with no wrappers.
- Renamed `Param.value` to `Param._value` (private; use `jnp.asarray(param)` instead of accessing `._value` directly or it can cause problems with autograd).
- Deleted `ion/transforms.py`.

# Changelog

## 0.9.0

- **Path-based model surgery with `Module.at`.** New `at` property, the model version of
  JAX's `x.at[idx].set(v)`: navigate to any field, index, or key, then `set` a new value,
  e.g. `model.at.blocks[2].attn.w_q.set(new_w)`. Works uniformly for arrays, Params,
  sub-modules, static values, and structural changes like `b=None`; rebuilds only the nodes
  along the path and shares untouched subtrees with the original. Chain for several edits.
- **`Module.replace` removed.** `at` subsumes it: `model.replace(b=None)` becomes
  `model.at.b.set(None)`, and multi-field updates chain as `model.at.a.set(x).at.b.set(y)`.

## 0.8.1

- **Style consistency pass.** Source, tests and vault docs audited against the house style
  (docstring formats, RNG key naming, import ordering, doctest shape comments); newer code
  brought in line with the established layers. No behavioural changes.
- **Error messages compacted and unified.** Every `raise` is now at most three lines with a
  single-line message, shaped `name (value) must be ...` with no trailing period. Some
  messages were reworded to fit (e.g. checkpoint mismatch errors no longer list the file's
  available keys).
- **`MLP.__call__` annotation fixed.**
- **Line length enforced.** Ruff now checks `E501` with a 100 character limit.

## 0.8.0

- **`GINConv`.** Graph Isomorphism Network layer (Xu et al., 2019): sum-aggregates neighbor
  features and applies a caller-supplied MLP to `(1 + eps) * x + aggregated`. `eps` defaults
  to a fixed `0.0`; `train_eps=True` makes it a learnable scalar. Takes no `key` since the
  update network is built by the caller. Do not add self-loops; own features enter through
  the `(1 + eps)` term.
- **Graph-level readout pooling.** `ion.gnn` gains `mean_pool`, `sum_pool` and `max_pool`,
  which pool node features `(n, d)` into per-graph vectors `(g, d)` via a `graph_ids`
  assignment array, enabling graph classification. `max_pool` returns zeros for empty graphs
  rather than `segment_max`'s `-inf` fill; `mean_pool` likewise guards empty graphs to zeros.
- **`segment_mean` and a coherent segment namespace.** New `segment_mean` (per-segment mean;
  JAX ships sum/max/min/prod but no mean), and `segment_sum`, `segment_max`, `segment_min`
  and `segment_prod` re-exported from `jax.ops`, so every segment reduction is reachable as
  `gnn.segment_*` alongside the existing `segment_softmax`.
- **`batch_graphs`.** Packs a list of graphs into one disconnected graph: node features are
  concatenated, edge indices offset by cumulative node counts, and a `graph_ids` array is
  returned for the pooling functions. Message passing on the union is exactly equivalent to
  processing each graph separately. Call it outside `jit`; per-batch shapes vary and each new
  shape recompiles.

## 0.7.0

- **Nested containers in Module fields are now pytree children.** Field classification
  previously looked only one level deep, so a `list[list[Module]]` or `dict[str, list[Module]]`
  field was silently treated as static metadata: its params were invisible to `jax.grad`,
  skipped by `ion.Optimizer`, and baked into `jit` traces as compile-time constants, meaning
  such models silently did not train. Containers are now classified by their contents at any
  depth. Containers holding only modules and arrays traverse natively; mixed containers wrap
  non-array elements (callables, strings) as static metadata at any nesting depth. Flatten of
  flat all-module containers is also slightly faster. Empty containers remain static.
- **Tree calls migrated to the `jax.tree` namespace.** All `jax.tree_util.tree_*` operation
  calls (`tree_leaves`, `tree_flatten_with_path`) now use their `jax.tree` equivalents, which
  JAX documents as the current API (the `tree_util` spellings are listed as legacy).
  Registration APIs (`register_dataclass`, `register_pytree_with_keys`) remain on
  `jax.tree_util`, their supported home. No user-facing change.
- **Attention rebuilt on `jax.nn.dot_product_attention`.** `SelfAttention` and `CrossAttention`
  delegate the scaled-dot-product core to JAX's kernel. The softmax now runs in float32 even
  under `bfloat16` (previously it ran in the input dtype, losing precision in mixed-precision
  training), and cuDNN flash attention is used automatically on supported GPUs.
- **Grouped-query and multi-query attention.** `SelfAttention` and `TransformerBlock` take
  `num_kv_heads`: fewer key/value heads than query heads gives grouped-query attention, `1`
  gives multi-query. `num_heads` must be divisible by it; the default (`None`) keeps standard
  multi-head attention.
- **Sliding-window attention.** `SelfAttention` takes `window` (a symmetric `int` or a
  `(left, right)` tuple) for local attention; `None` (default) is full attention.
- **Breaking: `SelfAttention.w_qkv` split into `w_q` and `w_kv`.** The fused QKV weight is now a
  separate query projection `w_q` `(dim, num_heads, head_dim)` and a fused key/value projection
  `w_kv` `(2, dim, num_kv_heads, head_dim)`, required to let key/value heads differ from query
  heads. `CrossAttention.w_kv` likewise moves its split axis first. Total parameter count is
  unchanged, but checkpoints saved before 0.7.0 no longer load into the new layers.
- **Breaking: `SelfAttention` constructor argument order.** `num_kv_heads` is inserted after
  `num_heads` and `window` after `causal`, so positional arguments past `num_heads` shift.
- **Fully masked attention rows output the mean of the value vectors.** A query position with
  no attendable positions now returns the mean of the values (dot_product_attention's
  behaviour) rather than the zeros introduced in 0.6.1. Output and gradients stay finite; mask
  these positions out of the loss as usual.
- **Migration.** Access attention weights via `w_q` / `w_kv` instead of `w_qkv`, and pass
  `bias`, `causal`, and initializers by keyword (or update positional calls for the new
  `num_kv_heads` and `window` slots). Re-save any checkpoints written before 0.7.0.

## 0.6.2

- **Dropout validates `p` at construction.** `p` outside `[0, 1]` now raises `ValueError`;
  previously out-of-range values silently corrupted outputs (negative `p` rescaled
  activations with no mask applied).
- **GAT layers require `x_edge` and `edge_dim` together.** `GATConv` and `GATv2Conv` now
  raise `ValueError` when one is provided without the other; previously an edge-aware layer
  called without `x_edge` silently ignored its edge parameters, and passing `x_edge` to a
  non-edge layer failed with a cryptic shape error.
- **Pooling padding must be smaller than the kernel.** `MaxPool` and `AvgPool` now raise
  `ValueError` at construction when a `padding` value is greater than or equal to the
  corresponding `kernel_shape` entry; previously such a config produced a window landing
  entirely in padding, which gave `AvgPool` a `0 / 0 = NaN` output and `MaxPool` a `-inf`
  output. String paddings (`'SAME'`, `'VALID'`) are unaffected.

## 0.6.1

- **Breaking: `rope` and `apply_rope` replaced by the `RoPE` module.** `nn.RoPE(theta)`
  applies rotary embeddings directly to query or key tensors. Frequency tables are computed
  on the fly from the input shape and constant-folded under `jit`, so there is no `max_len`,
  no stored tables, and no `key`. Tables are computed in float32 and cast to the input dtype.
- **Migration.** `cos, sin = nn.rope(s, d)` followed by `nn.apply_rope(q, cos, sin)` becomes
  `rope = nn.RoPE()` then `rope(q)`.
- **Fully masked attention rows output zeros.** `SelfAttention` and `CrossAttention` now mask
  via `jax.nn.softmax(..., where=)`: a query position with no attendable positions outputs
  zeros instead of NaN. Rows with at least one attendable position are unchanged.
- **Exact `segment_softmax` normalization.** Removed the `1e-6` denominator epsilon, which
  biased every attention weight slightly low; per-segment weights now sum to exactly 1.
- **Class-level field defaults work with custom `__init__`.** A Module field with a class
  default (`eps: float = 1e-5`) no longer needs to be assigned in a custom `__init__`;
  construction previously crashed with a bare `KeyError`. A field with no default that is
  never assigned now raises a clear `AttributeError` naming the field instead.
- **NamedTuple Module fields work.** A Module field holding a NamedTuple of array-like
  elements (Params, Modules, arrays) previously crashed on flatten/unflatten because
  containers were rebuilt by passing an iterable to the constructor; NamedTuples are now
  rebuilt with positional fields.
- **`Optimizer.update` rejects trainability changes with a clear error.** The
  frozen/trainable partition is baked into the optimizer state at construction, so calling
  `freeze()`/`unfreeze()` on the model and reusing the old optimizer previously failed with a
  cryptic pytree mismatch (or, for stateless transforms, silently left unfrozen params
  untrained). `update()` now checks the model structure against the one recorded at
  construction and raises `ValueError` telling you to create a new Optimizer.

## 0.6.0

- **Breaking: `dtype` removed from all layer and block constructors.** Parameters are now
  created in JAX's default float dtype (`float32`, or `float64` under `jax_enable_x64`).
  Control precision with `model.astype(...)` / `ion.astype(...)` instead: keep float32 master
  params and cast to `bfloat16` inside the loss for mixed-precision training, or cast once for
  full low-precision inference. See the new precision section in
  [internals.md](docs/internals.md).
- **Migration.** Drop any `dtype=...` argument from layer construction. Because `dtype` was a
  positional parameter (before the initializer arguments), code passing later arguments
  positionally must drop the `dtype` slot too; keyword arguments are unaffected.
- SSM layers (LRU, S4D, S5) keep their `complex64` recurrent state, and the `sinusoidal`,
  `alibi`, and `rope` positional-encoding functions keep their `dtype` argument (they build
  arrays rather than parameters).
- **Breaking: `MLP` takes a single `dims` sequence.** `MLP(in_dim, out_dim, hidden_dim,
  num_hidden_layers, ...)` is now `MLP(dims, ...)` where `dims` lists every layer width
  from input to output: `MLP([3, 64, 64, 1], key=key)` builds two hidden layers of 64.
  Hidden widths may now vary per layer (`MLP([784, 512, 128, 10], key=key)`), and a single
  linear layer is `MLP([3, 1], key=key)`.
- **Migration.** `MLP(i, o, h, n, ...)` becomes `MLP([i, *[h] * n, o], ...)`. All other
  arguments (`activation`, `final_activation`, `bias`, initializers, `key`) are unchanged.

## 0.5.3

- **Attention mask shapes.** `SelfAttention` and `CrossAttention` masks may be `(s, t)`,
  `(b, s, t)`, or `(b, h, s, t)`. Rank-3 masks now apply per batch element; previously they
  broadcast over heads, silently misapplying the mask when batch size equalled head count.
- **Correct ALiBi slopes.** `alibi` uses the paper's geometric head slopes `2^(-8i/n)`.
  The previous fixed ratio of 0.5 was only correct for 8 heads.
- **GroupNorm rank handling.** `GroupNorm` normalizes over trailing dimensions like
  `LayerNorm`, supporting unbatched inputs and arbitrary leading batch dims.
- **GAT Glorot init.** `GATConv` and `GATv2Conv` multi-head projections initialize with
  correct Glorot fans; previously variance shrank with head count and input width.
  Custom `w_init` now receives the flat `(in_dim, out_dim)` shape.
- **bfloat16 checkpoints.** `save` and `load` round-trip extension dtypes (`bfloat16`,
  `float8`) exactly; previously they were silently unrecoverable. `load` also appends
  `.npz` to the path when missing, matching `save`.
- **Recurrent initial state dtype.** `initial_state` on RNN, LSTM, and GRU cells follows
  the parameter dtype instead of always returning float32.

## 0.5.2

- **Per-field optimizer transforms.** `Optimizer` accepts a dict mapping field names
  (or tuples of field names) to separate optax transforms, enabling different learning
  rates or schedules for different parts of a model (e.g. generator vs discriminator).
- Fixed SSM demo notebook links.
- Improved generic test coverage (bfloat16 tests, refactored layer type grouping).

## 0.5.1

- **Edge masking.** `GATConv` and `GATv2Conv` accept an optional `edge_mask` argument to
  zero out attention on selected edges. Masked edges receive zero attention weight and
  their edge features (if any) are zeroed.
- **Numerically stable segment softmax.** `segment_softmax` now handles empty segments
  (e.g. from full masking) cleanly instead of producing NaN.
- **RL demos.** New DQN (Atari), PPO, and PQN example scripts and notebooks.

## 0.5.0

- **State Space Models.** New `S4D` (Gu et al., 2022), `S5` (Smith et al., 2023), and `LRU` (Orvieto et al., 2023) layers with matching cell variants. All three use `lax.associative_scan` for parallel sequence processing.
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

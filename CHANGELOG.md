# Changelog

## 0.11.0

- **Layer weight inits now default to uniform variance scaling, matched to the following activation.** `Linear`, `Conv`, `ConvTranspose`, `GCNConv`, `SelfAttention`/`CrossAttention`, and the SSM layers (`S4D`, `S5`, `LRU`) default to `glorot_uniform` (gain 1) instead of `he_normal`/`truncated_normal(0.02)`; `MLP` uses `he_uniform`, since it applies ReLU between its layers. He assumes a ReLU that a bare projection or attention does not have, so gain-1 Glorot is the safer default and matches PyTorch and Flax. Pass an explicit `w_init` to restore the old behaviour.
- **`Embedding` and `LearnedPositionalEmbedding` init scales with dimension.** The default is now `variance_scaling(1.0, "fan_in", "uniform", out_axis=0)`, giving std `1/sqrt(dim)` so each row starts near unit norm regardless of `dim` and independent of vocabulary or sequence length. Replaces the dimension-blind `truncated_normal(0.02)` constant, matching Flax's `Embed`.

## 0.10.2

- **`sinusoidal` takes `theta`.** The frequency base is now a parameter (`nn.sinusoidal(seq_len, dim, theta=10_000.0)`) instead of hardcoded, matching `RoPE`.

## 0.10.1

- **Type steps in `model.at`.** Indexing `at` with a layer type applies the rest of the path to every matching node in the subtree: `model.at[nn.Dropout].p.set(0.5)`, or `model.at.encoder[nn.Dropout].p.set(0.5)` to scope it. Raises `ValueError` if nothing matches.
- **`Sequential` takes an optional `key`.** `model(x, key=key)` splits the key per layer and forwards a subkey to any layer whose signature accepts `key` (e.g. `Dropout`).

## 0.10.0

- **New `.ion` checkpoint format.** `ion.save`/`ion.load` write `.ion` files, a safetensors-style container replacing `.npz`, fully in-house (stdlib + numpy + `ml_dtypes`, no new dependencies); safetensors tools can read Ion checkpoints directly. `bfloat16`, `float8`, and `complex64` are stored natively. Tensor names are clean tree paths (no `._value` suffix); format version is checked on load. Old `.npz` checkpoints are incompatible.
- **`GroupNorm` requires `num_spatial_dims`.** The old default of `0` silently diverged from standard GroupNorm on spatial data. Pass `num_spatial_dims=0` for the old behaviour.
- **`GATConv`/`GATv2Conv` take `x_edge` and `edge_mask` as keyword-only.** Easy to confuse positionally: `gat(x, senders, receivers, x_edge=x_edge, edge_mask=mask)`.
- **Attention weights stored flat 2D with separate `w_k` and `w_v`.** `SelfAttention`/`CrossAttention` replace `w_kv` with separate key and value projections, all weights at flat 2D shapes. Fixes `w_init` fan calculations, which previously inflated with head count. Checkpoints from before don't load.
- **`CrossAttention` takes `context_dim`.** Keys/values may come from a context with a different feature dim than the query; `None` (default) matches `dim`. Breaking: positional args after `num_heads` shift by one.
- **`TransformerBlock`/`CrossTransformerBlock` removed.** Ion ships primitives, not architecture opinions. Compose from `SelfAttention`/`CrossAttention`, `LayerNorm`, and `Linear` instead.

## 0.9.0

- **Path-based model surgery with `Module.at`.** New `at` property, the model version of JAX's `x.at[idx].set(v)`: navigate to any field, index, or key, then `set` a new value, e.g. `model.at.blocks[2].attn.w_q.set(new_w)`. Shares untouched subtrees with the original; chain for multiple edits.
- **`Module.replace` removed.** `at` subsumes it: `model.replace(b=None)` becomes `model.at.b.set(None)`.
- **Core internals simplified.** `module.py`, `param.py`, `checkpoint.py`, and `optimizer.py` slimmed with equivalent behaviour. No user-facing change.

## 0.8.1

- **Style consistency pass.** Source, tests, and vault docs audited against house style. No behavioural changes.
- **Error messages compacted and unified.** Every `raise` is now at most three lines, shaped `name (value) must be ...`.
- **`MLP.__call__` annotation fixed.**
- **Line length enforced.** Ruff now checks `E501` with a 100 character limit.

## 0.8.0

- **`GINConv`.** Graph Isomorphism Network layer (Xu et al., 2019): sum-aggregates neighbor features and applies a caller-supplied MLP to `(1 + eps) * x + aggregated`. `train_eps=True` makes `eps` learnable.
- **Graph-level readout pooling.** `ion.gnn` gains `mean_pool`, `sum_pool`, and `max_pool`, pooling node features `(n, d)` into per-graph vectors `(g, d)` via `graph_ids`.
- **`segment_mean` and a coherent segment namespace.** New `segment_mean`; `segment_sum`/`max`/`min`/`prod` re-exported from `jax.ops` alongside `segment_softmax`.
- **`batch_graphs`.** Packs a list of graphs into one disconnected graph for batched message passing; returns `graph_ids` for the pooling functions.

## 0.7.0

- **Nested containers in Module fields are now pytree children.** Field classification previously looked only one level deep, so e.g. a `list[list[Module]]` field was silently treated as static metadata and such models silently did not train. Containers are now classified by their contents at any depth.
- **Tree calls migrated to the `jax.tree` namespace.** `tree_leaves`, `tree_flatten_with_path`, etc. now use their `jax.tree` equivalents. No user-facing change.
- **Attention rebuilt on `jax.nn.dot_product_attention`.** Softmax now runs in float32 even under `bfloat16`; cuDNN flash attention is used automatically on supported GPUs.
- **Grouped-query and multi-query attention.** `SelfAttention`/`TransformerBlock` take `num_kv_heads`: fewer KV heads than query heads gives GQA, `1` gives MQA.
- **Sliding-window attention.** `SelfAttention` takes `window` (a symmetric `int` or `(left, right)` tuple) for local attention.
- **Breaking: `SelfAttention.w_qkv` split into `w_q` and `w_kv`.** Required to let key/value heads differ from query heads. `CrossAttention.w_kv` also moves its split axis first. Checkpoints from before 0.7.0 no longer load.
- **Breaking: `SelfAttention` constructor argument order.** `num_kv_heads` inserted after `num_heads`, `window` after `causal`.
- **Fully masked attention rows output the mean of the value vectors** instead of zeros, matching `dot_product_attention`'s behaviour.
- **Migration.** Access attention weights via `w_q`/`w_kv` instead of `w_qkv`; re-save checkpoints written before 0.7.0.

## 0.6.2

- **Dropout validates `p` at construction.** `p` outside `[0, 1]` now raises `ValueError` instead of silently corrupting outputs.
- **GAT layers require `x_edge` and `edge_dim` together.** `GATConv`/`GATv2Conv` now raise `ValueError` when one is given without the other.
- **Pooling padding must be smaller than the kernel.** `MaxPool`/`AvgPool` raise `ValueError` at construction instead of producing NaN/-inf outputs.

## 0.6.1

- **Breaking: `rope` and `apply_rope` replaced by the `RoPE` module.** `nn.RoPE(theta)` applies rotary embeddings directly; frequency tables are computed on the fly and constant-folded under `jit`.
- **Migration.** `cos, sin = nn.rope(s, d); nn.apply_rope(q, cos, sin)` becomes `nn.RoPE()(q)`.
- **Fully masked attention rows output zeros** instead of NaN, via `jax.nn.softmax(..., where=)`.
- **Exact `segment_softmax` normalization.** Removed the `1e-6` denominator epsilon; weights now sum to exactly 1.
- **Class-level field defaults work with custom `__init__`.** A Module field with a class default no longer needs to be assigned in a custom `__init__`.
- **NamedTuple Module fields work.** Previously crashed on flatten/unflatten.
- **`Optimizer.update` rejects trainability changes with a clear error** instead of a cryptic pytree mismatch.

## 0.6.0

- **Breaking: `dtype` removed from all layer and block constructors.** Params are now created in JAX's default float dtype. Use `model.astype(...)` for precision control.
- **Migration.** Drop `dtype=...` from layer construction; positional args after it shift.
- SSM layers (LRU, S4D, S5) keep their `complex64` recurrent state; `sinusoidal`, `alibi`, and `rope` keep their `dtype` argument.
- **Breaking: `MLP` takes a single `dims` sequence.** `MLP(in_dim, out_dim, hidden_dim, num_hidden_layers, ...)` is now `MLP(dims, ...)`, e.g. `MLP([3, 64, 64, 1], key=key)`.
- **Migration.** `MLP(i, o, h, n, ...)` becomes `MLP([i, *[h] * n, o], ...)`.

## 0.5.3

- **Attention mask shapes.** `SelfAttention` and `CrossAttention` masks may be `(s, t)`, `(b, s, t)`, or `(b, h, s, t)`. Rank-3 masks now apply per batch element; previously they broadcast over heads, silently misapplying the mask when batch size equalled head count.
- **Correct ALiBi slopes.** `alibi` uses the paper's geometric head slopes `2^(-8i/n)`. The previous fixed ratio of 0.5 was only correct for 8 heads.
- **GroupNorm rank handling.** `GroupNorm` normalizes over trailing dimensions like `LayerNorm`, supporting unbatched inputs and arbitrary leading batch dims.
- **GAT Glorot init.** `GATConv` and `GATv2Conv` multi-head projections initialize with correct Glorot fans; previously variance shrank with head count and input width. Custom `w_init` now receives the flat `(in_dim, out_dim)` shape.
- **bfloat16 checkpoints.** `save` and `load` round-trip extension dtypes (`bfloat16`, `float8`) exactly; previously they were silently unrecoverable. `load` also appends `.npz` to the path when missing, matching `save`.
- **Recurrent initial state dtype.** `initial_state` on RNN, LSTM, and GRU cells follows the parameter dtype instead of always returning float32.

## 0.5.2

- **Per-field optimizer transforms.** `Optimizer` accepts a dict mapping field names (or tuples of field names) to separate optax transforms, enabling different learning rates or schedules for different parts of a model (e.g. generator vs discriminator).
- Fixed SSM demo notebook links.
- Improved generic test coverage (bfloat16 tests, refactored layer type grouping).

## 0.5.1

- **Edge masking.** `GATConv` and `GATv2Conv` accept an optional `edge_mask` argument to zero out attention on selected edges. Masked edges receive zero attention weight and their edge features (if any) are zeroed.
- **Numerically stable segment softmax.** `segment_softmax` now handles empty segments (e.g. from full masking) cleanly instead of producing NaN.
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

- **Optimizer.** New `ion.Optimizer` wraps an optax transform with Param-aware updates, replacing `apply_updates` as the third core abstraction alongside `Param` and `Module`. Frozen params are automatically partitioned so no optimizer memory is wasted on them.
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

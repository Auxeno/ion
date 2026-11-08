# Changelog

## 0.18.3

- **`gnn.segment_softmax` NaN gradient fix for padding.** Padded segment ids could turn every gradient into NaN. They now get exactly zero weight.
- **`nn.MultiHeadAttention` no longer leaks future tokens.** A row left with no valid keys by `mask` together with `causal` or `window`, as under left padding, averaged every value, future ones included. Such rows now output zeros.
- **Breaking: `gnn.unbatch_graphs` takes `num_graphs`.** Inferring the count dropped trailing empty graphs and failed on padded batches. Padded batches now round-trip.

## 0.18.2

- **Breaking: `nn.GroupNorm` takes `num_groups` and `num_spatial_dims` as keywords**, matching the other normalization layers. It also gains `use_bias`.
- **`gnn.SAGEConv` rejects an unknown `aggregate`** with a `ValueError` instead of a `KeyError`.
- **`nn.MaxPool` reduces in the input dtype.** It no longer casts to `float32`, which rounded integer inputs.
- **`nn.BatchNorm` NaN running variance fix.** A batch of one no longer makes the running variance NaN.
- **`gnn.HGTConv` NaN gradient fix.** Masked edges could produce NaN gradients. Their logits are now set to `-inf` before the softmax.
- **Subclassing a layer keeps its constructor.** `class MyLinear(nn.Linear): pass` no longer replaces the inherited `__init__`.

## 0.18.1

- **`Optimizer` repr handles an abstract step.** It no longer fails when `step` is a shape placeholder, as when costing a training step.
- **`ion.cost` analyses methods other than `__call__`.** Use `model.cost(x, method="critic")` or `ion.cost(model.critic, x)`.
- **`ion.cost` keeps its per-layer breakdown under JAX transforms.** `jax.grad`, `jax.jit`, `jax.checkpoint` and `jax.vmap` now report layer by layer instead of one total.
- **Buffer distributions in `__repr__`.** Buffers show the same histogram and moments as parameters.
- **Abbreviated dtypes in `__repr__`.** `float32` prints as `f32` and `bfloat16` as `bf16`.
- **`__repr__` formatting tweaks** for `Module` and `Cost`.
- **`Optimizer` `__repr__` shows its transforms and state**, with the hyperparameters of each stage.

## 0.18.0

- **`nn.DropPath`.** Stochastic depth, dropping a whole residual branch per sample.
- **Breaking: `Dropout.broadcast_dims` removed.** Use `DropPath` instead.
- **Native typing module.** Annotations now come from `ion.typing`, removing the `jaxtyping` dependency.
- **Improved Treescope colors.** Modules are colored by import order.
- **Parameter statistics in every model repr.** Printing a model shows each parameter's histogram, mean and std in any terminal. Set `ion.statistics = False` to hide them.
- **`model.cost`.** Compiles a call without running it, then reports FLOPs, memory and per-layer outputs. `ion.cost` does the same for any function that takes a model, such as a loss or a training step.
- **Flatter repr palette.** Each layer class has a fixed color, all at the same lightness.
- **Breaking: Treescope is optional and no longer activated on import.** `ion.enable_treescope` and `ion.disable_treescope` are gone. Install `treescope` and call `treescope.basic_interactive_setup()`.

## 0.17.1

- **Edge masking for edge-aware and relational layers.** Layers that take `x_edge`, plus `RGCNConv` and `HGTConv`, accept an `edge_mask` that leaves edges out of aggregation. Masked edges are still updated and returned.
- **`gnn.RGCNConv` NaN gradient fix.** Padded or masked edges could produce NaN gradients for `w_neigh`.

## 0.17.0

- **Neural network blocks.** `nn.Residual` adds a layer's output to its input. `nn.Bidirectional` runs sequence layers in both directions.
- **`nn.Ensemble`.** Builds independently initialized copies of a model and evaluates them with `vmap`.
- **Nested call routing.** `Sequential`, `Residual` and `Ensemble` pass `training` and `key` through nested blocks.
- **Breaking: consistent GNN naming.** Root transforms are `w_root`, `SAGEConv` takes `aggregate`, relational layers take `num_edge_types`, `GATConv.w` is `w_sender`, and fixed sizes use `node_capacity` and `edge_capacity`.
- **`gnn.remove_self_loops` returns retained indices**, for slicing edge features to match.
- **GNN fixes and options.** `GraphNorm` gains `use_bias`, GAT masks handle non-finite edge features, and GIN, GINE and global attention accept any callable.
- **Subgraph extraction.** `gnn.induced_subgraph` takes the graph over selected nodes, and `gnn.k_hop_subgraph` takes a node's k-hop neighbourhood. Both return the original indices for slicing features.
- **`gnn.GraphNorm`.** Normalizes node features within each graph.
- **Documentation fixes.** GNN layers that accept bipartite inputs now say so in their docstrings.

## 0.16.0

- **Breaking: GNN layer modules named by mechanism.** Modules are now `conv`, `attention`, `isomorphism`, `composite`, `relational`, `gated` and `pool`, and `ion.gnn.ops.readout` is `ion.gnn.ops.pool`. The flat `ion.gnn` API is unchanged, but imports such as `ion.gnn.layers.gcn` no longer work.
- **`gnn.min_pool`.** Per-graph minimum of node features, alongside `mean_pool`, `sum_pool` and `max_pool`.
- **`gnn.GraphNetwork`.** Combines user-supplied edge and node models into one Graph Network layer.
- **`gnn.NodeUpdate`.** Aggregates edge features at each receiver, then applies a user-supplied node model.
- **`gnn.EdgeUpdate` edge features are optional.** Without `x_edge`, edges are built from their two nodes alone.
- **`gnn.HGTConv`.** Heterogeneous graph transformer, with projections per node type and attention per edge type.
- **`gnn.RGCNConv`.** Relational graph convolution with a separate transform per edge type. `num_bases` shares weights across types.
- **Bipartite message passing.** Layers that support it accept `(x_src, x_dst)` node features and `in_dim` as a `(src_dim, dst_dim)` tuple.
- **`gnn.pad_graphs`.** Pads a batch to fixed node, edge and graph counts, so a compiled step traces once. Layers need no changes.
- **Faster graph batching.** `batch_graphs`, `pad_graphs` and `unbatch_graphs` now use NumPy internally. A streaming epoch over 41k graphs went from 187s to 1.4s.

## 0.15.1

- **`nn.Identity` restored.** Parameterless pass-through module.
- **`gnn.GatedGCNConv`.** Updates node and edge features together, using edges as gates on node messages.
- **`gnn.EdgeUpdate`.** Learns edge features from the sender, receiver and current edge features with a user-supplied module.
- **`gnn.GINEConv`.** `GINConv` with edge features added to each message (Hu et al., 2020).
- **`gnn.unbatch_graphs`.** Splits a batched graph back into individual graphs. Call it outside `jax.jit`.
- **Fixed LaTeX in demo notebooks.**

## 0.15.0

- **Breaking: `Identity` removed.** Use any plain callable instead.
- **Breaking: `LRU` and `LRUCell` removed.** They overlapped heavily with `S5`.
- **`gnn.GlobalAttentionPool`.** Pools each graph with a learned, attention-weighted sum of its nodes.
- **Treescope rendering moved to `_treescope.py`.** No user-facing change.
- **Dropout supports broadcast masks.** Pass `broadcast_dims` for structured dropout and stochastic depth.
- **Breaking: GNN source modules reorganized.** Layers live under `ion.gnn.layers` and operations under `ion.gnn.ops`. The flat `ion.gnn` API is unchanged, but imports such as `ion.gnn.gcn` no longer work.

## 0.14.1

- **`gnn.segment_var` and `gnn.segment_std`.** Per-segment variance and standard deviation, matching `jnp.var` and `jnp.std`. Empty segments return zero.

## 0.14.0

- **Breaking: `LoRALinear` removed.** Build LoRA from a frozen base plus trainable `a` and `b`: `base(x) + (x @ a @ b) * (alpha / rank)`.
- **Breaking: `dot_product_attention_with_rope` removed.** Compose `RoPE` with attention directly.
- **`gnn.line_graph`.** Builds the line graph, where edges become nodes. `non_backtracking` drops pairs that walk straight back along the reverse edge.
- **`gnn.to_undirected`.** Adds the reverse of every edge and removes duplicates.
- **`gnn.coalesce`.** Sorts edges and removes duplicates. It also returns the indices of kept edges, for filtering edge features.
- **`gnn.degree`.** Counts the edges at each node. Pass `senders` for out-degree or `receivers` for in-degree.
- **`gnn.remove_self_loops`.** Drops every `i -> i` edge. Call it outside `jax.jit`.
- **Treescope rendering reworked.** Modules group parameters, buffers and child modules under headings, with a parameter count and size.
- **`Module.disk_usage`.** Size of the model's arrays as a string, such as `'75 KB'`.

## 0.13.0

- **Breaking: optional constructor arguments are keyword-only.** `MultiHeadAttention(64, 8)` becomes `MultiHeadAttention(64, num_heads=8)`. Shape arguments, as in `Linear(3, 16)`, stay positional.
- **Breaking: `GATConv` and `GATv2Conv` rename `b` to `b_out`**, matching `MultiHeadAttention`. Old GAT checkpoints with a bias no longer load.
- **Breaking: component toggles take a `use_` prefix.** `bias` becomes `use_bias`, `root_weight` becomes `use_root_weight` and `beta` becomes `use_beta`.
- **Constructors assign fields in declaration order.** No behaviour change.
- **Breaking: `SelfAttention` and `CrossAttention` merged into `MultiHeadAttention`.** Use `attn(x)` for self-attention and `attn(x, x_kv)` for cross-attention, with `kv_dim` replacing `context_dim`. Checkpoints still load.
- **`MultiHeadAttention` accepts a custom `attention_fn`**, for example to use cuDNN or apply `RoPE`.
- **Breaking: `RoPE` takes the sequence `axis`, defaulting to `-3`.** This matches the `(batch, seq, heads, head_dim)` layout `MultiHeadAttention` uses. Pass `axis=-2` for the old behaviour.
- **Breaking: `sinusoidal` is now the `SinusoidalPositionalEmbedding` layer.** `x + nn.sinusoidal(s, d)` becomes `nn.SinusoidalPositionalEmbedding()(x)`.
- **Breaking: `alibi` removed.** Use `RoPE`, or pass an ALiBi bias to `jax.nn.dot_product_attention` yourself.
- **Benchmark ResNets use BatchNorm** instead of GroupNorm. Stored ResNet results need re-running.
- **Benchmark GPT can use cuDNN flash attention** through `use_flash`, matching PyTorch's fused kernels.
- **Benchmark first-step metric removed.** Compile time already covers it.
- **RoPE handles N-dimensional positions.** Pass `shape` for axial RoPE on images, and `num_prefix_tokens` for CLS or register tokens. Breaking: pass `theta` by keyword.

## 0.12.1

- **Optimizer step counters use `uint32`.** This raises the maximum update count from roughly 2.1 billion to 4.3 billion.
- **Numerically sensitive reductions compute in `float32`.** Normalization and pooling layers, plus segment sum, mean, and softmax, cast results back to the input dtype for improved reduced-precision stability.
- **Fully masked attention rows no longer leak values.** `SelfAttention` and `CrossAttention` now zero their attention contribution when no keys are valid.
- **BatchNorm stores unbiased running variance.** Training still normalizes with the biased batch variance, while the running estimate applies `n / (n - 1)`.
- **Checkpoint writes are atomic.** Interrupted saves no longer destroy an existing checkpoint at the destination path.

## 0.12.0

- **New `Param.value` property.** Reads the parameter as autodiff sees it, with `stop_gradient` when frozen. Use it instead of `jnp.asarray(param)`.
- **New `nn.Buffer` for stateful layers.** Holds mutable, non-trainable values in a model. Read with `.value` and write with `.set`.
- **New `BatchNorm` and `SpectralNorm` layers.** Both use buffers and are called as `norm(x, training=True)`.
- **Breaking: `Dropout` uses explicit training mode.** `deterministic` is gone. Pass `training=True` with a key, or `training=False`.
- **Checkpoints include buffers.** `ion.save` writes running statistics, and `ion.load` returns a model with its own buffers.
- **New `ion.clone` and `Module.clone`.** Copies a model with independent buffers. See [Sharp edges](docs/sharp-edges.md) for which operations share them.
- **New `ion.is_buffer` predicate.** Companion to `ion.is_param`.
- **Optimizers exclude buffers from optax state**, so a saved `(model, optimizer)` pair resumes normally.
- **Requires JAX 0.7.2 or newer**, up from 0.5.0. Buffers are built on `jax.new_ref`.

## 0.11.2

- **`AvgPool` gains `count_include_pad`.** Defaults to `True`, matching PyTorch and Flax, which changes padded pooling. Pass `count_include_pad=False` for the old behaviour.
- **`LayerNorm` gains a `bias` flag.** Pass `bias=False` to drop the learned shift. It still subtracts the mean, unlike `RMSNorm`.
- **New benchmark suite.** Compares Ion with Equinox, Flax NNX and PyTorch on MLP, ResNet and GPT workloads.

## 0.11.1

- **New `gnn.GraphConv` layer.** Graph convolution from Morris et al. (2019) with separate neighbour and root transforms.
- **New `gnn.TransformerConv` layer.** Graph attention from Shi et al. (2020), with optional edge features and gated residuals.
- **GAT projection weights are stored as 2D matrices.** Old `GATConv` and `GATv2Conv` checkpoints don't load.

## 0.11.0

- **New MkDocs documentation site.** Guides, API reference, examples and workflows under `docs/`.
- **New `gnn.SAGEConv` layer.** GraphSAGE (Hamilton et al., 2017) with `mean`, `max` or `sum` aggregation.
- **`padding` typed as `Literal["SAME", "VALID"]`**, so type checkers catch invalid modes.
- **Weight inits default to `glorot_uniform`.** This covers `Linear`, `Conv`, `ConvTranspose`, `GCNConv`, attention and the SSM layers. `MLP` uses `he_uniform`, since it applies ReLU. Pass `w_init` for the old behaviour.
- **`Embedding` and `LearnedPositionalEmbedding` init scales with dimension**, giving std `1/sqrt(dim)` as in Flax.
- **`Sequential.__call__` accepts and returns `Any`**, since chained callables may pass non-array values.

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

- **Style consistency pass.** Source, tests, and docs audited against house style. No behavioural changes.
- **Error messages compacted and unified.** Every `raise` is now at most three lines, shaped `name (value) must be ...`.
- **`MLP.__call__` annotation fixed.**
- **Line length enforced.** Ruff now checks `E501` with a 100 character limit.

## 0.8.0

- **`GINConv`.** Graph Isomorphism Network layer (Xu et al., 2019): sum-aggregates neighbour features and applies a caller-supplied MLP to `(1 + eps) * x + aggregated`. `train_eps=True` makes `eps` learnable.
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

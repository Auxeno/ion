# Changelog

## 0.16.0

- **Breaking: GNN layer modules grouped by family.** `ion.gnn.layers` now names its modules after the mechanism rather than the model, matching `ion.nn.layers`: `conv` (`GCNConv`, `GraphConv`, `SAGEConv`), `attention` (`GATConv`, `GATv2Conv`, `TransformerConv`), `isomorphism`, `composite` (`GraphNetwork`, `EdgeUpdate`, `NodeUpdate`), `relational` (`RGCNConv`, `HGTConv`), `gated`, and `pool`. `ion.gnn.ops.readout` became `ion.gnn.ops.pool`. The flat `ion.gnn` API is unchanged, but direct imports from the old modules such as `ion.gnn.layers.gcn` no longer work, and the documentation pages moved with them.
- **`gnn.min_pool`.** Elementwise minimum of node features within each graph, completing the fixed readouts alongside `mean_pool`, `sum_pool`, and `max_pool`. Empty graphs give zeros rather than the `+inf` fill `segment_min` leaves behind.
- **`gnn.GraphNetwork`.** Composes caller-supplied edge and node models around a configurable edge-to-node reduction. Updated edges serve as messages and are returned alongside the updated destination nodes; input edge features are optional.
- **`gnn.NodeUpdate`.** Aggregates caller-supplied edge features at their receivers, concatenates them with current destination nodes, and applies a caller-supplied node model.
- **`gnn.EdgeUpdate` edge features are optional.** Omitting `x_edge` builds edge representations from the incident nodes alone, so `GraphNetwork` decomposes into `EdgeUpdate` and `NodeUpdate` for graphs that carry no input edge features as well as for those that do.
- **`gnn.HGTConv`.** Heterogeneous graph transformer. Key, query, value, and output projections follow a per-node `node_type` array while attention matrices, message matrices, and a learned prior follow a per-edge `edge_type` array. Attention is normalized over all of a node's incoming edges at once, so relations compete rather than being combined after separate softmaxes.
- **`gnn.RGCNConv`.** Gives each edge type its own neighbour transform, selected by a per-edge `edge_type` index array so one edge list carries every relation. `num_bases` builds each relation's transform from shared matrices, holding the parameter count flat as relations grow.
- **Bipartite message passing.** Every message-passing layer except `GCNConv` accepts `(x_src, x_dst)` node features and returns one row per destination node. Pass `in_dim` as a `(src_dim, dst_dim)` tuple to give each partition its own input projection.

## 0.15.1

- **`nn.Identity` restored.** Parameterless pass-through module.
- **`gnn.GatedGCNConv`.** Jointly updates node and edge features, using the new edge representations as normalized feature-wise gates on incoming node messages. Activation, normalization, and residual connections remain explicit compositions outside the layer.
- **`gnn.EdgeUpdate`.** Learns an edge representation from the concatenated sender, receiver, and current edge features with a caller-supplied module, following the Graph Network edge update.
- **`gnn.GINEConv`.** `GINConv` with edge features added to each message before the ReLU, following Hu et al., 2020. Edge features are added to sender features rather than projected, so `x_edge` shares the node dimension and the layer keeps `GINConv`'s property of creating no weights of its own.
- **`gnn.unbatch_graphs`.** Splits a batched graph back into per-graph node features and edge indices, undoing `batch_graphs`. It returns Python lists of variable-length arrays, so it is a data preparation step rather than something to call inside `jax.jit`.
- **Fix LaTeX in demo notebooks**. LaTeX in demo notebooks was formatted incorrectly.

## 0.15.0

- **Breaking: `Identity` removed.** Ion accepts ordinary callables wherever a pass-through operation might be used, so a dedicated layer added API surface without adding capability.
- **Breaking: `LRU` and `LRUCell` removed.** Their shared-state MIMO recurrence substantially overlapped `S5`; keeping `S4D` and `S5` leaves a clearer SISO/MIMO pair.
- **`gnn.GlobalAttentionPool`.** Learns a softmax-normalized node score and takes an attention-weighted sum within each graph. The caller supplies the score and optional value modules, which decide how much each node contributes and what features it contributes.
- **Refactor `__treescope_repr__` code to `_treescope.py`**. Slim down `module.py`, `param.py`, `optimizer.py` and `buffer.py` by porting `__treescope_repr__` logic to a separate file.
- **Dropout supports broadcast masks.** Pass `broadcast_dims` to express structured dropout and stochastic depth; its source and docs now live under `stochastic`.
- **Breaking: GNN source modules reorganized.** Layer implementations now live under `ion.gnn.layers`, matching `ion.nn.layers` and the documentation structure. Operations are grouped under `ion.gnn.ops.segment`, `readout`, `graph`, and `batching`. The flat `ion.gnn` API is unchanged, but direct imports from the old layer modules such as `ion.gnn.gcn` no longer work.

## 0.14.1

- **`gnn.segment_var` and `gnn.segment_std`.** Population variance and standard deviation within each segment, dividing by the segment size to match `jnp.var` and `jnp.std` at their default `ddof=0`. Both take the mean squared deviation from the segment mean rather than `E[x^2] - E[x]^2`, so they stay accurate when values are large relative to their spread, and empty or single-element segments return zero rather than `NaN`.

## 0.14.0

- **Breaking: `LoRALinear` removed.** Low-rank adaptation is a fine-tuning protocol rather than a layer, and it wrapped `Linear` alone, so it never reached the attention projections real LoRA targets. Build it from `Param` and `freeze`: a frozen base plus trainable `a` and `b`, returning `base(x) + (x @ a @ b) * (alpha / rank)`.
- **Breaking: `dot_product_attention_with_rope` removed.** The helper saved one composition and did not earn a slot in the public API. 
- **`gnn.line_graph`.** Rebuilds a graph with its edges as nodes, joined where one edge ends and another begins. Line-graph nodes are the rows of the input edge arrays, so `x_edge` becomes the node features unchanged and any existing convolution updates edge representations. `non_backtracking` drops the pairs that walk straight back down the reverse edge, and the returned pivot node is where angle features attach.
- **`gnn.to_undirected`.** Appends the reverse of every edge and coalesces, so an undirected graph is stored with both directions present. Coalescing makes it idempotent, and the returned indices address the original edges concatenated with the reversed ones, so direction-dependent edge features can be negated rather than copied.
- **`gnn.coalesce`.** Sorts edges by `(sender, receiver)` and drops duplicates, putting an edge list into the canonical form that sparse COO layouts use. It also returns the index of the row kept for each surviving edge, so edge features can be filtered to match.
- **`gnn.degree`.** Counts how many edges reference each node. It counts a single index array, so `degree(senders, n)` gives out-degree and `degree(receivers, n)` gives in-degree.
- **`gnn.remove_self_loops`.** Drops every `i -> i` edge, pairing with `add_self_loops` for datasets that ship with self-loops already present. The number of edges removed depends on the data, so it is a data preparation step rather than something to call inside `jax.jit`.
- **Treescope rendering reworked.** Modules put configuration on one line, then group parameters, buffers, and child modules under headings, and annotate themselves with a parameter count and size. Arrays show dtype and shape until a `Param` is expanded.
- **`Module.disk_usage`.** Size of the arrays a checkpoint would hold, as a string such as `'75 KB'`.

## 0.13.0

- **Breaking: optional constructor arguments are keyword-only.** Every layer now places `*` immediately after its last required argument, so the arguments that define a layer's shape stay positional and everything that configures its behaviour is passed by name. `Linear(3, 16)` and `Conv(3, 64, (3, 3))` are unchanged, while `MultiHeadAttention(64, 8)` becomes `MultiHeadAttention(64, num_heads=8)` and `Conv(3, 64, (3, 3), 2, 1)` becomes `Conv(3, 64, (3, 3), stride=2, padding=1)`.
- **Breaking: `GATConv` and `GATv2Conv` rename `b` to `b_out`.** The output bias now matches `MultiHeadAttention` and `TransformerConv`, and moves after the edge projections so it is declared last among the parameters. Checkpoints holding a GAT bias no longer load, since tensor names follow attribute paths.
- **Breaking: booleans that toggle a component take a `use_` prefix.** `bias` becomes `use_bias` (19 layers), `root_weight` becomes `use_root_weight`, and `beta` becomes `use_beta`, matching `flax.nnx`, `flax.linen`, and `equinox`. Booleans that describe behaviour rather than switch a component on keep their names: `causal`, `normalize`, `count_include_pad`, and `train_eps` are unchanged.
- **Constructors assign fields in declaration order.** Every layer now builds its pytree children (sub-modules, params, buffers) before its static configuration, matching the order the fields are declared in the class body. Argument defaulting and derived values come first, then validation, then the assignments. Behaviour is unchanged.
- **Breaking: `SelfAttention` and `CrossAttention` merged into `MultiHeadAttention`.** One layer covers both. `attn(x)` draws keys and values from `x`; `attn(x, x_kv)` draws them from a context sequence. `SelfAttention(dim, num_heads)` becomes `MultiHeadAttention(dim, num_heads=num_heads)`, and `CrossAttention(dim, num_heads, context_dim=c)` becomes `MultiHeadAttention(dim, num_heads=num_heads, kv_dim=c)`. Cross-attention gains grouped-query and multi-query attention, sliding windows, causal masking, and `attention_fn`, none of which it previously supported. Parameter names are unchanged, so existing checkpoints still load.
- **`MultiHeadAttention` accepts a custom `attention_fn`.** Use a partial to select JAX's cuDNN backend, or wrap `jax.nn.dot_product_attention` to rotate query and key with `RoPE` first.
- **Breaking: `RoPE` takes the sequence `axis` as constructor config, defaulting to `-3`.** The default matches the `(batch, sequence, heads, head_dim)` queries and keys that `MultiHeadAttention` builds and `jax.nn.dot_product_attention` consumes, so composing the two through `attention_fn` needs no override. Under the old default of `-2`, that layout silently rotated across heads rather than positions. Pass `RoPE(axis=-2)` for a head-first layout or a single unbatched head shaped `(sequence, head_dim)`.
- **Breaking: `sinusoidal` is now the `SinusoidalPositionalEmbedding` layer.** The function returned a table for the caller to add; the layer adds it, matching `LearnedPositionalEmbedding` so the two are interchangeable at a call site. Sequence length and feature dimension come from the input, so the layer takes only `theta`, and `dtype` is gone in favour of following the input. `x + nn.sinusoidal(s, d)` becomes `nn.SinusoidalPositionalEmbedding()(x)`.
- **Breaking: `alibi` removed.** The ALiBi position bias was the one positional feature that could not be composed through `attention_fn`, and materializing its `(heads, seq, seq)` bias defeats fused attention kernels. Use `RoPE`, or build the bias inline and pass it to `jax.nn.dot_product_attention` through its `bias` argument.
- **Benchmark ResNets use BatchNorm.** Ion's `nn.BatchNorm` and its buffers replace the `GroupNorm` stand-in that existed only because the suite could not carry mutable running statistics, and Equinox, Flax NNX, and PyTorch follow. Parameter counts are unchanged, so the cross-framework parity checks are unaffected, but every stored ResNet result needs re-running.
- **Benchmark GPT can select cuDNN flash attention.** `use_flash` threads `implementation="cudnn"` into Ion's `attention_fn` and NNX's, and is set for the small and medium GPT so each framework uses the fastest kernel available for that shape, as PyTorch's `scaled_dot_product_attention` already does automatically. JAX's default of `None` resolves to `xla`, so the JAX frameworks were previously measured against PyTorch's fused kernels. The tiny GPT stays on XLA, and Ion and NNX fall back to it when no GPU is present.
- **Benchmark first-step metric is removed.** Compile time already reports first-step latency minus the median warmed step, so the raw first-step number added a column and a figure without adding information. Compile-time plots now use seconds rather than milliseconds.
- **RoPE handles N-dimensional positions.** Pass `shape` to lay positions on a lattice instead of a flat sequence, splitting the head dimension evenly across its axes, as axial RoPE does for images. `num_prefix_tokens` holds leading CLS or register tokens at position 0, where the rotation is the identity. `head_dim` must be divisible by `2 * len(shape)`. Breaking: `theta` moves to the end of the constructor, so pass it by keyword.

## 0.12.1

- **Optimizer step counters use `uint32`.** This raises the maximum update count from roughly 2.1 billion to 4.3 billion.
- **Numerically sensitive reductions compute in `float32`.** Normalization and pooling layers, plus segment sum, mean, and softmax, cast results back to the input dtype for improved reduced-precision stability.
- **Fully masked attention rows no longer leak values.** `SelfAttention` and `CrossAttention` now zero their attention contribution when no keys are valid.
- **BatchNorm stores unbiased running variance.** Training still normalizes with the biased batch variance, while the running estimate applies `n / (n - 1)`.
- **Checkpoint writes are atomic.** Interrupted saves no longer destroy an existing checkpoint at the destination path.

## 0.12.0

- **New `Param.value` property.** Reads the parameter as autodiff sees it, applying `stop_gradient` when frozen, and matches `Buffer.value`. Use it instead of `jnp.asarray(param)` where a plain array is needed. The private `_value` field is unchanged and still holds the raw stored array.
- **New `nn.Buffer` for stateful layers.** Buffers hold mutable, non-trainable values directly in a model. Read them with `.value` and update them with `.set`, which applies `stop_gradient`. They contribute no pytree leaves, so `jax.grad`, `ion.Optimizer` and `Module.astype` leave them alone.
- **New `BatchNorm` and `SpectralNorm` layers.** They use buffers for running statistics and power-iteration vectors, and are called normally with `y = norm(x, training=True)`, including in `Sequential`. `SpectralNorm` takes a constructor `key` to initialize its vectors, requires real floating parameters, and initializes its power-iteration vectors with JAX's default floating dtype. Both layers preserve their input dtype at the output boundary.
- **Breaking: `Dropout` uses explicit training mode.** The `deterministic` constructor and call arguments are removed. Pass `training=True` with a key to sample a mask, or `training=False` for the evaluation identity.
- **Checkpoints include buffers.** `ion.save(path, model)` writes running statistics as ordinary named tensors. `ion.load` returns a model with its own buffers, leaving the reference model's state untouched. The format version is unchanged.
- **New `ion.clone` and `Module.clone`.** Returns a copy whose buffers are independent of the original, as do `freeze`, `unfreeze` and `load`. `astype` is the exception and shares them so mixed-precision copies update the model's state; `Optimizer.update` and plain `jax.tree.map` copies share them too. See [Sharp edges](docs/sharp-edges.md).
- **New `ion.is_buffer` predicate.** Companion to `ion.is_param`, for tree code that needs to find buffers.
- **Optimizers exclude buffers from optax state.** This keeps mutable references out of optimizer checkpoints, so a saved `(model, optimizer)` pair resumes normally with the loaded model's independent buffers.
- **Requires JAX 0.7.2 or newer**, up from 0.5.0. Buffers are built on `jax.new_ref`.

## 0.11.2

- **`AvgPool` gains a `count_include_pad` flag.** Controls whether padded positions count towards the window size. Defaults to `True`, matching `torch.nn.AvgPool2d` and `flax.linen.avg_pool`. This changes existing behaviour: `AvgPool` previously always divided by the real element count. Pass `count_include_pad=False` to restore it. Only affects padded pooling; results without padding are unchanged.
- **`LayerNorm` gains a `bias` flag.** Pass `bias=False` to drop the learnable shift, as used in LLaMA-style transformers. Note that a bias-less `LayerNorm` still subtracts the mean, so it is not equivalent to `RMSNorm`.
- **New benchmark suite.** Compares Ion with Equinox, Flax NNX, and PyTorch eager and compiled across MLP, ResNet, and GPT workloads, measuring forward, backward, full-step, compilation, first-step, throughput, and peak-memory performance. Includes reproducible JSON results and interactive Plotly reports.

## 0.11.1

- **New `gnn.GraphConv` layer.** Graph convolution from Morris et al. (2019) with independently learned neighbour and root transforms, unnormalized sum aggregation, and optional scalar edge weights.
- **New `gnn.TransformerConv` layer.** Sparse scaled dot-product graph attention from Shi et al. (2020), with multi-head Q/K/V projections, optional edge features, a root transform, edge masking, and optional gated residuals.
- **GAT projection weights are now stored as 2D matrices.** `GATConv` and `GATv2Conv` add head axes to activations in the forward pass, matching the attention layers in `ion.nn`. Existing GAT checkpoints are incompatible.

## 0.11.0

- **New MkDocs documentation site.** Adds guides, API reference, examples, workflows, and sharp-edge documentation under `docs/`.
- **New `gnn.SAGEConv` layer.** GraphSAGE (Hamilton et al., 2017) with `mean`, `max`, or `sum` neighbour aggregation, an optional `root_weight` term for the central node, and optional L2 `normalize`.
- **`padding` typed as `Literal["SAME", "VALID"]`.** `Conv`, `ConvTranspose`, `MaxPool`, and `AvgPool` now type the string form of `padding` as a literal instead of `str`, so type checkers catch invalid modes at the call site.
- **Layer weight inits now default to uniform variance scaling, matched to the following activation.** `Linear`, `Conv`, `ConvTranspose`, `GCNConv`, `SelfAttention`/`CrossAttention`, and the SSM layers (`S4D`, `S5`, `LRU`) default to `glorot_uniform` (gain 1) instead of `he_normal`/`truncated_normal(0.02)`; `MLP` uses `he_uniform`, since it applies ReLU between its layers. He assumes a ReLU that a bare projection or attention does not have, so gain-1 Glorot is the safer default and matches PyTorch and Flax. Pass an explicit `w_init` to restore the old behaviour.
- **`Embedding` and `LearnedPositionalEmbedding` init scales with dimension.** The default is now `variance_scaling(1.0, "fan_in", "uniform", out_axis=0)`, giving std `1/sqrt(dim)` so each row starts near unit norm regardless of `dim` and independent of vocabulary or sequence length. Replaces the dimension-blind `truncated_normal(0.02)` constant, matching Flax's `Embed`.
- **`Sequential.__call__` accepts and returns `Any`.** The annotation now reflects that chained callables may pass non-array values.

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

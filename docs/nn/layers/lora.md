# LoRA

Low-rank adaptation of a linear layer. `LoRALinear` wraps a frozen base weight with a trainable low-rank update `B @ A`, so only the small `A` and `B` factors receive gradients. Pairs with the optimizer's [auto-partitioning](../../core/optimizer.md), which allocates no state for the frozen base.

::: ion.nn.LoRALinear

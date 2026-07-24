# Examples

End-to-end training scripts and notebooks live in the [`examples/`](https://github.com/auxeno/ion/tree/main/examples) directory. Each is self-contained and runnable.

## Getting started

| Example | What it covers |
|---------|----------------|
| [Quickstart](../quickstart.md) | A small MLP trained with native JAX gradients and an Ion optimizer. |
| [Ion Tour](ion-tour.ipynb) | A guided walkthrough of `Module`, `Param`, `Optimizer`, and the training loop. |

## Vision

| Example | What it covers |
|---------|----------------|
| [CNN on MNIST](cnn-mnist.md) | Convolutional classifier with `Conv` and pooling. |
| [RNN on MNIST](rnn-mnist.md) | Sequence classification reading a flattened pixel stream. |
| [VAE on MNIST](vae-mnist.ipynb) | Variational autoencoder with a reparameterized latent. |

## Sequence modeling

| Example | What it covers |
|---------|----------------|
| [GPT on TinyStories](gpt-tinystories.ipynb) | Transformer language model with attention, RoPE, and mixed-precision training. |
| [SSM on Pathfinder](ssm-pathfinder.ipynb) | Deep state space model on a long-range dependency benchmark. |

## Graphs

| Example | What it covers |
|---------|----------------|
| [GNN on Cora](gnn-cora.md) | Node classification with `GCNConv`, `GATConv`, and self-loops. |
| [GNN on BBBP](gnn-bbbp.ipynb) | Molecular property prediction with graph pooling and batching. |

## Reinforcement learning

| Example | What it covers |
|---------|----------------|
| [DQN on Atari](dqn-atari.ipynb) | Deep Q-network with a convolutional encoder. |
| [PPO on Gymnax](ppo-gymnax.md) | Clipped actor-critic across vectorized environments. |
| [PQN on Gymnax](pqn-gymnax.md) | Parallelized Q-network with layer norm, no replay buffer. |

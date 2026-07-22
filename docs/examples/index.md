# Examples

End-to-end training scripts and notebooks live in the [`examples/`](https://github.com/auxeno/ion/tree/main/examples) directory. Each is self-contained and runnable.

## Getting started

| Example | What it covers |
|---------|----------------|
| [Ion Tour](https://github.com/auxeno/ion/blob/main/examples/ion_tour.ipynb) | A guided walkthrough of `Module`, `Param`, `Optimizer`, and the training loop. |

## Vision

| Example | What it covers |
|---------|----------------|
| [CNN on MNIST](https://github.com/auxeno/ion/blob/main/examples/cnn_mnist.py) | Convolutional classifier with `Conv` and pooling. |
| [RNN on MNIST](https://github.com/auxeno/ion/blob/main/examples/rnn_mnist.py) | Sequence classification reading images row by row. |
| [VAE on MNIST](https://github.com/auxeno/ion/blob/main/examples/vae_mnist.ipynb) | Variational autoencoder with a reparameterized latent. |

## Sequence modeling

| Example | What it covers |
|---------|----------------|
| [GPT on TinyStories](https://github.com/auxeno/ion/blob/main/examples/gpt_tinystories.ipynb) | Transformer language model with attention, RoPE, and mixed-precision training. |
| [SSM on Pathfinder](https://github.com/auxeno/ion/blob/main/examples/ssm_pathfinder.ipynb) | Deep state space model on a long-range dependency benchmark. |

## Graphs

| Example | What it covers |
|---------|----------------|
| [GNN on Cora](https://github.com/auxeno/ion/blob/main/examples/gnn_cora.py) | Node classification with `GCNConv` and self-loops. |
| [GNN on BBBP](https://github.com/auxeno/ion/blob/main/examples/gnn_bbbp.ipynb) | Molecular property prediction with graph pooling and batching. |

## Reinforcement learning

| Example | What it covers |
|---------|----------------|
| [DQN on Atari](https://github.com/auxeno/ion/blob/main/examples/dqn_atari.ipynb) | Deep Q-network with a convolutional encoder. |
| [PPO on Gymnax](https://github.com/auxeno/ion/blob/main/examples/ppo_gymnax.py) | Actor-critic with per-field optimizers. |
| [PQN on Gymnax](https://github.com/auxeno/ion/blob/main/examples/pqn_gymnax.py) | Parallelized Q-network training. |

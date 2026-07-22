# PQN on Gymnax

Parallelized Q-Network on a vectorized [Gymnax](https://github.com/RobertTLange/gymnax) environment (CartPole by default). PQN is a simplified value-based method: no replay buffer and no target network, just on-policy rollouts from 16 parallel environments with Q-lambda returns and layer normalization for stability.

Points of interest:

- A plain `QNetwork` MLP with `LayerNorm` replaces DQN's target network and replay buffer, so the whole update is a single differentiable pass.
- Rollouts use linearly decayed epsilon-greedy exploration, annealed over the first half of training.
- Bootstrapped Q-lambda returns are computed on-policy each rollout, then fit over a few epochs of minibatches.

## Source

[examples/pqn_gymnax.py](https://github.com/auxeno/ion/blob/main/examples/pqn_gymnax.py) on GitHub.

```python title="examples/pqn_gymnax.py" linenums="1"
--8<-- "examples/pqn_gymnax.py"
```

## Output

```bash
uv run --group examples examples/pqn_gymnax.py
```

```
  Step    98,304 | Mean reward: 33.3
  Step   198,656 | Mean reward: 87.0
  Step   299,008 | Mean reward: 211.9
  Step   399,360 | Mean reward: 399.6
  Step   499,712 | Mean reward: 477.1
  Step   598,016 | Mean reward: 500.0
  Step   698,368 | Mean reward: 500.0
  Step   798,720 | Mean reward: 414.1
  Step   899,072 | Mean reward: 490.9
  Step   999,424 | Mean reward: 495.5
```

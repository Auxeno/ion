# PPO on Gymnax

Proximal Policy Optimization on a vectorized [Gymnax](https://github.com/RobertTLange/gymnax) environment (CartPole by default). A shared-torso `ActorCritic` network runs across 16 parallel environments; rollouts are collected, advantages estimated with GAE, and the clipped PPO objective optimized over several epochs of minibatches.

Points of interest:

- The whole rollout and learning inner loop is `jax.jit`ed and runs over 16 environments at once via `jax.vmap`.
- One `ActorCritic` module carries both the policy and value heads; a single `Optimizer` updates the whole pytree.
- Advantages come from GAE, and the surrogate loss combines the clipped policy objective, a value loss, and an entropy bonus.

## Source

[examples/ppo_gymnax.py](https://github.com/auxeno/ion/blob/main/examples/ppo_gymnax.py) on GitHub.

```python title="examples/ppo_gymnax.py" linenums="1"
--8<-- "examples/ppo_gymnax.py"
```

## Output

```bash
uv run --group examples examples/ppo_gymnax.py
```

```
  Step    99,328 | Mean reward: 268.8
  Step   199,680 | Mean reward: 463.1
  Step   299,008 | Mean reward: 475.1
  Step   399,360 | Mean reward: 494.7
  Step   499,712 | Mean reward: 496.5
  Step   599,040 | Mean reward: 494.9
  Step   699,392 | Mean reward: 493.0
  Step   798,720 | Mean reward: 496.7
  Step   899,072 | Mean reward: 500.0
  Step   999,424 | Mean reward: 484.1
```

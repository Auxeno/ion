"""PQN on CartPole-v1 with gymnax vectorized environments."""

from collections import deque
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tqdm import tqdm

import ion
from ion import nn


class QNetwork(nn.Module):
    """Q-network with LayerNorm for stable TD learning without a target network."""

    hidden_layers: tuple[tuple[nn.Linear, nn.LayerNorm], ...]
    output_layer: nn.Linear

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jax.random.split(key, num_hidden_layers + 1)

        layers = []
        in_dim = obs_dim
        for i in range(num_hidden_layers):
            layers.append(
                (
                    nn.Linear(in_dim, hidden_dim, key=keys[i]),
                    nn.LayerNorm(hidden_dim),
                )
            )
            in_dim = hidden_dim

        self.hidden_layers = tuple(layers)
        self.output_layer = nn.Linear(in_dim, action_dim, key=keys[-1])

    def __call__(self, observations: Float[Array, "... d"]) -> Float[Array, "... a"]:
        x = observations
        for linear, norm in self.hidden_layers:
            x = jax.nn.relu(norm(linear(x)))
        return self.output_layer(x)


def eps_greedy_action(
    q_values: Float[Array, "n a"],
    epsilon: Float[Array, ""],
    *,
    key: PRNGKeyArray,
) -> Int[Array, " n"]:
    """Select actions via epsilon-greedy exploration."""
    key_eps, key_rand = jax.random.split(key)
    greedy = q_values.argmax(axis=-1)
    random = jax.random.randint(key_rand, greedy.shape, 0, q_values.shape[-1])
    explore = jax.random.uniform(key_eps, greedy.shape) < epsilon
    return jnp.where(explore, random, greedy)


class Transition(NamedTuple):
    observations: Float[Array, "... d"]
    next_observations: Float[Array, "... d"]
    rewards: Float[Array, "..."]
    terminations: Bool[Array, "..."]
    truncations: Bool[Array, "..."]
    actions: Int[Array, "..."]
    q_values: Float[Array, "... a"]


RolloutCarry = tuple[PRNGKeyArray, gymnax.EnvState, Float[Array, "n d"]]


@jax.jit
def rollout(
    network: QNetwork,
    carry: RolloutCarry,
    epsilon: Float[Array, ""],
) -> tuple[RolloutCarry, Transition]:
    """Collect transitions from vectorized environments via lax.scan."""

    def step_fn(carry: RolloutCarry, _: None) -> tuple[RolloutCarry, Transition]:
        rng, env_states, observations = carry
        rng, key_action, key_step = jax.random.split(rng, 3)

        q_values = network(observations)
        actions = eps_greedy_action(q_values, epsilon, key=key_action)

        next_observations, next_states, rewards, terminations, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(
            jax.random.split(key_step, NUM_ENVS),
            env_states,
            actions,
            env_params,
        )
        truncations = jnp.zeros_like(terminations)

        transition = Transition(
            observations,
            next_observations,
            rewards,
            terminations,
            truncations,
            actions,
            q_values,
        )
        return (rng, next_states, next_observations), transition

    new_carry, transitions = jax.lax.scan(f=step_fn, init=carry, xs=None, length=ROLLOUT_STEPS)
    return new_carry, transitions


def calculate_td_lambda(
    network: QNetwork,
    rewards: Float[Array, "t n"],
    q_values: Float[Array, "t n a"],
    terminations: Bool[Array, "t n"],
    truncations: Bool[Array, "t n"],
    last_observations: Float[Array, "n d"],
    gamma: float,
    td_lambda: float,
) -> Float[Array, "t n"]:
    """Compute TD(lambda) return targets via reversed scan over timesteps."""

    # Max Q-values at each step: max_q[t] = max Q(s_t)
    max_q = q_values.max(axis=-1)

    # Bootstrap from last next observations: max Q(s_T)
    last_q = network(last_observations).max(axis=-1)
    non_term_last = 1.0 - terminations[-1]
    lambda_return_init = rewards[-1] + gamma * non_term_last * last_q

    def lambda_step(
        lambda_return: Float[Array, " n"],
        xs: tuple,
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        reward, next_q, termination, truncation = xs
        non_termination = 1.0 - termination
        non_truncation = 1.0 - truncation
        target_bootstrap = reward + gamma * non_termination * next_q
        lambda_return = target_bootstrap + gamma * td_lambda * non_termination * non_truncation * (
            lambda_return - next_q
        )
        return lambda_return, lambda_return

    # next_q for step t is max_q[t+1] = max Q(s_{t+1})
    _, targets = jax.lax.scan(
        f=lambda_step,
        init=lambda_return_init,
        xs=(rewards[:-1], max_q[1:], terminations[:-1], truncations[:-1]),
        reverse=True,
    )
    return jnp.concatenate([targets, lambda_return_init[None]])


def td_loss(
    network: QNetwork,
    observations: Float[Array, "b d"],
    actions: Int[Array, " b"],
    targets: Float[Array, " b"],
) -> Float[Array, ""]:
    """Mean squared TD-error loss."""
    q_values = network(observations)
    chosen_q = jnp.take_along_axis(q_values, actions[..., None], axis=-1).squeeze(-1)
    return 0.5 * ((chosen_q - targets) ** 2).mean()


@jax.jit
def learn(
    network: QNetwork,
    optimizer: ion.Optimizer,
    batch: Transition,
    *,
    key: PRNGKeyArray,
) -> tuple[QNetwork, ion.Optimizer]:
    """Compute TD(lambda) targets then scan over minibatch updates."""

    # Compute lambda return targets
    targets = calculate_td_lambda(
        network,
        batch.rewards,
        batch.q_values,
        batch.terminations,
        batch.truncations,
        batch.next_observations[-1],
        GAMMA,
        TD_LAMBDA,
    )

    # Flatten (rollout_steps, num_envs, ...) -> (batch_size, ...)
    batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    targets = targets.flatten()

    # Shuffled minibatch indices
    indices = jnp.tile(jnp.arange(BATCH_SIZE, dtype=jnp.int32), (NUM_EPOCHS, 1))
    mb_indices = jax.vmap(jax.random.permutation)(jax.random.split(key, NUM_EPOCHS), indices)
    mb_indices = mb_indices.reshape(NUM_EPOCHS * NUM_MINIBATCHES, MINIBATCH_SIZE)

    def minibatch_update(carry, indices):
        network, optimizer = carry
        loss, grads = jax.value_and_grad(td_loss)(
            network,
            batch.observations[indices],
            batch.actions[indices],
            targets[indices],
        )
        network, optimizer = optimizer.update(network, grads)
        return (network, optimizer), loss

    (network, optimizer), _ = jax.lax.scan(minibatch_update, (network, optimizer), mb_indices)
    return network, optimizer


def train_pqn(
    network: QNetwork,
    *,
    seed: int = 42,
) -> QNetwork:
    """Train a PQN agent on a gymnax environment."""

    rng = jax.random.key(seed)
    rng, key_reset, rng_rollout = jax.random.split(rng, 3)

    # Initialize optimizer
    optimizer = ion.Optimizer(
        optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate=LR, eps=1e-8)),
        network,
    )

    # Reset vectorized environments
    observations, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key_reset, NUM_ENVS), env_params
    )
    carry = (rng_rollout, env_states, observations)

    # Epsilon schedule: linear decay over EPS_DECAY fraction of training
    total_rollouts = TOTAL_STEPS // BATCH_SIZE
    eps_decay_rollouts = int(total_rollouts * EPS_DECAY)

    # Episode tracking
    current_returns = np.zeros(NUM_ENVS)
    recent_returns: deque[float] = deque(maxlen=100)
    checkpoints = {total_rollouts * p // 10 for p in range(1, 11)}

    bar = tqdm(range(total_rollouts), desc=f"PQN {GYMNAX_ENV_NAME}")
    for i in bar:
        rng, key_learn = jax.random.split(rng)

        # Linear epsilon decay
        frac = jnp.minimum(1.0, i / max(eps_decay_rollouts, 1))
        epsilon = EPS_START + (EPS_FINISH - EPS_START) * frac

        carry, transitions = rollout(network, carry, epsilon)
        network, optimizer = learn(network, optimizer, transitions, key=key_learn)

        # Track episode returns
        rewards_np = np.asarray(transitions.rewards)
        dones_np = np.asarray(transitions.terminations | transitions.truncations)
        for step_r, step_d in zip(rewards_np, dones_np):
            current_returns += step_r
            for ret in current_returns[step_d]:
                recent_returns.append(float(ret))
            current_returns[step_d] = 0.0

        if recent_returns:
            mean_reward = np.mean(recent_returns)
            bar.set_postfix(reward=f"{mean_reward:.1f}", eps=f"{epsilon:.2f}")
            if i + 1 in checkpoints:
                tqdm.write(f"  Step {(i + 1) * BATCH_SIZE:>9,} | Mean reward: {mean_reward:.1f}")

    return network


if __name__ == "__main__":
    GYMNAX_ENV_NAME = "CartPole-v1"
    TOTAL_STEPS = 1_000_000
    ROLLOUT_STEPS = 128
    NUM_ENVS = 16
    LR = 1e-3
    GAMMA = 0.99
    TD_LAMBDA = 0.65
    NUM_EPOCHS = 4
    NUM_MINIBATCHES = 4
    EPS_START = 1.0
    EPS_FINISH = 0.05
    EPS_DECAY = 0.5
    SEED = 42

    BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
    MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES

    # Create gymnax environment
    env, env_params = gymnax.make(GYMNAX_ENV_NAME)
    OBS_DIM = env.observation_space(env_params).shape[0]  # type: ignore[reportArgumentType]
    ACTION_DIM = int(env.action_space(env_params).n)  # type: ignore[reportArgumentType]

    rng = jax.random.key(SEED)
    rng, key_network = jax.random.split(rng)
    network = QNetwork(OBS_DIM, ACTION_DIM, 128, 2, key=key_network)

    trained = train_pqn(network, seed=SEED)

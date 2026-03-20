"""DQN on CartPole-v1 with gymnasium vectorized environments.

Demonstrates replay buffers, target networks, and the `donate_argnums` trick
for memory-efficient buffer updates in JAX.
"""

from functools import partial
from typing import NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

import ion
from ion import nn


class QNetwork(nn.Module):
    """Q-network for discrete action spaces."""

    q: nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, *, key: PRNGKeyArray) -> None:
        self.q = nn.MLP(obs_dim, action_dim, 64, 2, activation=jax.nn.relu, key=key)

    def __call__(self, observations: Float[Array, "... d"]) -> Float[Array, "... a"]:
        return self.q(observations)


class Transition(NamedTuple):
    """Environment transition."""

    observations: Float[Array, "... d"]
    next_observations: Float[Array, "... d"]
    actions: Int[Array, "..."]
    rewards: Float[Array, "..."]
    terminations: Float[Array, "..."]
    truncations: Float[Array, "..."]


class BufferState(NamedTuple):
    """Circular replay buffer for storing transitions."""

    transitions: Transition
    idx: Int[Array, ""]
    size: Int[Array, ""]


def init_buffer(obs_dim: int) -> BufferState:
    """Allocate an empty replay buffer."""
    return BufferState(
        transitions=Transition(
            observations=jnp.zeros((BUFFER_SIZE, obs_dim)),
            next_observations=jnp.zeros((BUFFER_SIZE, obs_dim)),
            actions=jnp.zeros(BUFFER_SIZE, dtype=jnp.int32),
            rewards=jnp.zeros(BUFFER_SIZE),
            terminations=jnp.zeros(BUFFER_SIZE),
            truncations=jnp.zeros(BUFFER_SIZE),
        ),
        idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


@partial(jax.jit, donate_argnums=(0,))
def buffer_push(buffer: BufferState, experience: tuple[Transition, ...]) -> BufferState:
    """Write rollout experiences into replay buffer."""
    batch = jax.tree.map(lambda *x: jnp.concatenate(x), *experience)
    num_steps = batch.observations.shape[0]
    indices = (buffer.idx + jnp.arange(num_steps)) % BUFFER_SIZE
    new_transitions = jax.tree.map(lambda buf, t: buf.at[indices].set(t), buffer.transitions, batch)
    return BufferState(
        transitions=new_transitions,
        idx=(buffer.idx + num_steps) % BUFFER_SIZE,
        size=jnp.minimum(buffer.size + num_steps, BUFFER_SIZE),
    )


@jax.jit
def select_action(
    network: QNetwork,
    observations: Float[Array, "n d"],
    epsilon: Float[Array, ""],
    *,
    key: PRNGKeyArray,
) -> Int[Array, " n"]:
    """Epsilon-greedy action selection."""
    key_random, key_choice = jax.random.split(key)
    num_envs = observations.shape[0]
    greedy_actions = network(observations).argmax(axis=-1)
    random_actions = jax.random.randint(key_random, (num_envs,), 0, ACTION_DIM)
    use_random = jax.random.uniform(key_choice, (num_envs,)) < epsilon
    return jnp.where(use_random, random_actions, greedy_actions)


def rollout(
    network: QNetwork,
    observations: np.ndarray,
    epsilon: float,
    *,
    key: PRNGKeyArray,
) -> tuple[np.ndarray, tuple[Transition, ...]]:
    """Collect rollout from vectorized environments."""
    experience = []

    for _ in range(ROLLOUT_STEPS):
        key, key_action = split(key)
        actions = np.asarray(
            select_action(network, jnp.asarray(observations), jnp.float32(epsilon), key=key_action)
        )

        next_observations, rewards, terminations, truncations, infos = envs.step(actions)

        experience.append(
            Transition(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations.astype(np.float32),
                truncations=truncations.astype(np.float32),
            )
        )

        # Track episode returns and reset done environments
        current_returns[:] += rewards
        dones = np.logical_or(terminations, truncations)
        if np.any(dones):
            for ret in current_returns[dones]:
                recent_returns.append(float(ret))
            current_returns[dones] = 0.0
            next_observations, _ = envs.reset(options={"reset_mask": dones})

        observations = next_observations

    return observations, tuple(experience)


@jax.jit
def learn(
    network: QNetwork,
    target_network: QNetwork,
    optimizer: ion.Optimizer,
    buffer: BufferState,
    *,
    key: PRNGKeyArray,
) -> tuple[QNetwork, QNetwork, ion.Optimizer]:
    """Double DQN update with soft target network update."""

    # Sample from replay buffer
    indices = jax.random.randint(key, (BATCH_SIZE,), 0, buffer.size)
    batch = jax.tree.map(lambda x: x[indices], buffer.transitions)

    def dqn_loss(network: QNetwork) -> Float[Array, ""]:
        # Double DQN
        next_actions = network(batch.next_observations).argmax(axis=-1)
        next_q = target_network(batch.next_observations)[jnp.arange(BATCH_SIZE), next_actions]
        targets = batch.rewards + GAMMA * (1.0 - batch.terminations) * next_q

        q_values = network(batch.observations)[jnp.arange(BATCH_SIZE), batch.actions]
        return ((targets - q_values) ** 2).mean()

    grads = jax.grad(dqn_loss)(network)
    network, optimizer = optimizer.update(network, grads)

    # Polyak soft update target network
    target_network = jax.tree.map(lambda t, o: TAU * o + (1.0 - TAU) * t, target_network, network)
    return network, target_network, optimizer


if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    TOTAL_STEPS = 1_000_000
    ROLLOUT_STEPS = 4
    NUM_ENVS = 8
    LR = 1e-3
    GAMMA = 0.99
    TAU = 0.05
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 128
    LEARNING_STARTS = 1_000
    EPS_START = 1.0
    EPS_FINAL = 0.05
    EPS_FRACTION = 0.5
    SEED = 42

    STEPS_PER_ROLLOUT = ROLLOUT_STEPS * NUM_ENVS
    TOTAL_ROLLOUTS = TOTAL_STEPS // STEPS_PER_ROLLOUT

    # Create gymnasium environments (manual reset via reset_mask)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(ENV_NAME) for _ in range(NUM_ENVS)],
        autoreset_mode="Disabled",
    )
    OBS_DIM = envs.single_observation_space.shape[0]
    ACTION_DIM = int(envs.single_action_space.n)

    # Initialize RNG
    rng = jax.random.key(SEED)
    rng, key_network = jax.random.split(rng)

    # Initialize network, target network, and optimizer
    network = QNetwork(OBS_DIM, ACTION_DIM, key=key_network)
    target_network = network
    optimizer = ion.Optimizer(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=LR)),
        network,
    )

    split = jax.jit(jax.random.split, static_argnums=1)

    # Initialize replay buffer and environments
    buffer = init_buffer(OBS_DIM)
    observations, _ = envs.reset(seed=SEED)

    # Episode tracking
    current_returns = np.zeros(NUM_ENVS)
    recent_returns: list[float] = []
    steps = 0

    bar = tqdm(total=TOTAL_STEPS, desc=f"DQN {ENV_NAME}")
    for rollout_idx in range(TOTAL_ROLLOUTS):
        rng, key_rollout, key_learn = split(rng)

        # Update epsilon
        epsilon = max(
            EPS_FINAL, EPS_START - (EPS_START - EPS_FINAL) * steps / (TOTAL_STEPS * EPS_FRACTION)
        )

        # Perform rollout gathering experience batch
        observations, experience = rollout(network, observations, epsilon, key=key_rollout)

        # Push experience batch to replay buffer
        buffer = buffer_push(buffer, experience)

        steps += STEPS_PER_ROLLOUT
        bar.update(STEPS_PER_ROLLOUT)

        # Update network parameters once replay buffer has accumulated enough transitions
        if steps >= LEARNING_STARTS:
            network, target_network, optimizer = learn(
                network, target_network, optimizer, buffer, key=key_learn
            )

        if recent_returns and steps % 1000 < STEPS_PER_ROLLOUT:
            mean_reward = np.mean(recent_returns[-100:])
            bar.set_postfix(reward=f"{mean_reward:.1f}", eps=f"{epsilon:.2f}")

    bar.close()
    envs.close()

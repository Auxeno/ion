"""DQN on CartPole-v1 with gymnasium vectorized environments.

Demonstrates replay buffers, target networks, and the `donate_argnums` trick
for memory-efficient buffer updates in JAX.
"""

from collections import deque
from functools import partial
from typing import Callable, NamedTuple

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
    """Dueling Q-network for estimating Q-values."""

    torso: nn.MLP
    advantage_head: nn.Linear
    value_head: nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, dim: int = 64, *, key: PRNGKeyArray) -> None:
        key_torso, key_a, key_v = jax.random.split(key, 3)
        self.torso = nn.MLP(obs_dim, dim, dim, num_hidden_layers=2, key=key_torso)
        self.advantage_head = nn.Linear(dim, action_dim, key=key_a)
        self.value_head = nn.Linear(dim, 1, key=key_v)

    def __call__(self, observations: Float[Array, "... d"]) -> Float[Array, "... a"]:
        x = jax.nn.relu(self.torso(observations))
        advantages = self.advantage_head(x)
        value = self.value_head(x)
        q = value + (advantages - advantages.mean(axis=-1, keepdims=True))
        return q


class Transition(NamedTuple):
    observations: Float[Array, "... d"]
    next_observations: Float[Array, "... d"]
    rewards: Float[Array, "..."]
    terminations: Float[Array, "..."]
    truncations: Float[Array, "..."]
    actions: Int[Array, "..."]


class BufferState(NamedTuple):
    """Circular replay buffer for storing transitions."""

    transitions: Transition
    idx: Int[Array, ""]
    size: Int[Array, ""]


def init_buffer(obs_shape: tuple[int, ...], buffer_size: int) -> BufferState:
    """Allocate an empty replay buffer."""
    return BufferState(
        transitions=Transition(
            observations=jnp.zeros((buffer_size, *obs_shape)),
            next_observations=jnp.zeros((buffer_size, *obs_shape)),
            rewards=jnp.zeros(buffer_size),
            terminations=jnp.zeros(buffer_size),
            truncations=jnp.zeros(buffer_size),
            actions=jnp.zeros(buffer_size, dtype=jnp.int32),
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
    envs: gym.vector.SyncVectorEnv,
    initial_observations: np.ndarray,
    epsilon: float,
    current_returns: np.ndarray,
    recent_returns: deque[float],
    *,
    key: PRNGKeyArray,
) -> tuple[np.ndarray, tuple[Transition, ...]]:
    """Collect transitions from vectorized environments."""
    experience = []

    observations = initial_observations
    for _ in range(ROLLOUT_STEPS):
        key, key_action = split(key)
        actions = np.asarray(
            select_action(network, jnp.asarray(observations), jnp.float32(epsilon), key=key_action)
        )

        next_observations, rewards, terminations, truncations, _ = envs.step(actions)

        experience.append(
            Transition(
                observations=observations,  # type: ignore[arg-type]
                next_observations=next_observations,  # type: ignore[arg-type]
                rewards=rewards,
                terminations=terminations.astype(np.float32),
                truncations=truncations.astype(np.float32),
                actions=actions,  # type: ignore[arg-type]
            )
        )

        # Track episode returns and reset done environments
        current_returns[:] += rewards
        dones = terminations | truncations
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
    indices = jax.random.randint(key, (BATCH_SIZE,), 0, buffer.size)
    batch = jax.tree.map(lambda x: x[indices], buffer.transitions)

    # Double DQN bootstrap targets
    next_actions = network(batch.next_observations).argmax(axis=-1)
    next_q = target_network(batch.next_observations)[jnp.arange(BATCH_SIZE), next_actions]
    targets = batch.rewards + GAMMA * (1.0 - batch.terminations) * next_q

    def loss_fn(network: QNetwork) -> Float[Array, ""]:
        q_values = network(batch.observations)[jnp.arange(BATCH_SIZE), batch.actions]
        return ((targets - q_values) ** 2).mean()

    grads = jax.grad(loss_fn)(network)
    network, optimizer = optimizer.update(network, grads)

    # Polyak soft update target network
    target_network = jax.tree.map(lambda t, o: TAU * o + (1.0 - TAU) * t, target_network, network)
    return network, target_network, optimizer


def train_dqn(
    network: QNetwork,
    env_fn: Callable[[], gym.Env],
    *,
    seed: int = 42,
) -> QNetwork:
    """Train a DQN agent on a gymnasium environment."""

    steps_per_rollout = ROLLOUT_STEPS * NUM_ENVS
    total_rollouts = TOTAL_STEPS // steps_per_rollout

    # Create vectorized environments (we manually reset)
    envs = gym.vector.SyncVectorEnv(
        [env_fn for _ in range(NUM_ENVS)],
        autoreset_mode="Disabled",
    )

    # Initialize target network and optimizer
    rng = jax.random.key(seed)
    target_network = network
    optimizer = ion.Optimizer(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=LR)),
        network,
    )

    # Initialize replay buffer
    obs_shape = envs.single_observation_space.shape  # type: ignore[union-attr]
    buffer = init_buffer(obs_shape, BUFFER_SIZE)  # type: ignore[arg-type]
    observations, _ = envs.reset(seed=seed)

    # Episode tracking
    current_returns = np.zeros(NUM_ENVS)
    recent_returns: deque[float] = deque(maxlen=100)
    steps = 0

    bar = tqdm(total=TOTAL_STEPS, desc="DQN")
    for rollout_idx in range(total_rollouts):
        rng, key_rollout, key_learn = split(rng, 3)

        epsilon = max(
            EPS_FINAL,
            EPS_START - (EPS_START - EPS_FINAL) * steps / (TOTAL_STEPS * EPS_FRACTION),
        )

        observations, experience = rollout(
            network, envs, observations, epsilon, current_returns, recent_returns, key=key_rollout
        )
        buffer = buffer_push(buffer, experience)

        steps += steps_per_rollout
        bar.update(steps_per_rollout)

        if steps >= LEARNING_STARTS:
            network, target_network, optimizer = learn(
                network, target_network, optimizer, buffer, key=key_learn
            )

        if recent_returns and steps % 1000 < steps_per_rollout:
            mean_reward = np.mean(recent_returns)
            bar.set_postfix(reward=f"{mean_reward:.1f}", eps=f"{epsilon:.2f}")

    bar.close()
    envs.close()
    return network


if __name__ == "__main__":
    GYMNASIUM_ENV_NAME = "CartPole-v1"
    TOTAL_STEPS = 1_000_000
    ROLLOUT_STEPS = 4
    NUM_ENVS = 8
    NETWORK_DIM = 64
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

    split = jax.jit(jax.random.split, static_argnums=1)

    # Create gymnasium environment
    sample_env = gym.make(GYMNASIUM_ENV_NAME)
    OBS_DIM = sample_env.observation_space.shape[0]  # type: ignore[index]
    ACTION_DIM = int(sample_env.action_space.n)  # type: ignore[attr-defined]
    sample_env.close()

    rng = jax.random.key(SEED)
    rng, key_network = jax.random.split(rng)
    network = QNetwork(OBS_DIM, ACTION_DIM, NETWORK_DIM, key=key_network)

    env_fn = lambda: gym.make(GYMNASIUM_ENV_NAME)
    trained = train_dqn(network, env_fn, seed=SEED)

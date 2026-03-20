"""PPO on CartPole-v1 with gymnax vectorized environments."""

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


class ActorCritic(nn.Module):
    """Actor-critic network for discrete action spaces."""

    actor: nn.MLP
    critic: nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, *, key: PRNGKeyArray) -> None:
        key_a, key_c = jax.random.split(key)
        self.actor = nn.MLP(obs_dim, action_dim, 64, 2, activation=jax.nn.tanh, key=key_a)
        self.critic = nn.MLP(obs_dim, 1, 64, 2, activation=jax.nn.tanh, key=key_c)

    def get_value(self, observations: Float[Array, "... d"]) -> Float[Array, "..."]:
        return self.critic(observations).squeeze(-1)

    def get_action_and_value(
        self,
        observations: Float[Array, "... d"],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Int[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Sample action, compute its log-prob and value estimate."""
        logits = self.actor(observations)
        action = jax.random.categorical(key, logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        value = self.critic(observations).squeeze(-1)
        return action, log_prob, value

    def get_log_prob_entropy_value(
        self,
        observations: Float[Array, "... d"],
        action: Int[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Compute action log-prob, entropy, and value estimate."""
        logits = self.actor(observations)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        value = self.critic(observations).squeeze(-1)
        return log_prob, entropy, value


class Transition(NamedTuple):
    observations: Float[Array, "... d"]
    next_observations: Float[Array, "... d"]
    rewards: Float[Array, "..."]
    terminations: Bool[Array, "..."]
    truncations: Bool[Array, "..."]
    actions: Int[Array, "..."]
    log_probs: Float[Array, "..."]
    values: Float[Array, "..."]


RolloutCarry = tuple[PRNGKeyArray, gymnax.EnvState, Float[Array, "n d"]]


@jax.jit
def rollout(
    network: ActorCritic,
    carry: RolloutCarry,
) -> tuple[RolloutCarry, Transition]:
    """Collect transitions from vectorized environments via lax.scan."""

    def step_fn(carry: RolloutCarry, _: None) -> tuple[RolloutCarry, Transition]:
        rng, env_states, observations = carry
        rng, key_action, key_step = jax.random.split(rng, 3)

        actions, log_probs, values = network.get_action_and_value(observations, key=key_action)

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
            log_probs,
            values,
        )
        return (rng, next_states, next_observations), transition

    new_carry, transitions = jax.lax.scan(f=step_fn, init=carry, xs=None, length=ROLLOUT_STEPS)
    return new_carry, transitions


def calculate_gae(
    rewards: Float[Array, "t n"],
    values: Float[Array, "t n"],
    next_values: Float[Array, "t n"],
    terminations: Bool[Array, "t n"],
    truncations: Bool[Array, "t n"],
    gamma: float,
    gae_lambda: float,
) -> Float[Array, "t n"]:
    """Compute GAE advantages via reversed scan over timesteps."""

    def gae_step(advantage, carry):
        reward, value, next_value, termination, truncation = carry
        non_termination = 1.0 - termination
        non_truncation = 1.0 - truncation
        delta = reward + gamma * next_value * non_termination - value
        advantage = delta + gamma * gae_lambda * non_termination * non_truncation * advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        f=gae_step,
        init=jnp.zeros(rewards.shape[1]),
        xs=(rewards, values, next_values, terminations, truncations),
        reverse=True,
    )
    return advantages


def ppo_loss(
    network: ActorCritic,
    observations: Float[Array, "b d"],
    actions: Int[Array, " b"],
    advantages: Float[Array, " b"],
    returns: Float[Array, " b"],
    old_log_probs: Float[Array, " b"],
) -> Float[Array, ""]:
    """PPO clipped surrogate loss with value MSE and entropy bonus."""
    new_log_probs, entropies, values = network.get_log_prob_entropy_value(observations, actions)

    # Clipped surrogate policy loss
    ratio = jnp.exp(new_log_probs - old_log_probs)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * jnp.clip(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
    policy_loss = jnp.maximum(loss_unclipped, loss_clipped).mean()

    # Value loss and entropy bonus
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    entropy_loss = -entropies.mean()

    return policy_loss + value_loss + entropy_loss * ENTROPY_BETA


@jax.jit
def learn(
    network: ActorCritic,
    optimizer: ion.Optimizer,
    batch: Transition,
    *,
    key: PRNGKeyArray,
) -> tuple[ActorCritic, ion.Optimizer]:
    """Compute GAE advantages then scan over minibatch PPO updates."""

    # Compute advantages with GAE
    next_values = jax.vmap(network.get_value)(batch.next_observations)
    advantages = calculate_gae(
        batch.rewards,
        batch.values,
        next_values,
        batch.terminations,
        batch.truncations,
        GAMMA,
        GAE_LAMBDA,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    returns = advantages + batch.values

    # Flatten (rollout_steps, num_envs, ...) -> (batch_size, ...)
    batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    advantages, returns = advantages.flatten(), returns.flatten()

    # Shuffled minibatch indices
    indices = jnp.tile(jnp.arange(BATCH_SIZE, dtype=jnp.int32), (NUM_EPOCHS, 1))
    mb_indices = jax.vmap(jax.random.permutation)(jax.random.split(key, NUM_EPOCHS), indices)
    mb_indices = mb_indices.reshape(NUM_EPOCHS * NUM_MINIBATCHES, MINIBATCH_SIZE)

    def minibatch_update(carry, indices):
        network, optimizer = carry
        loss, grads = jax.value_and_grad(ppo_loss)(
            network,
            batch.observations[indices],
            batch.actions[indices],
            advantages[indices],
            returns[indices],
            batch.log_probs[indices],
        )
        network, optimizer = optimizer.update(network, grads)
        return (network, optimizer), loss

    (network, optimizer), _ = jax.lax.scan(minibatch_update, (network, optimizer), mb_indices)
    return network, optimizer


def train_ppo(
    network: ActorCritic,
    *,
    seed: int = 42,
) -> ActorCritic:
    """Train a PPO agent on a gymnax environment."""

    rng = jax.random.key(seed)
    rng, key_reset, rng_rollout = jax.random.split(rng, 3)

    # Initialize optimizer
    optimizer = ion.Optimizer(
        optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=LR, eps=1e-5)),
        network,
    )

    # Reset vectorized environments
    observations, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key_reset, NUM_ENVS), env_params
    )
    carry = (rng_rollout, env_states, observations)

    # Episode tracking
    current_returns = np.zeros(NUM_ENVS)
    recent_returns: deque[float] = deque(maxlen=100)
    total_rollouts = TOTAL_STEPS // BATCH_SIZE
    checkpoints = {total_rollouts * p // 10 for p in range(1, 11)}

    bar = tqdm(range(total_rollouts), desc=f"PPO {GYMNAX_ENV_NAME}")
    for i in bar:
        rng, key_learn = jax.random.split(rng)

        carry, transitions = rollout(network, carry)
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
            bar.set_postfix(reward=f"{mean_reward:.1f}")
            if i + 1 in checkpoints:
                tqdm.write(f"  Step {(i + 1) * BATCH_SIZE:>9,} | Mean reward: {mean_reward:.1f}")

    return network


if __name__ == "__main__":
    GYMNAX_ENV_NAME = "CartPole-v1"
    TOTAL_STEPS = 1_000_000
    ROLLOUT_STEPS = 64
    NUM_ENVS = 16
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    NUM_EPOCHS = 8
    NUM_MINIBATCHES = 4
    PPO_CLIP = 0.2
    ENTROPY_BETA = 0.01
    SEED = 42

    BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
    MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES

    # Create gymnax environment
    env, env_params = gymnax.make(GYMNAX_ENV_NAME)
    OBS_DIM = env.observation_space(env_params).shape[0]  # type: ignore[reportArgumentType]
    ACTION_DIM = int(env.action_space(env_params).n)  # type: ignore[reportArgumentType]

    rng = jax.random.key(SEED)
    rng, key_network = jax.random.split(rng)
    network = ActorCritic(OBS_DIM, ACTION_DIM, key=key_network)

    trained = train_ppo(network, seed=SEED)

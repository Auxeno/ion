"""PPO on CartPole-v1 with gymnax vectorized environments."""

from collections import deque
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

import ion
from ion import nn


TOTAL_STEPS = 400_000
ROLLOUT_STEPS = 64
NUM_ENVS = 16
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_EPOCHS = 8
NUM_MINIBATCHES = 4
PPO_CLIP = 0.2
ENTROPY_BETA = 0.01
GRAD_NORM_CLIP = 0.5
HIDDEN_DIM = 64
NUM_HIDDEN_LAYERS = 2
SEED = 42

BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
TOTAL_ROLLOUTS = TOTAL_STEPS // BATCH_SIZE


class ActorCritic(nn.Module):
    """Actor-critic network for discrete action spaces."""

    actor: nn.MLP
    critic: nn.MLP

    def __init__(self, obs_dim: int, act_dim: int, *, key: PRNGKeyArray) -> None:
        key_a, key_c = jax.random.split(key)
        self.actor = nn.MLP(
            obs_dim,
            act_dim,
            HIDDEN_DIM,
            NUM_HIDDEN_LAYERS,
            activation=jax.nn.tanh,
            w_init=jax.nn.initializers.orthogonal(),
            key=key_a,
        )
        self.critic = nn.MLP(
            obs_dim,
            1,
            HIDDEN_DIM,
            NUM_HIDDEN_LAYERS,
            activation=jax.nn.tanh,
            w_init=jax.nn.initializers.orthogonal(),
            key=key_c,
        )

    def get_value(self, obs: Float[Array, " O"]) -> Float[Array, ""]:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: Float[Array, " O"],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Int[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Sample action, compute its log-prob and value estimate."""
        logits = self.actor(obs)
        action = jax.random.categorical(key, logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def get_log_prob_entropy_value(
        self,
        obs: Float[Array, " O"],
        action: Int[Array, ""],
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Compute action log-prob, entropy, and value estimate."""
        logits = self.actor(obs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


def calculate_gae(
    rewards: Float[Array, "T N"],
    values: Float[Array, "T N"],
    next_values: Float[Array, "T N"],
    terminations: Float[Array, "T N"],
    truncations: Float[Array, "T N"],
    gamma: float,
    gae_lambda: float,
) -> Float[Array, "T N"]:
    """Compute GAE advantages via a reversed scan over timesteps."""

    def gae_step(advantage, carry):
        reward, value, next_value, termination, truncation = carry
        non_termination = 1.0 - termination
        non_truncation = 1.0 - truncation

        # TD-error + recursive GAE formula
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
    observations: Float[Array, "B O"],
    actions: Int[Array, " B"],
    advantages: Float[Array, " B"],
    returns: Float[Array, " B"],
    old_log_probs: Float[Array, " B"],
) -> Float[Array, ""]:
    """Combined PPO-Clip loss: clipped surrogate + value MSE + entropy."""
    new_log_probs, entropies, values = jax.vmap(network.get_log_prob_entropy_value)(
        observations, actions
    )

    # Clipped surrogate policy loss
    ratio = jnp.exp(new_log_probs - old_log_probs)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * jnp.clip(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
    policy_loss = jnp.maximum(loss_unclipped, loss_clipped).mean()

    # Value function loss (MSE)
    value_loss = 0.5 * ((returns - values) ** 2).mean()

    # Entropy bonus (negative to maximize entropy)
    entropy_loss = -entropies.mean()

    return policy_loss + value_loss + entropy_loss * ENTROPY_BETA


class Transition(NamedTuple):
    observations: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    values: jax.Array


RolloutCarry = tuple[PRNGKeyArray, gymnax.EnvState, Float[Array, "N O"]]


@jax.jit
def rollout(
    network: ActorCritic,
    carry: RolloutCarry,
) -> tuple[RolloutCarry, Transition]:
    """Collect transitions from vectorized environments via scan."""

    def step_fn(carry: RolloutCarry, _: None) -> tuple[RolloutCarry, Transition]:
        rng, env_states, obs = carry
        rng, key_action, key_step = jax.random.split(rng, 3)

        # Select actions from policy
        actions, log_probs, values = jax.vmap(lambda o, k: network.get_action_and_value(o, key=k))(
            obs, jax.random.split(key_action, NUM_ENVS)
        )

        # Step vectorized environments
        next_obs, next_states, rewards, terminations, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(
            jax.random.split(key_step, NUM_ENVS),
            env_states,
            actions,
            env_params,
        )
        truncations = jnp.zeros_like(terminations)  # dummy truncations

        transition = Transition(
            obs,
            next_obs,
            rewards,
            terminations,
            truncations,
            actions,
            log_probs,
            values,
        )
        return (rng, next_states, next_obs), transition

    return jax.lax.scan(step_fn, carry, None, length=ROLLOUT_STEPS)


@jax.jit
def learn(
    network: ActorCritic,
    opt_state: optax.OptState,
    batch: Transition,
    *,
    key: PRNGKeyArray,
) -> tuple[ActorCritic, optax.OptState]:
    """PPO update: compute GAE then scan over minibatch gradient steps."""

    # Compute advantages with GAE
    next_values = jax.vmap(jax.vmap(network.get_value))(batch.next_observations)
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

    # Generate shuffled minibatch indices
    minibatch_size = BATCH_SIZE // NUM_MINIBATCHES
    indices = jnp.tile(jnp.arange(BATCH_SIZE, dtype=jnp.int32), (NUM_EPOCHS, 1))
    mb_indices = jax.vmap(jax.random.permutation)(jax.random.split(key, NUM_EPOCHS), indices)
    mb_indices = mb_indices.reshape(NUM_EPOCHS * NUM_MINIBATCHES, minibatch_size)

    def minibatch_update(carry, indices):
        network, opt_state = carry

        # Compute PPO loss and parameter gradients
        loss, grads = ion.value_and_grad(ppo_loss)(
            network,
            batch.observations[indices],
            batch.actions[indices],
            advantages[indices],
            returns[indices],
            batch.log_probs[indices],
        )

        # Apply optimizer update
        updates, opt_state = optimizer.update(grads, opt_state)
        network = ion.apply_updates(network, updates)
        return (network, opt_state), loss

    (network, opt_state), _ = jax.lax.scan(minibatch_update, (network, opt_state), mb_indices)
    return network, opt_state


if __name__ == "__main__":
    # Create Gymnax environment
    env, env_params = gymnax.make("CartPole-v1")
    obs_dim = env.observation_space(env_params).shape[0]  # type: ignore[reportArgumentType]
    act_dim = int(env.action_space(env_params).n)  # type: ignore[reportArgumentType]

    # Initialize RNG
    rng = jax.random.key(SEED)
    rng, key_network, key_reset, rng_rollout = jax.random.split(rng, 4)

    # Initialize network and optimizer
    network = ActorCritic(obs_dim, act_dim, key=key_network)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_NORM_CLIP),
        optax.adam(learning_rate=LR, eps=1e-5),
    )
    opt_state = optimizer.init(network.params)

    # Reset vectorized environments
    observations, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key_reset, NUM_ENVS), env_params
    )
    carry = (rng_rollout, env_states, observations)

    # Episode tracking
    current_returns = np.zeros(NUM_ENVS)
    recent_returns: deque[float] = deque(maxlen=100)
    mean_reward = 0.0
    checkpoints = {TOTAL_ROLLOUTS * p // 4 for p in range(1, 5)}

    bar = tqdm(range(TOTAL_ROLLOUTS), desc="PPO CartPole-v1")
    for i in bar:
        rng, key_learn = jax.random.split(rng)

        # Collect transitions and update policy
        carry, transitions = rollout(network, carry)
        network, opt_state = learn(network, opt_state, transitions, key=key_learn)

        # Track episode statistics
        t = jax.device_get(transitions)
        rewards_np = np.asarray(t.rewards)
        dones_np = np.asarray(t.terminations | t.truncations)
        for step_r, step_d in zip(rewards_np, dones_np):
            current_returns += step_r
            for ret in current_returns[step_d]:
                recent_returns.append(float(ret))
            current_returns[step_d] = 0.0

        if recent_returns:
            mean_reward = np.mean(recent_returns)
            bar.set_postfix(reward=f"{mean_reward:.1f}")
        if i + 1 in checkpoints and recent_returns:
            step = (i + 1) * BATCH_SIZE
            tqdm.write(f"  Step {step:>9,} | Mean reward: {mean_reward:.1f}")

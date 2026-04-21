"""
Proximal Policy Optimization (PPO-Clip) agent.

PPO collects a rollout buffer of N steps, then performs K epochs of minibatch
gradient updates using the clipped surrogate objective.  The clip prevents the
policy from changing too drastically in a single update, which stabilises training.

Key hyperparameters to tune:
    clip_epsilon  – clipping range for the probability ratio r_t(θ); typical 0.1–0.3
    epochs        – number of gradient epochs per rollout; typical 3–10
    batch_size    – minibatch size; typical 32–256
    gae_lambda    – GAE smoothing parameter; 0 = TD(0), 1 = Monte Carlo
    entropy_coef  – weight of the entropy bonus; encourages exploration
    value_coef    – weight of the critic loss relative to policy loss

Implementation outline (update method TODO):
    1. Compute GAE advantages and discounted returns.
    2. For `epochs` epochs, shuffle data into minibatches and for each:
        a. Re-evaluate log_probs and state values under the current policy.
        b. Compute probability ratio r = exp(new_log_prob - old_log_prob).
        c. Policy loss  = -mean( min(r*A, clip(r, 1-ε, 1+ε)*A) )
        d. Value loss   = mean( (V(s) - returns)^2 )
        e. Entropy loss = -mean( entropy of action distribution )
        f. Total loss   = policy_loss + value_coef*value_loss - entropy_coef*entropy
    3. Backprop total_loss, clip gradients, optimizer step.
    4. Clear rollout_buffer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base_agent import BaseAgent, ActorCritic


class PPOAgent(BaseAgent):
    """PPO-Clip with Generalised Advantage Estimation (GAE)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 64,
    ):
        super().__init__(obs_dim, action_dim)
        self.clip_epsilon  = clip_epsilon
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(obs_dim, action_dim, hidden_dim)
        self.optimizer    = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Buffer stores dicts with keys:
        # obs, action, reward, done, log_prob, value
        self.rollout_buffer: list = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Sample an action from the current policy."""
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.actor_critic(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def select_action_with_info(self, obs: np.ndarray):
        """Sample action and return (action, log_prob, value) for buffer storage."""
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.actor_critic(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store_transition(self, obs, action, reward, done, log_prob, value) -> None:
        """Append a single transition to the rollout buffer."""
        self.rollout_buffer.append({
            "obs":      np.array(obs,    dtype=np.float32),
            "action":   int(action),
            "reward":   float(reward),
            "done":     bool(done),
            "log_prob": float(log_prob),
            "value":    float(value),
        })

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: dict = None) -> dict:
        """Run PPO update on the current rollout buffer.

        TODO: Implement the PPO update loop (see module docstring).
        The rollout_buffer contains transitions in chronological order.
        After updating, clear rollout_buffer.
        """
        if not self.rollout_buffer:
            return {}

        # TODO: implement _compute_gae() and the K-epoch minibatch loop
        self.rollout_buffer.clear()
        return {}

    def _compute_gae(self, last_value: float = 0.0):
        """Compute Generalised Advantage Estimates (GAE-λ).

        Args:
            last_value: V(s_T) bootstrap value for the last state.

        Returns:
            advantages (np.ndarray), returns (np.ndarray)

        TODO: Implement backward pass over rollout_buffer:
            delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t       = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            returns_t = A_t + V(s_t)
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "model":     self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path)
        self.actor_critic.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

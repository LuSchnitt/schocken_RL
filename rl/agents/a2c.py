"""
Advantage Actor-Critic (A2C) agent.

A2C updates the actor and critic simultaneously after every n_steps transitions,
using the advantage A(s,a) = R_t - V(s) as a variance-reducing baseline.

Compared to PPO:
    + Simpler implementation (no clipping, single gradient pass per update).
    + Lower computational overhead per update.
    - Less stable; sensitive to learning rate.
    - No experience replay or minibatching.

Compared to Tabular Q-Learning:
    + Generalises across states via neural networks.
    + Works with continuous or large observation spaces.

Implementation outline (update method TODO):
    1. Compute discounted returns R_t backwards through the trajectory.
    2. Forward pass: get log_probs and V(s) for all stored observations.
    3. Advantage: A_t = R_t - V(s_t)   (detach from graph for actor loss)
    4. Actor  loss = -mean( log_prob(a_t) * A_t )
    5. Critic loss =  mean( (R_t - V(s_t))^2 )
    6. Entropy     = -mean( sum_a π(a|s) * log π(a|s) )   (encourages exploration)
    7. Total  loss =  actor_loss + value_coef * critic_loss - entropy_coef * entropy
    8. Zero grad, backprop, clip gradients, optimizer step.
    9. Clear trajectory.

Typical hyperparameters:
    lr           = 7e-4
    gamma        = 0.99
    n_steps      = 5    (update every 5 env steps)
    entropy_coef = 0.01
    value_coef   = 0.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base_agent import BaseAgent, ActorCritic


class A2CAgent(BaseAgent):
    """Synchronous Advantage Actor-Critic (A2C)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 64,
    ):
        super().__init__(obs_dim, action_dim)
        self.gamma         = gamma
        self.n_steps       = n_steps
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(obs_dim, action_dim, hidden_dim)
        # RMSprop is conventional for A2C (matches the original paper)
        self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=lr, eps=1e-5)

        # Trajectory buffer — cleared after each update
        # Each entry: {'obs': ..., 'action': ..., 'reward': ..., 'done': ...}
        self.trajectory: list = []

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

    def store_transition(self, obs, action: int, reward: float, done: bool) -> None:
        """Append a single (s, a, r, done) to the trajectory buffer."""
        self.trajectory.append({
            "obs":    np.array(obs, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "done":   bool(done),
        })

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: dict = None) -> dict:
        """A2C update using the current trajectory buffer.

        Called automatically when len(trajectory) >= n_steps or on episode end.

        TODO: Implement the A2C update (see module docstring).
        After updating, clear self.trajectory.
        """
        if not self.trajectory:
            return {}

        # TODO: implement discounted returns, advantage, actor + critic losses
        self.trajectory.clear()
        return {}

    def _compute_returns(self, last_value: float = 0.0) -> np.ndarray:
        """Compute discounted returns backwards through the trajectory.

        Args:
            last_value: V(s_T) bootstrap for non-terminal last states.

        Returns:
            np.ndarray of shape (len(trajectory),)

        TODO: Implement:
            R_T        = last_value
            R_t        = r_t + gamma * R_{t+1} * (1 - done_t)
        """
        # TODO
        raise NotImplementedError

    def ready_to_update(self) -> bool:
        """Return True when the trajectory has n_steps transitions."""
        return len(self.trajectory) >= self.n_steps

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

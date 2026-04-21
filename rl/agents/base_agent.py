"""
Shared neural network architectures and abstract base agent.

Networks
--------
PolicyNet     – observation -> action probabilities (for standalone policy agents)
ValueNet      – observation -> scalar value         (for standalone critic)
ActorCritic   – shared backbone + actor head + critic head (used by PPO and A2C)

BaseAgent
---------
Abstract class that every concrete RL agent must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """Maps an observation to a probability distribution over actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    """Maps an observation to a scalar state value V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Combined actor-critic with a shared feature backbone.

    Used by both PPO and A2C.  A shared backbone helps with sample efficiency;
    separate heads keep the policy and value objectives decoupled.

    forward() returns (action_probs, state_value):
        action_probs : Tensor[batch, action_dim]  (Softmax output)
        state_value  : Tensor[batch, 1]
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        probs = self.actor_head(features)
        value = self.critic_head(features)
        return probs, value


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class every RL agent must subclass."""

    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Choose an action given an observation.

        Called during both training (exploration) and evaluation (greedy).
        """

    @abstractmethod
    def update(self, batch: dict) -> dict:
        """Update agent parameters from a collected batch of experience.

        Args:
            batch: dict with keys such as 'obs', 'action', 'reward',
                   'next_obs', 'done' — exact schema depends on the agent.

        Returns:
            dict of scalar training metrics (e.g. {'loss': 0.42}).
        """

    def save(self, path: str) -> None:
        """Persist the agent to disk."""
        raise NotImplementedError(f"{self.__class__.__name__}.save() not implemented")

    def load(self, path: str) -> None:
        """Restore the agent from disk."""
        raise NotImplementedError(f"{self.__class__.__name__}.load() not implemented")

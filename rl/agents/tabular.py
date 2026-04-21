"""
Tabular Q-Learning agent.

The continuous observation is discretised into a hashable string key so that
Q-values can be stored in a plain Python dict.  This is the simplest possible
RL approach and serves as a proof-of-concept before moving to neural methods.

Key design choices to make when implementing:
    - Discretisation granularity (_discretize): coarser bins -> smaller table,
      faster convergence, less precision.  Start with 1 decimal place.
    - Epsilon decay schedule: linear or exponential; tune epsilon_min.
    - Whether to use Q-Learning (off-policy) or SARSA (on-policy).
"""

import pickle
import numpy as np
from .base_agent import BaseAgent


class TabularQAgent(BaseAgent):
    """ε-greedy Tabular Q-Learning agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        alpha: float = 0.1,        # learning rate
        gamma: float = 0.99,       # discount factor
        epsilon: float = 1.0,      # initial exploration rate
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(obs_dim, action_dim)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict = {}  # state_key -> np.ndarray of shape (action_dim,)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        key = self._discretize(obs)
        return int(np.argmax(self._get_q(key)))

    def update(self, batch: dict) -> dict:
        """Q-Learning update from a single (s, a, r, s', done) transition.

        Expected batch keys: 'obs', 'action', 'reward', 'next_obs', 'done'

        TODO: Implement the Q-Learning update rule:
            current_q  = Q(s, a)
            target_q   = r + gamma * max_a' Q(s', a') * (1 - done)
            Q(s, a)   += alpha * (target_q - current_q)
        """
        obs      = batch["obs"]
        action   = batch["action"]
        reward   = batch["reward"]
        next_obs = batch["next_obs"]
        done     = batch["done"]

        key      = self._discretize(obs)
        next_key = self._discretize(next_obs)

        q_values      = self._get_q(key)
        next_q_values = self._get_q(next_key)

        # TODO: compute target and update q_values[action]
        # target = reward + self.gamma * np.max(next_q_values) * (1 - done)
        # q_values[action] += self.alpha * (target - q_values[action])

        return {"q_table_size": len(self.q_table)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discretize(self, obs: np.ndarray) -> str:
        """Convert a continuous observation to a hashable dict key.

        TODO: Choose an appropriate rounding precision.
        Finer precision = larger table.  Start with 1 decimal place.
        Example: np.round(obs, 1) -> tuple -> string
        """
        # TODO
        return str(tuple(np.round(obs, 1)))

    def _get_q(self, state_key: str) -> np.ndarray:
        """Return Q-values for a state, initialising to zeros on first visit."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        return self.q_table[state_key]

    def decay_epsilon(self) -> None:
        """Reduce exploration rate. Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]

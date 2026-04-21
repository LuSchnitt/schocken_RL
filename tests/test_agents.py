"""
Tests for the RL agents and environment.

Run with: python -m pytest tests/test_agents.py -v

Tests that depend on unimplemented stubs are marked xfail.
"""

import numpy as np
import pytest

from rl.agents.base_agent import PolicyNet, ValueNet, ActorCritic
from rl.agents.tabular import TabularQAgent
from rl.agents.ppo import PPOAgent
from rl.agents.a2c import A2CAgent


OBS_DIM    = 11   # matches SchockenEnv with 3 players
ACTION_DIM = 9


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

def test_policy_net_output_shape():
    import torch
    net = PolicyNet(OBS_DIM, ACTION_DIM)
    x   = torch.zeros(1, OBS_DIM)
    out = net(x)
    assert out.shape == (1, ACTION_DIM)

def test_policy_net_output_sums_to_one():
    import torch
    net = PolicyNet(OBS_DIM, ACTION_DIM)
    x   = torch.zeros(4, OBS_DIM)
    out = net(x)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)

def test_value_net_output_shape():
    import torch
    net = ValueNet(OBS_DIM)
    x   = torch.zeros(1, OBS_DIM)
    assert net(x).shape == (1, 1)

def test_actor_critic_shapes():
    import torch
    net       = ActorCritic(OBS_DIM, ACTION_DIM)
    x         = torch.zeros(2, OBS_DIM)
    probs, v  = net(x)
    assert probs.shape == (2, ACTION_DIM)
    assert v.shape     == (2, 1)


# ---------------------------------------------------------------------------
# Tabular Q-Learning
# ---------------------------------------------------------------------------

def test_tabular_action_in_range():
    agent = TabularQAgent(OBS_DIM, ACTION_DIM)
    obs   = np.zeros(OBS_DIM)
    for _ in range(50):
        assert 0 <= agent.select_action(obs) < ACTION_DIM

def test_tabular_epsilon_decay():
    agent = TabularQAgent(OBS_DIM, ACTION_DIM, epsilon=1.0, epsilon_decay=0.9)
    agent.decay_epsilon()
    assert agent.epsilon < 1.0

def test_tabular_epsilon_floor():
    agent = TabularQAgent(OBS_DIM, ACTION_DIM, epsilon=0.05, epsilon_min=0.05)
    agent.decay_epsilon()
    assert agent.epsilon == 0.05

@pytest.mark.xfail(reason="TabularQAgent.update() Q-learning not yet implemented")
def test_tabular_update_changes_q_value():
    agent = TabularQAgent(OBS_DIM, ACTION_DIM, epsilon=0.0)
    obs   = np.ones(OBS_DIM) * 0.5
    batch = {"obs": obs, "action": 0, "reward": 1.0,
             "next_obs": obs, "done": False}
    agent.update(batch)
    key = agent._discretize(obs)
    assert agent.q_table[key][0] != 0.0  # should have been updated


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def test_ppo_action_in_range():
    agent = PPOAgent(OBS_DIM, ACTION_DIM)
    obs   = np.zeros(OBS_DIM)
    for _ in range(20):
        assert 0 <= agent.select_action(obs) < ACTION_DIM

def test_ppo_select_action_with_info():
    agent            = PPOAgent(OBS_DIM, ACTION_DIM)
    obs              = np.zeros(OBS_DIM)
    action, lp, val  = agent.select_action_with_info(obs)
    assert 0 <= action < ACTION_DIM
    assert isinstance(lp,  float)
    assert isinstance(val, float)

def test_ppo_store_transition():
    agent = PPOAgent(OBS_DIM, ACTION_DIM)
    obs   = np.zeros(OBS_DIM)
    agent.store_transition(obs, 0, 1.0, False, -0.5, 0.1)
    assert len(agent.rollout_buffer) == 1

def test_ppo_update_clears_buffer():
    agent = PPOAgent(OBS_DIM, ACTION_DIM)
    obs   = np.zeros(OBS_DIM)
    for _ in range(5):
        agent.store_transition(obs, 0, 0.0, False, -1.0, 0.0)
    agent.update()
    assert len(agent.rollout_buffer) == 0


# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------

def test_a2c_action_in_range():
    agent = A2CAgent(OBS_DIM, ACTION_DIM)
    obs   = np.zeros(OBS_DIM)
    for _ in range(20):
        assert 0 <= agent.select_action(obs) < ACTION_DIM

def test_a2c_ready_to_update():
    agent = A2CAgent(OBS_DIM, ACTION_DIM, n_steps=3)
    obs   = np.zeros(OBS_DIM)
    assert not agent.ready_to_update()
    for _ in range(3):
        agent.store_transition(obs, 0, 0.0, False)
    assert agent.ready_to_update()

def test_a2c_update_clears_trajectory():
    agent = A2CAgent(OBS_DIM, ACTION_DIM, n_steps=2)
    obs   = np.zeros(OBS_DIM)
    agent.store_transition(obs, 0, 1.0, False)
    agent.store_transition(obs, 1, 0.0, True)
    agent.update()
    assert len(agent.trajectory) == 0


# ---------------------------------------------------------------------------
# Environment (requires game to be implemented)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="SchockenEnv.reset() not yet implemented")
def test_env_reset_obs_shape():
    from rl.environment import SchockenEnv
    env = SchockenEnv(n_players=3)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32

@pytest.mark.xfail(reason="SchockenEnv.step() not yet implemented")
def test_env_step_returns_correct_types():
    from rl.environment import SchockenEnv
    env  = SchockenEnv(n_players=3)
    obs  = env.reset()
    obs2, reward, done, info = env.step(0)
    assert obs2.shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(done,   bool)
    assert isinstance(info,   dict)

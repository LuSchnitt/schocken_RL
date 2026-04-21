"""
train.py – Train an RL agent to play Schocken.

Examples:
    python train.py --agent tabular --episodes 50000
    python train.py --agent ppo     --episodes 10000 --n_players 3
    python train.py --agent a2c     --episodes 10000 --save_path models/a2c.pt
"""

import argparse
import csv
import os
import time

from rl.environment import SchockenEnv
from rl.agents.tabular import TabularQAgent
from rl.agents.ppo import PPOAgent
from rl.agents.a2c import A2CAgent

AGENT_CLASSES = {
    "tabular": TabularQAgent,
    "ppo":     PPOAgent,
    "a2c":     A2CAgent,
}


def train(args):
    env       = SchockenEnv(n_players=args.n_players)
    agent_cls = AGENT_CLASSES[args.agent]
    agent     = agent_cls(obs_dim=env.obs_dim, action_dim=env.NUM_ACTIONS)

    print(f"Training {args.agent.upper()} | "
          f"{args.episodes} episodes | {args.n_players} players")

    log_rows = []
    episode_rewards = []
    start = time.time()

    for ep in range(1, args.episodes + 1):
        obs  = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # ----- action selection -----
            if args.agent == "ppo":
                action, log_prob, value = agent.select_action_with_info(obs)
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = env.step(action)

            # ----- store transition / update -----
            if args.agent == "tabular":
                agent.update({
                    "obs": obs, "action": action, "reward": reward,
                    "next_obs": next_obs, "done": done,
                })
            elif args.agent == "ppo":
                agent.store_transition(obs, action, reward, done, log_prob, value)
                if len(agent.rollout_buffer) >= args.rollout_steps:
                    agent.update()
            elif args.agent == "a2c":
                agent.store_transition(obs, action, reward, done)
                if agent.ready_to_update() or done:
                    agent.update()

            obs           = next_obs
            total_reward += reward

        # ----- per-episode bookkeeping -----
        if args.agent == "tabular":
            agent.decay_epsilon()

        episode_rewards.append(total_reward)
        log_rows.append({"episode": ep, "reward": total_reward})

        if ep % max(1, args.episodes // 20) == 0:
            avg = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            elapsed = time.time() - start
            eps_str = (f"  ε={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else "")
            print(f"  ep {ep:6d} | avg reward (100): {avg:+.3f}{eps_str} "
                  f"| {elapsed:.0f}s elapsed")

    # ----- save model -----
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        agent.save(args.save_path)
        print(f"Model saved -> {args.save_path}")

    # ----- save reward log -----
    if args.log_path:
        with open(args.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "reward"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Reward log  -> {args.log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Schocken RL agent")
    parser.add_argument("--agent", choices=list(AGENT_CLASSES.keys()),
                        default="tabular", help="RL algorithm to use")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Number of training episodes")
    parser.add_argument("--n_players", type=int, default=3,
                        help="Total number of players (1 RL agent + N-1 CPU)")
    parser.add_argument("--rollout_steps", type=int, default=128,
                        help="Steps per PPO rollout (only used for --agent ppo)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the trained model (optional)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Where to save episode reward CSV (optional)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

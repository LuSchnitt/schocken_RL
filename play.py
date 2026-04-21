"""
play.py – Start an interactive Schocken game from the command line.

Examples:
    # 1 human vs 2 random CPUs
    python play.py --human_players 1 --cpu_players 2

    # 2 humans vs 1 rule-based CPU
    python play.py --human_players 2 --cpu_players 1 --cpu_type rule

    # 1 human vs 1 trained PPO agent
    python play.py --human_players 1 --rl_agent ppo --model_path models/ppo.pt
"""

import argparse
from schocken.game import SchockenGame
from schocken.players import HumanPlayer, RandomCPUPlayer, RuleBasedCPUPlayer


def build_players(args) -> list:
    players = []

    for i in range(args.human_players):
        name = input(f"Enter name for human player {i + 1}: ").strip()
        players.append(HumanPlayer(name or f"Human_{i + 1}"))

    cpu_cls = RuleBasedCPUPlayer if args.cpu_type == "rule" else RandomCPUPlayer
    for i in range(args.cpu_players):
        players.append(cpu_cls(f"CPU_{i + 1}"))

    if args.rl_agent:
        # TODO: load the chosen RL agent model and wrap it in a Player subclass
        # so that SchockenGame can call choose_kept_dice / should_reroll.
        # Example skeleton:
        #
        #   from rl.environment import SchockenEnv
        #   from rl.agents import TabularQAgent, PPOAgent, A2CAgent
        #   from schocken.players import Player
        #
        #   class RLPlayer(Player):
        #       def __init__(self, name, agent, env):
        #           super().__init__(name)
        #           self.agent = agent
        #           self.env   = env
        #
        #       def choose_kept_dice(self, state):
        #           obs    = self.env._encode_state()   # reuse env encoding
        #           action = self.agent.select_action(obs)
        #           return self.env._decode_action(action)
        #
        #       def should_reroll(self, state):
        #           # action == ACTION_END_TURN means stop
        #           obs    = self.env._encode_state()
        #           action = self.agent.select_action(obs)
        #           return action != ACTION_END_TURN
        raise NotImplementedError(
            "RL agent play integration not yet implemented. "
            "See the TODO in play.py::build_players()."
        )

    return players


def main():
    parser = argparse.ArgumentParser(description="Play Schocken interactively")
    parser.add_argument("--human_players", type=int, default=1,
                        help="Number of human (CLI) players")
    parser.add_argument("--cpu_players", type=int, default=2,
                        help="Number of CPU players")
    parser.add_argument("--cpu_type", choices=["random", "rule"], default="rule",
                        help="CPU strategy: 'random' baseline or 'rule' heuristic")
    parser.add_argument("--rl_agent", choices=["tabular", "ppo", "a2c"], default=None,
                        help="Add a trained RL agent as a player")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved RL model (required if --rl_agent is set)")
    args = parser.parse_args()

    if args.rl_agent and not args.model_path:
        parser.error("--model_path is required when --rl_agent is specified")

    players = build_players(args)

    if len(players) < 2:
        print("Need at least 2 players total.")
        return

    print(f"\nStarting Schocken with: {[p.name for p in players]}\n")
    game = SchockenGame(players)
    loser = game.run()
    print(f"\n=== Game over!  The loser is: {loser} ===")


if __name__ == "__main__":
    main()

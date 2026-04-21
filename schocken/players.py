"""
Player classes for the Schocken simulator.

Hierarchy:
    Player (ABC)
    ├── HumanPlayer       – reads moves from stdin
    ├── RandomCPUPlayer   – random decisions (baseline for RL comparison)
    └── RuleBasedCPUPlayer – heuristic decisions (TODO: implement)
"""

import random
from abc import ABC, abstractmethod
from .game import GameState, NUM_DICE


class Player(ABC):
    """Abstract base class for all player types."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_kept_dice(self, state: GameState) -> list:
        """Decide which dice to keep before a potential reroll.

        Args:
            state: current GameState (read-only, do not modify)

        Returns:
            list of booleans, length == len(state.current_dice).
            True means keep that die; False means reroll it.
        """

    @abstractmethod
    def should_reroll(self, state: GameState) -> bool:
        """Decide whether to use another roll.

        Called only when rolls remain (state.rolls_used < state.max_rolls).

        Returns:
            True to reroll at least one die, False to accept the current hand.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"


# ---------------------------------------------------------------------------
# Human player (CLI)
# ---------------------------------------------------------------------------

class HumanPlayer(Player):
    """Reads decisions from stdin."""

    def choose_kept_dice(self, state: GameState) -> list:
        print(f"\n{self.name} | Dice: {state.current_dice}  "
              f"(roll {state.rolls_used}/{state.max_rolls})")
        print("Which dice to KEEP? Enter 0-based indices separated by spaces "
              "(e.g. '0 2'), or press Enter to keep ALL:")
        raw = input("  > ").strip()
        kept = [False] * len(state.current_dice)
        if raw == "":
            kept = [True] * len(state.current_dice)
        else:
            for token in raw.split():
                try:
                    idx = int(token)
                    if 0 <= idx < len(state.current_dice):
                        kept[idx] = True
                except ValueError:
                    pass
        return kept

    def should_reroll(self, state: GameState) -> bool:
        print(f"Reroll? {state.rolls_used}/{state.max_rolls} rolls used. (y/n)")
        answer = input("  > ").strip().lower()
        return answer.startswith("y")


# ---------------------------------------------------------------------------
# Random CPU (baseline)
# ---------------------------------------------------------------------------

class RandomCPUPlayer(Player):
    """Makes uniformly random decisions. Used as baseline for RL evaluation."""

    def choose_kept_dice(self, state: GameState) -> list:
        return [random.choice([True, False]) for _ in state.current_dice]

    def should_reroll(self, state: GameState) -> bool:
        return random.choice([True, False])


# ---------------------------------------------------------------------------
# Rule-based CPU
# ---------------------------------------------------------------------------

class RuleBasedCPUPlayer(Player):
    """Simple heuristic player.

    TODO: Implement sensible heuristics, for example:
        - Always keep aces (1s).
        - If current hand is SCHOCK or JENNIE, stop rolling.
        - If current hand is SIMPLE and rolls remain, reroll the lowest die.
        - Keep STREET components (1, 2, 3) if a street is reachable.
    """

    def choose_kept_dice(self, state: GameState) -> list:
        # TODO: implement heuristic keep logic
        # Minimal stub: keep all 1s (aces), reroll everything else
        return [d == 1 for d in state.current_dice]

    def should_reroll(self, state: GameState) -> bool:
        # TODO: implement heuristic reroll decision
        # Minimal stub: reroll if any die is not kept and rolls remain
        return state.rolls_used < state.max_rolls

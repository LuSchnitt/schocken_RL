"""
Schocken game simulator: dice utilities, hand evaluation, game state, and engine.

Hand rankings (best to worst):
    SCHOCK_AUS  – three aces (1-1-1), ends the round immediately
    SCHOCK      – two aces + one other die; value = the other die (lower is better)
    JENNIE      – three of a kind (not aces); value = the repeated number
    STREET      – 1-2-3; value = 0 (only one possible street)
    SIMPLE      – everything else; value = three-digit number (dice sorted descending)
"""

import random
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_DICE = 3
MAX_ROLLS = 3
NUM_TOKENS = 13


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HandRank(Enum):
    SCHOCK_AUS = 7
    SCHOCK     = 6  # two aces + one other (lower other = better, but stored as raw value)
    JENNIE     = 5  # three of a kind
    STREET     = 4  # 1-2-3
    SIMPLE     = 1  # everything else


class GamePhase(Enum):
    LADEN    = auto()   # opening round, each player rolls once
    ROUND    = auto()   # normal round
    HALF_END = auto()   # one player accumulated all 13 tokens
    GAME_END = auto()   # both halves resolved


# ---------------------------------------------------------------------------
# Dice utilities
# ---------------------------------------------------------------------------

def roll(n: int) -> list:
    """Roll n dice and return a list of ints in [1, 6]."""
    return [random.randint(1, 6) for _ in range(n)]


def apply_sechsen_drehen(dice: list) -> list:
    """Apply the 'Sechsen drehen' rule to a list of three dice values.

    Rules:
        - Exactly two sixes  -> replace them with one ace (1), keep the third die.
        - Exactly three sixes -> replace all with two aces (1, 1), add a random third die.

    TODO: Verify edge cases against your house rules and implement.
    """
    # TODO: implement Sechsen-drehen
    return dice[:]


# ---------------------------------------------------------------------------
# Hand evaluation
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    dice: tuple          # sorted tuple of the three final dice values
    rank: HandRank
    value: int           # secondary sort key within the same rank
    rolls_used: int = 1

    def __repr__(self) -> str:
        return f"Hand(dice={self.dice}, rank={self.rank.name}, value={self.value})"


def evaluate(dice: list) -> Hand:
    """Evaluate a set of three dice and return a Hand.

    TODO: Implement hand recognition. Suggested order of checks:
        1. sorted_dice == (1, 1, 1)  -> SCHOCK_AUS,  value = 0
        2. two values are 1          -> SCHOCK,       value = the non-ace die
        3. all three equal           -> JENNIE,       value = the repeated number
        4. sorted_dice == (1, 2, 3)  -> STREET,       value = 0
        5. else                      -> SIMPLE,        value = int of digits sorted desc
                                        e.g. [6, 3, 5] -> 653
    """
    # TODO
    raise NotImplementedError("evaluate() not yet implemented — fill in the hand logic")


def compare(a: "Hand", b: "Hand") -> int:
    """Compare two hands.  Returns 1 if a wins, -1 if b wins, 0 if tied.

    TODO: Implement comparison logic:
        1. Higher HandRank.value wins.
        2. Within SCHOCK: lower .value wins (lower non-ace die is better).
        3. Within JENNIE / SIMPLE: higher .value wins.
        4. Fewer rolls_used wins on a tie.
        5. True ties are broken externally by player order.
    """
    # TODO
    raise NotImplementedError("compare() not yet implemented")


def best_hand(hands: dict) -> str:
    """Return the player name whose Hand is best.

    Args:
        hands: dict mapping player name -> Hand
    """
    # TODO: use compare() to find the winner
    raise NotImplementedError


def worst_hand(hands: dict) -> str:
    """Return the player name whose Hand is worst (receives tokens)."""
    # TODO: use compare() to find the loser
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Snapshot of the complete game state at any point in time."""

    players: list                        # ordered list of player names
    tokens: dict                         # player_name -> int (tokens accumulated)
    current_player_idx: int = 0
    current_dice: list = field(default_factory=list)   # current roll values
    kept_dice: list = field(default_factory=list)      # list[bool]: which dice are kept
    rolls_used: int = 0
    max_rolls: int = MAX_ROLLS           # set by the first player each round
    round_hands: dict = field(default_factory=dict)    # player_name -> Hand
    phase: GamePhase = GamePhase.LADEN
    half: int = 1                        # 1 or 2

    @property
    def current_player(self) -> str:
        return self.players[self.current_player_idx]

    def copy(self) -> "GameState":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------

class SchockenGame:
    """Main game engine.

    Usage:
        game = SchockenGame([HumanPlayer("Alice"), RandomCPUPlayer("Bob")])
        loser = game.run()
    """

    def __init__(self, players: list):
        """
        Args:
            players: list of Player instances (any mix of Human / CPU / RL agents).
                     Minimum 2 players required.
        """
        assert len(players) >= 2, "Schocken requires at least 2 players"
        self.players = players
        self.state = GameState(
            players=[p.name for p in players],
            tokens={p.name: 0 for p in players},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Play a complete game (two halves).

        Returns the name of the overall loser.

        TODO: Implement two-half structure:
            - Run half 1, record half_loser_1.
            - Reset tokens, run half 2, record half_loser_2.
            - If same player lost both -> that player is the overall loser.
            - If different players -> play a playoff (Stechen) to determine loser.
        """
        # TODO
        raise NotImplementedError("run() not yet implemented")

    # ------------------------------------------------------------------
    # Internal helpers (stubs to fill in)
    # ------------------------------------------------------------------

    def _run_half(self) -> str:
        """Play one half until one player holds all 13 tokens.

        Returns the name of that half's loser.

        TODO: Reset tokens, run Laden phase, then loop _run_round() until
              a player's token count reaches NUM_TOKENS.
        """
        self.state.tokens = {p.name: 0 for p in self.players}
        self.state.phase = GamePhase.LADEN
        # TODO
        raise NotImplementedError

    def _run_laden(self) -> None:
        """Opening round: every player rolls exactly once; worst hand gets tokens.

        TODO: Each player rolls once (no rerolls), evaluate hands, distribute tokens.
              The loser of Laden becomes the first player of the next normal round.
        """
        # TODO
        raise NotImplementedError

    def _run_round(self) -> None:
        """Play one full round: all players take a turn, then distribute tokens.

        TODO:
            - The current first player goes first; their roll count sets max_rolls.
            - If any player rolls SCHOCK_AUS, the round ends immediately.
            - Call _distribute_tokens() at the end.
            - Advance player order so the loser starts the next round.
        """
        self.state.round_hands = {}
        self.state.phase = GamePhase.ROUND
        # TODO
        raise NotImplementedError

    def _player_turn(self, player) -> "Hand":
        """Execute one player's turn (up to max_rolls rolls).

        Calls player.choose_kept_dice() and player.should_reroll() in a loop.
        Returns the final Hand.

        TODO: Implement the turn loop:
            1. Roll all dice.
            2. Ask player which dice to keep.
            3. Ask if they want to reroll (if rolls remain and not max_rolls reached).
            4. If first player: set state.max_rolls to their rolls_used when they stop.
            5. Apply apply_sechsen_drehen() after each roll.
            6. Return evaluate(final_dice).
        """
        # TODO
        raise NotImplementedError

    def _distribute_tokens(self, round_hands: dict) -> None:
        """Determine the round loser and add tokens to their count.

        TODO:
            - Find the worst_hand() in round_hands.
            - Determine how many tokens they receive based on the best hand:
                * SCHOCK_AUS -> all tokens in the pool (check rules for exact count)
                * SCHOCK X   -> X tokens
                * JENNIE     -> 3 tokens
                * STREET     -> 2 tokens
                * SIMPLE     -> 1 token
            - Update self.state.tokens[loser] += amount.
        """
        # TODO
        raise NotImplementedError

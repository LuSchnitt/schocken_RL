"""
Gymnasium-style environment wrapping the Schocken simulator for RL training.

Observation vector (flat float32 array):
    [dice_0..2 /6,  kept_0..2,  rolls_used/MAX_ROLLS,
     tokens_self/NUM_TOKENS,  tokens_others.../NUM_TOKENS,  half/2]

    Total length: 3 + 3 + 1 + 1 + (n_players-1) + 1 = n_players + 8

Action space: Discrete(9)
    Actions 0–7: 3-bit bitmask encoding which dice to KEEP before rerolling.
        bit 0 = die[0], bit 1 = die[1], bit 2 = die[2]
        e.g. action=5  (binary 101) -> keep die[0] and die[2], reroll die[1]
        All-ones action (7 = 111) keeps all dice but still counts as a reroll request.
    Action 8: end turn — accept the current hand without rerolling.

Note: The environment controls one RL agent; all other seats are filled with
RandomCPUPlayer instances so the agent can be trained in self-play or
against a CPU baseline.
"""

import numpy as np
from schocken.game import (
    SchockenGame, GameState, GamePhase,
    NUM_DICE, MAX_ROLLS, NUM_TOKENS, roll, evaluate,
)
from schocken.players import Player, RandomCPUPlayer

ACTION_END_TURN = 8
NUM_ACTIONS = 9  # 2^3 keep combos + end-turn


# ---------------------------------------------------------------------------
# RL agent wrapper — lets the Env drive a Player slot
# ---------------------------------------------------------------------------

class _RLPlayerProxy(Player):
    """Internal placeholder so SchockenGame can call player methods.

    The Env intercepts those calls and routes them through step().
    This proxy is never called during RL training; it exists only so that
    SchockenGame sees a valid Player object in its player list.
    """

    def choose_kept_dice(self, state: GameState) -> list:
        # Controlled externally by SchockenEnv.step()
        raise RuntimeError("RLPlayerProxy should not be called directly")

    def should_reroll(self, state: GameState) -> bool:
        raise RuntimeError("RLPlayerProxy should not be called directly")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SchockenEnv:
    """RL training environment for Schocken.

    Typical usage:
        env = SchockenEnv(n_players=3)
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
    """

    NUM_ACTIONS = NUM_ACTIONS

    def __init__(self, n_players: int = 3):
        assert n_players >= 2, "Need at least 2 players"
        self.n_players = n_players
        self.obs_dim = NUM_DICE + NUM_DICE + 1 + 1 + (n_players - 1) + 1

        self._rl_name = "RL_Agent"
        self._game: SchockenGame = None
        self._state: GameState = None
        self._prev_tokens: int = 0

    # ------------------------------------------------------------------
    # Core Gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns the initial observation.

        TODO: Create a fresh SchockenGame with one _RLPlayerProxy (the agent)
              and (n_players - 1) RandomCPUPlayer instances, then run the
              game up to the first point where the RL agent must act.
        """
        # TODO
        raise NotImplementedError("reset() not yet implemented")

    def step(self, action: int) -> tuple:
        """Apply the RL agent's action and advance the game.

        Args:
            action: int in [0, 8]
                    0–7  -> keep dice according to bitmask, then reroll
                    8    -> end turn (accept current hand)

        Returns:
            obs    (np.ndarray): next observation
            reward (float):      reward for this transition
            done   (bool):       True when the episode (game) is over
            info   (dict):       auxiliary information

        TODO: Implement the step logic:
            1. Decode action:
               - If action == 8: finalize the agent's hand, advance to next player.
               - Else: decode bitmask -> kept_dice, reroll un-kept dice,
                       update state.rolls_used.
            2. If rolls_used >= max_rolls, force end-turn.
            3. Advance through CPU players automatically (no RL input needed).
            4. Compute reward via _compute_reward().
            5. Check done: episode ends when the full game is over.
            6. Return (obs, reward, done, info).
        """
        # TODO
        raise NotImplementedError("step() not yet implemented")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: int) -> list:
        """Decode a bitmask action (0–7) into a kept_dice boolean list.

        Bit 0 -> die[0], bit 1 -> die[1], bit 2 -> die[2].
        Example: action=5 (0b101) -> [True, False, True]
        """
        return [(action >> i) & 1 == 1 for i in range(NUM_DICE)]

    def _encode_state(self) -> np.ndarray:
        """Encode the current GameState into a flat float32 observation vector."""
        s = self._state
        dice = np.array(s.current_dice if s.current_dice else [0]*NUM_DICE,
                        dtype=np.float32) / 6.0
        kept = np.array(s.kept_dice if s.kept_dice else [0]*NUM_DICE,
                        dtype=np.float32)
        rolls = np.array([s.rolls_used / MAX_ROLLS], dtype=np.float32)
        tokens_self = np.array(
            [s.tokens.get(self._rl_name, 0) / NUM_TOKENS], dtype=np.float32
        )
        other_names = [p for p in s.players if p != self._rl_name]
        tokens_others = np.array(
            [s.tokens.get(n, 0) / NUM_TOKENS for n in other_names],
            dtype=np.float32,
        )
        half = np.array([s.half / 2.0], dtype=np.float32)
        return np.concatenate([dice, kept, rolls, tokens_self, tokens_others, half])

    def _compute_reward(self, tokens_before: int, tokens_after: int, done: bool) -> float:
        """Compute the reward signal for the latest transition.

        TODO: Design your reward shaping. Some suggestions:
            - Penalise each token received: -(tokens_after - tokens_before)
            - Small bonus when an opponent receives tokens
            - Large negative reward for losing the game (holding all 13 tokens)
            - Sparse alternative: 0 during game, +1 win / -1 loss at episode end
        """
        # TODO
        return 0.0

    def render(self) -> None:
        """Print the current state to stdout (for debugging)."""
        if self._state is None:
            print("Environment not initialised — call reset() first.")
            return
        s = self._state
        print(f"Phase: {s.phase.name}  Half: {s.half}  "
              f"Roll: {s.rolls_used}/{s.max_rolls}")
        print(f"Dice:  {s.current_dice}  Kept: {s.kept_dice}")
        print(f"Tokens: {s.tokens}")

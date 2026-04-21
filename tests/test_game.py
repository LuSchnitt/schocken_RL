"""
Tests for the Schocken game simulator.

Run with: python -m pytest tests/test_game.py -v

All tests are currently marked xfail because the simulator functions are stubs.
Remove the @pytest.mark.xfail decorator once you implement each function.
"""

import pytest
from schocken.game import (
    roll, apply_sechsen_drehen,
    evaluate, compare, best_hand, worst_hand,
    Hand, HandRank, GameState, SchockenGame, GamePhase,
    NUM_DICE, MAX_ROLLS, NUM_TOKENS,
)
from schocken.players import RandomCPUPlayer


# ---------------------------------------------------------------------------
# Dice utilities
# ---------------------------------------------------------------------------

def test_roll_length():
    assert len(roll(3)) == 3

def test_roll_range():
    for _ in range(100):
        assert all(1 <= d <= 6 for d in roll(3))

def test_roll_zero_dice():
    assert roll(0) == []


@pytest.mark.xfail(reason="apply_sechsen_drehen not yet implemented")
def test_sechsen_drehen_two_sixes():
    result = apply_sechsen_drehen([6, 6, 3])
    assert result.count(1) == 1
    assert 3 in result
    assert len(result) == 3

@pytest.mark.xfail(reason="apply_sechsen_drehen not yet implemented")
def test_sechsen_drehen_three_sixes():
    result = apply_sechsen_drehen([6, 6, 6])
    assert result.count(1) == 2
    assert len(result) == 3

def test_sechsen_drehen_no_sixes():
    result = apply_sechsen_drehen([1, 2, 3])
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# Hand evaluation
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="evaluate() not yet implemented")
def test_evaluate_schock_aus():
    h = evaluate([1, 1, 1])
    assert h.rank == HandRank.SCHOCK_AUS

@pytest.mark.xfail(reason="evaluate() not yet implemented")
def test_evaluate_schock():
    h = evaluate([1, 1, 4])
    assert h.rank == HandRank.SCHOCK
    assert h.value == 4

@pytest.mark.xfail(reason="evaluate() not yet implemented")
def test_evaluate_jennie():
    h = evaluate([5, 5, 5])
    assert h.rank == HandRank.JENNIE
    assert h.value == 5

@pytest.mark.xfail(reason="evaluate() not yet implemented")
def test_evaluate_street():
    h = evaluate([3, 1, 2])
    assert h.rank == HandRank.STREET

@pytest.mark.xfail(reason="evaluate() not yet implemented")
def test_evaluate_simple():
    h = evaluate([6, 3, 5])
    assert h.rank == HandRank.SIMPLE
    assert h.value == 653  # digits sorted descending


# ---------------------------------------------------------------------------
# Hand comparison
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="compare() / evaluate() not yet implemented")
def test_schock_aus_beats_schock():
    a = evaluate([1, 1, 1])
    b = evaluate([1, 1, 6])
    assert compare(a, b) == 1

@pytest.mark.xfail(reason="compare() / evaluate() not yet implemented")
def test_schock_lower_beats_higher():
    # Schock 2 should beat Schock 6 (lower non-ace is better)
    a = evaluate([1, 1, 2])
    b = evaluate([1, 1, 6])
    assert compare(a, b) == 1

@pytest.mark.xfail(reason="compare() / evaluate() not yet implemented")
def test_jennie_beats_street():
    a = evaluate([3, 3, 3])
    b = evaluate([1, 2, 3])
    assert compare(a, b) == 1


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

def test_gamestate_current_player():
    state = GameState(
        players=["Alice", "Bob"],
        tokens={"Alice": 0, "Bob": 0},
        current_player_idx=1,
    )
    assert state.current_player == "Bob"

def test_gamestate_copy_is_independent():
    state = GameState(players=["A"], tokens={"A": 0})
    copy  = state.copy()
    copy.tokens["A"] = 5
    assert state.tokens["A"] == 0


# ---------------------------------------------------------------------------
# SchockenGame
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="SchockenGame.run() not yet implemented")
def test_game_returns_string_loser():
    players = [RandomCPUPlayer("Bot1"), RandomCPUPlayer("Bot2")]
    game    = SchockenGame(players)
    loser   = game.run()
    assert loser in ["Bot1", "Bot2"]

@pytest.mark.xfail(reason="SchockenGame.run() not yet implemented")
def test_game_three_players():
    players = [RandomCPUPlayer(f"Bot{i}") for i in range(3)]
    game    = SchockenGame(players)
    loser   = game.run()
    assert loser in [p.name for p in players]

def test_game_requires_two_players():
    with pytest.raises(AssertionError):
        SchockenGame([RandomCPUPlayer("Lonely")])

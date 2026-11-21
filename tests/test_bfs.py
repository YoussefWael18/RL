"""Tests for the BFS shortest-path baseline across all board configurations."""

import pytest
from smart_dice.baselines.bfs import bfs_shortest_path, verify_bfs_path
from smart_dice.envs.board_config import BOARD_CONFIG_30, BOARD_CONFIG_50, BOARD_CONFIG_100
from smart_dice.envs.smart_dice_env import SmartDiceEnv


@pytest.mark.parametrize("config", [BOARD_CONFIG_30, BOARD_CONFIG_50, BOARD_CONFIG_100])
def test_bfs_finds_valid_shortest_path(config):
    """Test that BFS finds a valid path for each required board configuration."""
    result = bfs_shortest_path(config)

    # Shortest path must have at least 1 move and at most n moves
    assert result.shortest_path_length > 0
    assert result.shortest_path_length <= config.n
    assert len(result.actions) == result.shortest_path_length
    assert len(result.states) == result.shortest_path_length + 1
    assert result.states[0] == 1
    assert result.states[-1] == config.n

    # All actions must be within [1, 6]
    for a in result.actions:
        assert 1 <= a <= 6

    # Verify execution inside an actual environment
    env = SmartDiceEnv(config=config)
    reached_goal, sim_reward, visited_states = verify_bfs_path(env, result.actions)
    assert reached_goal is True
    assert visited_states[-1] == config.n
    assert visited_states == result.states
    assert sim_reward == pytest.approx(result.total_reward)


def test_bfs_optimality_on_toy_board():
    """Verify that BFS finds the optimal minimum number of moves on a simple board."""
    # Board with N=10, ladder at 2 -> 9
    # From 1, roll 1 -> land on 2 -> climb to 9. From 9, roll 1 -> land on 10 (goal).
    # Minimal moves = 2.
    from smart_dice.envs.board_config import BoardConfig

    toy_cfg = BoardConfig(n=10, snakes={}, ladders={2: 9}, name="toy_board")
    res = bfs_shortest_path(toy_cfg)
    assert res.shortest_path_length == 2
    assert res.actions == [1, 1]
    assert res.states == [1, 9, 10]
    # Reward: step 1 (ladder +20), step 2 (goal +100) -> total 120
    assert res.total_reward == 120.0

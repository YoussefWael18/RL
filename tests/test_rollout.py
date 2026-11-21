"""Tests for random-policy rollout evaluation across all board configurations."""

import pytest
from smart_dice.scripts.random_rollout import evaluate_random_policy, run_single_episode
from smart_dice.envs.smart_dice_env import SmartDiceEnv
from smart_dice.envs.board_config import BOARD_CONFIG_30, BOARD_CONFIG_50, BOARD_CONFIG_100


@pytest.mark.parametrize("board_id", ["30", "50", "100"])
def test_random_rollout_runs_on_all_boards(board_id):
    """Test that random rollouts run on all three boards and produce sensible stats."""
    summary = evaluate_random_policy(board_id=board_id, episodes=10, seed=123, verbose=False)

    assert summary["episodes"] == 10
    assert 0.0 <= summary["success_rate_pct"] <= 100.0
    assert summary["avg_steps"] > 0
    assert "avg_reward" in summary
    assert "avg_invalid_moves" in summary
    assert "avg_ladders_hit" in summary
    assert "avg_snakes_hit" in summary


def test_single_episode_termination_or_truncation():
    """Verify that a single rollout respects step bounds and returns complete metrics."""
    env = SmartDiceEnv(config="30", max_steps=20)
    res = run_single_episode(env, seed=42)

    assert res["steps"] <= 20
    assert res["success"] in (0.0, 1.0)
    assert isinstance(res["reward"], float)
    assert res["invalid_moves"] >= 0
    assert res["ladders_hit"] >= 0
    assert res["snakes_hit"] >= 0

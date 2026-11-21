"""Tests for tabular RL training, convergence criterion, and evaluation."""

import numpy as np
import pytest

from smart_dice.envs.board_config import BoardConfig
from smart_dice.envs.smart_dice_env import SmartDiceEnv
from smart_dice.training.evaluator import determine_convergence, evaluate_agent
from smart_dice.training.tabular import train_q_learning, train_sarsa


@pytest.fixture
def mini_env():
    """Fast minimal environment for testing training loops."""
    cfg = BoardConfig(n=15, snakes={10: 4}, ladders={3: 11}, max_steps=40, name="mini")
    return SmartDiceEnv(config=cfg)


def test_convergence_criterion_logic():
    """Verify determine_convergence correctly detects convergence and non-convergence."""
    # 1. Non-converging run: low success rate and negative rewards
    rewards_bad = [-20.0] * 60
    success_bad = [0.0] * 60
    assert determine_convergence(rewards_bad, success_bad, window=50) is None

    # 2. Converging run: starts low, then 50 episodes of high success and high reward
    rewards_good = [-10.0] * 30 + [120.0] * 50
    success_good = [0.0] * 30 + [1.0] * 50
    conv_ep = determine_convergence(rewards_good, success_good, window=50)
    assert conv_ep is not None
    assert 75 <= conv_ep <= 80  # Window with >= 95% success (at least 48/50 successes) reached at episode 78


def test_q_learning_training(mini_env):
    """Test tabular Q-learning executes and returns valid structure."""
    res = train_q_learning(mini_env, num_episodes=50, seed=42)
    assert res["q_table"].shape == (16, 6)
    assert len(res["rewards_history"]) == 50
    assert len(res["moves_history"]) == 50
    assert len(res["success_history"]) == 50

    # Evaluate trained Q-table
    eval_res = evaluate_agent(res["q_table"], mini_env, num_episodes=10, seed=42)
    assert "avg_moves" in eval_res
    assert "success_rate" in eval_res
    assert "avg_reward" in eval_res
    assert eval_res["avg_moves"] > 0


def test_sarsa_training(mini_env):
    """Test tabular SARSA executes and returns valid structure."""
    res = train_sarsa(mini_env, num_episodes=50, seed=42)
    assert res["q_table"].shape == (16, 6)
    assert len(res["rewards_history"]) == 50

    eval_res = evaluate_agent(res["q_table"], mini_env, num_episodes=10, seed=42)
    assert eval_res["success_rate"] >= 0.0

"""Tests for tabular RL training, convergence criterion, and evaluation."""

import numpy as np
import pytest

from smart_dice.envs.board_config import BoardConfig
from smart_dice.envs.smart_dice_env import SmartDiceEnv
from smart_dice.training.evaluator import GreedyPolicyEvaluator, determine_convergence, evaluate_agent
from smart_dice.training.tabular import train_q_learning, train_sarsa


@pytest.fixture
def mini_env():
    """Fast minimal environment for testing training loops."""
    cfg = BoardConfig(n=15, snakes={10: 4}, ladders={3: 11}, max_steps=40, name="mini")
    return SmartDiceEnv(config=cfg)


def test_convergence_criterion_logic():
    """Return the start of the sustained streak in training episodes."""
    moves = [4] * 6 + [float("inf")] + [5, 3, 4, 4, 4, 4, 4]
    checkpoints = [(5 * episode, count) for episode, count in enumerate(moves, start=1)]
    assert determine_convergence(checkpoints[:-1], bfs_optimal_moves=3, window=3) is None
    assert determine_convergence(checkpoints, bfs_optimal_moves=3, window=3) == 60

    # A longer board uses the 10% tolerance instead of the one-move tolerance.
    checkpoints = [(5, 22), (10, 22)]
    assert determine_convergence(checkpoints, bfs_optimal_moves=20, window=2, rolling_window=1) == 5


def test_q_learning_training(mini_env):
    """Test tabular Q-learning executes and returns valid structure."""
    greedy = GreedyPolicyEvaluator(SmartDiceEnv(config=mini_env.config), 2, every_episodes=5)
    res = train_q_learning(mini_env, num_episodes=50, seed=42, greedy_evaluator=greedy)
    assert res["q_table"].shape == (16, 6)
    assert len(res["rewards_history"]) == 50
    assert len(res["moves_history"]) == 50
    assert len(res["success_history"]) == 50
    assert [episode for episode, _ in greedy.checkpoints] == list(range(5, 51, 5))

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

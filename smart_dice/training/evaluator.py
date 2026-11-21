"""Evaluation utilities for Smart Dice policies and models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym

from smart_dice.envs.smart_dice_env import SmartDiceEnv


def determine_convergence(
    episode_rewards: List[float],
    episode_successes: List[float],
    window: int = 50,
    min_success_rate: float = 0.95,
    min_avg_reward: float = 50.0,
    reward_stability_ratio: float = 0.90,
) -> Optional[int]:
    """Determine the convergence episode based on rolling window performance.

    Convergence Criterion:
        An agent is declared converged at episode E if, over the preceding `window`
        episodes (e.g., 50 episodes):
        1. Success Rate >= min_success_rate (at least 95% of games reach square N)
        2. Moving Average Return >= min_avg_reward (positive expected return)
        3. Moving Average Return >= reward_stability_ratio * max_achieved_return
           (performance is within 10% of the peak rolling average)

    Returns:
        The 1-based episode index of convergence, or None if criterion is not met.
    """
    if len(episode_rewards) < window:
        return None

    max_rolling_reward = -float("inf")
    for ep in range(window, len(episode_rewards) + 1):
        win_rewards = episode_rewards[ep - window : ep]
        win_success = episode_successes[ep - window : ep]

        avg_r = float(np.mean(win_rewards))
        avg_s = float(np.mean(win_success))

        if avg_r > max_rolling_reward:
            max_rolling_reward = avg_r

        if (
            avg_s >= min_success_rate
            and avg_r >= min_avg_reward
            and avg_r >= reward_stability_ratio * max_rolling_reward
        ):
            return ep

    return None


def extract_action_from_agent(
    agent_or_policy: Any,
    obs: int,
    deterministic: bool = True,
) -> int:
    """Extract an integer action in {0, 1, 2, 3, 4, 5} from various policy types."""
    # 1. Tabular Q-table: np.ndarray of shape (N + 1, 6)
    if isinstance(agent_or_policy, np.ndarray):
        q_row = agent_or_policy[obs]
        return int(np.argmax(q_row))

    # 2. SB3 Model with predict method
    if hasattr(agent_or_policy, "predict"):
        action, _ = agent_or_policy.predict(obs, deterministic=deterministic)
        if isinstance(action, (np.ndarray, list)):
            return int(action.item() if hasattr(action, "item") else action[0])
        return int(action)

    # 3. Callable policy function
    if callable(agent_or_policy):
        act = agent_or_policy(obs)
        return int(act)

    raise TypeError(f"Unsupported agent/policy type: {type(agent_or_policy)}")


def evaluate_agent(
    agent_or_policy: Any,
    env: SmartDiceEnv,
    num_episodes: int = 50,
    seed: int = 42,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Shared evaluation routine for all algorithms and models.

    Args:
        agent_or_policy: Tabular Q-table, SB3 model, or policy callable.
        env: The SmartDiceEnv instance to evaluate on.
        num_episodes: Number of evaluation episodes.
        seed: Random seed for evaluation reproducibility.
        deterministic: Whether to use deterministic action selection.

    Returns:
        Dictionary with:
            - avg_moves: mean moves per episode
            - success_rate: percentage of episodes that reached square N
            - avg_reward: mean cumulative reward
            - std_reward: standard deviation of cumulative reward
            - episode_rewards: list of returns per episode
            - episode_moves: list of move counts per episode
    """
    moves_list: List[int] = []
    rewards_list: List[float] = []
    success_list: List[float] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 101
        obs, _ = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_moves = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = extract_action_from_agent(agent_or_policy, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_moves += 1

        moves_list.append(ep_moves)
        rewards_list.append(ep_reward)
        success_list.append(100.0 if terminated else 0.0)

    return {
        "avg_moves": float(np.mean(moves_list)),
        "success_rate": float(np.mean(success_list)),
        "avg_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "episode_rewards": rewards_list,
        "episode_moves": moves_list,
        "episode_success": success_list,
    }

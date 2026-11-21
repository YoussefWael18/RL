"""Evaluation utilities for Smart Dice policies and models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from smart_dice.envs.smart_dice_env import SmartDiceEnv


def determine_convergence(
    checkpoints: Sequence[Tuple[int, float]],
    bfs_optimal_moves: int,
    window: int = 50,
    rolling_window: int = 5,
) -> Optional[int]:
    """Return the training episode at the start of the first sustained qualifying streak.

    Each checkpoint is a pair of training episode and greedy evaluation move count.
    A full rolling mean must be at most the BFS distance plus the larger of one
    move or ten percent for `window` consecutive checkpoints.
    """
    if window < 1 or rolling_window < 1:
        raise ValueError("Convergence and rolling windows must be positive")

    move_limit = bfs_optimal_moves + max(1, 0.10 * bfs_optimal_moves)
    streak = 0
    for index in range(rolling_window - 1, len(checkpoints)):
        recent = checkpoints[index - rolling_window + 1 : index + 1]
        mean_moves = sum(moves for _, moves in recent) / rolling_window
        streak = streak + 1 if mean_moves <= move_limit else 0
        if streak == window:
            return checkpoints[index - window + 1][0]
    return None


class GreedyPolicyEvaluator:
    """Record periodic greedy rollouts on a separate deterministic board."""

    def __init__(
        self, env: SmartDiceEnv, bfs_optimal_moves: int, every_episodes: int, seed: int = 42
    ) -> None:
        if every_episodes < 1:
            raise ValueError("Evaluation interval must be positive")
        self.env = env
        self.bfs_optimal_moves = bfs_optimal_moves
        self.every_episodes = every_episodes
        self.seed = seed
        self.checkpoints: List[Tuple[int, float]] = []

    def record(self, agent_or_policy: Any, episode: int) -> None:
        if episode % self.every_episodes == 0:
            result = evaluate_agent(agent_or_policy, self.env, num_episodes=1, seed=self.seed)
            moves = result["avg_moves"] if result["success_rate"] == 100.0 else float("inf")
            self.checkpoints.append((episode, moves))

    def get_convergence_episode(self, window: int) -> Optional[int]:
        return determine_convergence(self.checkpoints, self.bfs_optimal_moves, window=window)


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

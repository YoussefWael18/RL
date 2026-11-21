"""Tabular Reinforcement Learning algorithms: Q-Learning and SARSA."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from smart_dice.envs.smart_dice_env import SmartDiceEnv
from smart_dice.training.evaluator import GreedyPolicyEvaluator


def train_q_learning(
    env: SmartDiceEnv,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    *,
    greedy_evaluator: Optional[GreedyPolicyEvaluator] = None,
    convergence_window: int = 50,
) -> Dict[str, Any]:
    """Train a tabular Q-Learning agent with epsilon-greedy exploration.

    Update Rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    np.random.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Q-table: states 0..N, actions 0..5
    q_table = np.zeros((env.n + 1, 6), dtype=np.float64)
    epsilon = epsilon_start

    rewards_history: List[float] = []
    moves_history: List[int] = []
    success_history: List[float] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 7
        obs, _ = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_moves = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = int(env.action_space.sample())
            else:
                q_row = q_table[obs]
                max_val = np.max(q_row)
                best_actions = np.where(q_row == max_val)[0]
                action = int(np.random.choice(best_actions))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_moves += 1

            # TD target
            if terminated:
                td_target = reward
            elif truncated:
                td_target = reward
            else:
                td_target = reward + gamma * float(np.max(q_table[next_obs]))

            # Bellman update
            q_table[obs, action] += alpha * (td_target - q_table[obs, action])
            obs = next_obs

        rewards_history.append(ep_reward)
        moves_history.append(ep_moves)
        success_history.append(1.0 if terminated else 0.0)
        if greedy_evaluator is not None:
            greedy_evaluator.record(q_table, ep + 1)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    conv_ep = (
        greedy_evaluator.get_convergence_episode(window=convergence_window)
        if greedy_evaluator is not None else None
    )

    return {
        "q_table": q_table,
        "rewards_history": rewards_history,
        "moves_history": moves_history,
        "success_history": success_history,
        "convergence_episode": conv_ep,
        "final_epsilon": epsilon,
    }


def train_sarsa(
    env: SmartDiceEnv,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    *,
    greedy_evaluator: Optional[GreedyPolicyEvaluator] = None,
    convergence_window: int = 50,
) -> Dict[str, Any]:
    """Train a tabular SARSA agent with on-policy epsilon-greedy exploration.

    Update Rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    """
    np.random.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    q_table = np.zeros((env.n + 1, 6), dtype=np.float64)
    epsilon = epsilon_start

    rewards_history: List[float] = []
    moves_history: List[int] = []
    success_history: List[float] = []

    def choose_action(state: int, current_eps: float) -> int:
        if np.random.rand() < current_eps:
            return int(env.action_space.sample())
        q_row = q_table[state]
        max_val = np.max(q_row)
        best_actions = np.where(q_row == max_val)[0]
        return int(np.random.choice(best_actions))

    for ep in range(num_episodes):
        ep_seed = seed + ep * 7
        obs, _ = env.reset(seed=ep_seed)
        action = choose_action(obs, epsilon)

        ep_reward = 0.0
        ep_moves = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_moves += 1

            if terminated or truncated:
                td_target = reward
                q_table[obs, action] += alpha * (td_target - q_table[obs, action])
                break

            # On-policy next action choice
            next_action = choose_action(next_obs, epsilon)
            td_target = reward + gamma * q_table[next_obs, next_action]
            q_table[obs, action] += alpha * (td_target - q_table[obs, action])

            obs = next_obs
            action = next_action

        rewards_history.append(ep_reward)
        moves_history.append(ep_moves)
        success_history.append(1.0 if terminated else 0.0)
        if greedy_evaluator is not None:
            greedy_evaluator.record(q_table, ep + 1)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    conv_ep = (
        greedy_evaluator.get_convergence_episode(window=convergence_window)
        if greedy_evaluator is not None else None
    )

    return {
        "q_table": q_table,
        "rewards_history": rewards_history,
        "moves_history": moves_history,
        "success_history": success_history,
        "convergence_episode": conv_ep,
        "final_epsilon": epsilon,
    }

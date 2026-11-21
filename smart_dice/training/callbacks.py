"""Stable-Baselines3 training metrics callback."""

from __future__ import annotations

from typing import List, Optional
from stable_baselines3.common.callbacks import BaseCallback

from smart_dice.training.evaluator import GreedyPolicyEvaluator


class EpisodeMetricsCallback(BaseCallback):
    """Callback for tracking episode rewards, moves, and successes during SB3 training."""

    def __init__(
        self, verbose: int = 0, *, greedy_evaluator: Optional[GreedyPolicyEvaluator] = None
    ) -> None:
        super().__init__(verbose)
        self.greedy_evaluator = greedy_evaluator
        self.episode_rewards: List[float] = []
        self.episode_moves: List[int] = []
        self.episode_success: List[float] = []

        self._curr_reward: float = 0.0
        self._curr_moves: int = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self._curr_reward += float(rewards[0])
            self._curr_moves += 1

        if dones is not None and dones[0]:
            self.episode_rewards.append(self._curr_reward)
            self.episode_moves.append(self._curr_moves)

            # Smart Dice only terminates upon reaching the goal.
            info = self.locals["infos"][0]
            self.episode_success.append(0.0 if info.get("TimeLimit.truncated", False) else 1.0)
            if self.greedy_evaluator is not None:
                self.greedy_evaluator.record(self.model, len(self.episode_moves))

            self._curr_reward = 0.0
            self._curr_moves = 0

        return True

    def get_convergence_episode(self, window: int = 50) -> Optional[int]:
        if self.greedy_evaluator is None:
            return None
        return self.greedy_evaluator.get_convergence_episode(window=window)

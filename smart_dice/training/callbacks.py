"""Stable-Baselines3 training metrics callback."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from smart_dice.training.evaluator import determine_convergence


class EpisodeMetricsCallback(BaseCallback):
    """Callback for tracking episode rewards, moves, and successes during SB3 training."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
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

            # Check termination vs truncation from info
            infos = self.locals.get("infos", [{}])
            info = infos[0] if len(infos) > 0 else {}
            # Success is defined as landing on final square N
            final_sq = info.get("final_square", 0)
            target_sq = info.get("target_square", 0)
            is_goal = bool(final_sq == getattr(self.training_env.envs[0], "n", -1) or target_sq == getattr(self.training_env.envs[0], "n", -1))
            self.episode_success.append(1.0 if is_goal else 0.0)

            self._curr_reward = 0.0
            self._curr_moves = 0

        return True

    def get_convergence_episode(self, window: int = 50) -> Optional[int]:
        return determine_convergence(self.episode_rewards, self.episode_success, window=window)

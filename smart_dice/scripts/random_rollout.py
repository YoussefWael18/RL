"""Random-policy rollout evaluation script for Smart Dice Board Game."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional
import numpy as np

from smart_dice.envs.board_config import BOARD_CONFIGS, get_board_config
from smart_dice.envs.smart_dice_env import SmartDiceEnv


def run_single_episode(
    env: SmartDiceEnv,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Execute one rollout episode using a uniform random action policy."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    invalid_moves = 0
    ladders_hit = 0
    snakes_hit = 0

    step_limit = max_steps if max_steps is not None else env.max_steps

    while not (terminated or truncated) and steps < step_limit:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if info.get("invalid_move", False):
            invalid_moves += 1
        if info.get("ladder_applied", False):
            ladders_hit += 1
        if info.get("snake_applied", False):
            snakes_hit += 1

    return {
        "success": 1.0 if terminated else 0.0,
        "steps": float(steps),
        "reward": float(total_reward),
        "invalid_moves": float(invalid_moves),
        "ladders_hit": float(ladders_hit),
        "snakes_hit": float(snakes_hit),
    }


def evaluate_random_policy(
    board_id: str,
    episodes: int = 50,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate uniform random rollouts on a designated board configuration."""
    config = get_board_config(board_id)
    env = SmartDiceEnv(config=config)

    if seed is not None:
        np.random.seed(seed)
        env.action_space.seed(seed)

    episode_stats: List[Dict[str, float]] = []
    for ep in range(episodes):
        ep_seed = (seed + ep * 17) if seed is not None else None
        res = run_single_episode(env, seed=ep_seed)
        episode_stats.append(res)

    success_rate = float(np.mean([s["success"] for s in episode_stats]) * 100.0)
    avg_steps = float(np.mean([s["steps"] for s in episode_stats]))
    avg_reward = float(np.mean([s["reward"] for s in episode_stats]))
    avg_invalids = float(np.mean([s["invalid_moves"] for s in episode_stats]))
    avg_ladders = float(np.mean([s["ladders_hit"] for s in episode_stats]))
    avg_snakes = float(np.mean([s["snakes_hit"] for s in episode_stats]))

    summary = {
        "board_name": config.name,
        "n": float(config.n),
        "episodes": float(episodes),
        "success_rate_pct": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_invalid_moves": avg_invalids,
        "avg_ladders_hit": avg_ladders,
        "avg_snakes_hit": avg_snakes,
    }

    if verbose:
        print(f"\n{'='*65}")
        print(f"Random Policy Evaluation: Board {config.name} (N={config.n})")
        print(f"Snakes: {len(config.snakes)} | Ladders: {len(config.ladders)} | Max Steps: {config.max_steps}")
        print(f"Episodes: {episodes} | Base Seed: {seed}")
        print(f"{'-'*65}")
        print(f"  Success Rate:          {success_rate:6.1f} %")
        print(f"  Avg Moves per Episode: {avg_steps:6.2f}")
        print(f"  Avg Cumulative Reward: {avg_reward:6.2f}")
        print(f"  Avg Invalid Moves:     {avg_invalids:6.2f}")
        print(f"  Avg Ladders Hit:       {avg_ladders:6.2f}")
        print(f"  Avg Snakes Hit:        {avg_snakes:6.2f}")
        print(f"{'='*65}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random policy on Smart Dice boards.")
    parser.add_argument(
        "--board",
        type=str,
        default="all",
        help="Board configuration to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per board (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    target_boards = ["30", "50", "100"] if args.board == "all" else [args.board]

    print("=== Smart Dice Board Game: Random Policy Rollout Evaluation ===")
    for b in target_boards:
        evaluate_random_policy(board_id=b, episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()

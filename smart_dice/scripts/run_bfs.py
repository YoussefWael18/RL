"""BFS Shortest-Path Baseline runner for Smart Dice Board Game."""

from __future__ import annotations

import argparse
from smart_dice.baselines.bfs import bfs_shortest_path, verify_bfs_path
from smart_dice.envs.board_config import get_board_config
from smart_dice.envs.smart_dice_env import SmartDiceEnv


def run_bfs_for_board(board_id: str, verbose: bool = True) -> None:
    config = get_board_config(board_id)
    result = bfs_shortest_path(config)

    # Verify through environment simulation
    env = SmartDiceEnv(config=config)
    reached_goal, actual_reward, states = verify_bfs_path(env, result.actions)

    if not reached_goal or states[-1] != config.n:
        raise RuntimeError(
            f"Verification failed on board {config.name}: reached={reached_goal}, final state={states[-1]}"
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"BFS Shortest Path Baseline: Board {config.name} (N={config.n})")
        print(f"Snakes: {len(config.snakes)} | Ladders: {len(config.ladders)}")
        print(f"{'-'*70}")
        print(f"  Shortest Path Length (Dice Rolls): {result.shortest_path_length}")
        print(f"  Cumulative Path Reward:            {result.total_reward:.1f}")
        print(f"  States Visited Count in BFS:       {result.visited_states_count}")
        print(f"  Action Sequence:                   {result.actions}")
        print(f"  State Trajectory:                  {result.states}")
        print(f"\n  Step-by-step Execution:")
        for t in result.transitions:
            type_tag = f"[{t.transition_type.upper()}]" if t.transition_type != "normal" else ""
            if t.transition_type == "ladder":
                details = f"square {t.target_square} -> climbed to {t.to_state}"
            elif t.transition_type == "snake":
                details = f"square {t.target_square} -> slid to {t.to_state}"
            elif t.transition_type == "goal":
                details = f"reached GOAL square {t.to_state}"
            else:
                details = f"square {t.to_state}"
            print(
                f"    Step {t.step:2d}: Pos {t.from_state:3d} + Roll {t.action} -> {details:38s} | Reward: {t.reward:+6.1f} {type_tag}"
            )
        print(f"  Environment Verification: SUCCESS (reached square {states[-1]}, reward={actual_reward:.1f})")
        print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFS shortest-path baseline on Smart Dice boards.")
    parser.add_argument(
        "--board",
        type=str,
        default="all",
        choices=["all", "30", "50", "100", "board_30", "board_50", "board_100"],
        help="Board configuration to solve with BFS (default: all)",
    )
    args = parser.parse_args()

    target_boards = ["30", "50", "100"] if args.board == "all" else [args.board]

    print("=== Smart Dice Board Game: BFS Shortest Path Baseline ===")
    for b in target_boards:
        run_bfs_for_board(board_id=b)


if __name__ == "__main__":
    main()

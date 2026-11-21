"""Smart Dice Board Game Gymnasium Environment and Baselines."""

from smart_dice.envs.board_config import (
    BoardConfig,
    BOARD_CONFIG_30,
    BOARD_CONFIG_50,
    BOARD_CONFIG_100,
    BOARD_CONFIG_100_ASSIGNMENT,
    BOARD_CONFIGS,
    detect_positive_reward_cycle,
    get_board_config,
)
from smart_dice.envs.smart_dice_env import SmartDiceEnv
from smart_dice.baselines.bfs import BFSResult, bfs_shortest_path, solve_bfs
from smart_dice.training import (
    determine_convergence,
    evaluate_agent,
    extract_action_from_agent,
    train_q_learning,
    train_sarsa,
    EpisodeMetricsCallback,
)

__all__ = [
    "BoardConfig",
    "SmartDiceEnv",
    "BOARD_CONFIG_30",
    "BOARD_CONFIG_50",
    "BOARD_CONFIG_100",
    "BOARD_CONFIG_100_ASSIGNMENT",
    "BOARD_CONFIGS",
    "detect_positive_reward_cycle",
    "get_board_config",
    "BFSResult",
    "bfs_shortest_path",
    "solve_bfs",
    "determine_convergence",
    "evaluate_agent",
    "extract_action_from_agent",
    "train_q_learning",
    "train_sarsa",
    "EpisodeMetricsCallback",
]

"""Environment package for Smart Dice Board Game."""

from smart_dice.envs.board_config import (
    BoardConfig,
    BOARD_CONFIG_30,
    BOARD_CONFIG_50,
    BOARD_CONFIG_100,
    BOARD_CONFIGS,
    get_board_config,
)
from smart_dice.envs.smart_dice_env import (
    SmartDiceEnv,
    DiscreteActionSpace,
    DiscreteObservationSpace,
)

__all__ = [
    "BoardConfig",
    "BOARD_CONFIG_30",
    "BOARD_CONFIG_50",
    "BOARD_CONFIG_100",
    "BOARD_CONFIGS",
    "get_board_config",
    "SmartDiceEnv",
    "DiscreteActionSpace",
    "DiscreteObservationSpace",
]

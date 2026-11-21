"""Board configurations for the Smart Dice Board Game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class BoardConfig:
    """Configuration class for the Smart Dice Board Game.

    Attributes:
        n: Total number of squares on the board (from 1 to n).
        snakes: Mapping of snake start square to snake tail square (start > tail).
        ladders: Mapping of ladder base square to ladder top square (base < top).
        max_steps: Maximum number of allowed steps in an episode before truncation.
        name: A human-readable identifier for this configuration.
    """

    n: int
    snakes: Dict[int, int] = field(default_factory=dict)
    ladders: Dict[int, int] = field(default_factory=dict)
    max_steps: int = 1000
    name: str = "custom"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate board parameters."""
        if not isinstance(self.n, int) or self.n < 2:
            raise ValueError(f"Board size n must be an integer >= 2, got {self.n}")

        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {self.max_steps}")

        # Validate ladders
        for start, end in self.ladders.items():
            if not (1 < start < end <= self.n):
                raise ValueError(
                    f"Invalid ladder ({start} -> {end}): must satisfy 1 < start < end <= {self.n}"
                )

        # Validate snakes
        for start, end in self.snakes.items():
            if not (1 <= end < start < self.n):
                raise ValueError(
                    f"Invalid snake ({start} -> {end}): must satisfy 1 <= end < start < {self.n}"
                )

        # Ensure disjoint start squares for snakes and ladders
        overlap = set(self.snakes.keys()).intersection(set(self.ladders.keys()))
        if overlap:
            raise ValueError(f"Squares cannot be both snake and ladder starts: {overlap}")


# Configuration 1: N=30, 3 snakes, 3 ladders
BOARD_CONFIG_30 = BoardConfig(
    n=30,
    snakes={
        14: 4,
        21: 9,
        27: 7,
    },
    ladders={
        3: 12,
        8: 18,
        15: 24,
    },
    max_steps=500,
    name="board_30",
)

# Configuration 2: N=50, 5 snakes, 5 ladders
BOARD_CONFIG_50 = BoardConfig(
    n=50,
    snakes={
        17: 6,
        25: 10,
        34: 16,
        43: 22,
        48: 26,
    },
    ladders={
        4: 14,
        12: 28,
        20: 36,
        29: 42,
        37: 47,
    },
    max_steps=1000,
    name="board_50",
)

# Configuration 3: N=100, 8 snakes, 8 ladders
BOARD_CONFIG_100 = BoardConfig(
    n=100,
    snakes={
        31: 9,
        45: 16,
        58: 23,
        67: 38,
        78: 44,
        88: 52,
        93: 65,
        98: 40,
    },
    ladders={
        4: 25,
        13: 39,
        28: 55,
        41: 62,
        54: 76,
        63: 84,
        72: 91,
        80: 96,
    },
    max_steps=2000,
    name="board_100",
)

BOARD_CONFIGS: Dict[str, BoardConfig] = {
    "30": BOARD_CONFIG_30,
    "board_30": BOARD_CONFIG_30,
    "small": BOARD_CONFIG_30,
    "50": BOARD_CONFIG_50,
    "board_50": BOARD_CONFIG_50,
    "medium": BOARD_CONFIG_50,
    "100": BOARD_CONFIG_100,
    "board_100": BOARD_CONFIG_100,
    "large": BOARD_CONFIG_100,
}


def get_board_config(identifier: Union[int, str, BoardConfig]) -> BoardConfig:
    """Retrieve a BoardConfig instance from integer, name, or pass-through existing BoardConfig."""
    if isinstance(identifier, BoardConfig):
        return identifier
    key = str(identifier).lower()
    if key in BOARD_CONFIGS:
        return BOARD_CONFIGS[key]
    raise KeyError(
        f"Unknown board configuration '{identifier}'. Available presets: {list(BOARD_CONFIGS.keys())}"
    )

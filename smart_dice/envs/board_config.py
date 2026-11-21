"""Board configurations for the Smart Dice Board Game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def detect_positive_reward_cycle(
    n: int,
    snakes: Dict[int, int],
    ladders: Dict[int, int],
) -> Tuple[bool, Optional[List[int]], float]:
    """Detect if the deterministic board graph contains a cycle with net positive reward.

    Uses the Bellman-Ford algorithm on the inverted-reward graph (weight = -reward).
    A negative weight cycle in this graph corresponds to a positive reward cycle.

    Returns:
        (has_cycle, cycle_nodes, net_cycle_reward)
    """
    nodes = list(range(1, n))
    edges: List[Tuple[int, int, float, float, int]] = []

    # Build directed transitions for actions 0..5 (dice values 1..6)
    for u in nodes:
        for d in range(1, 7):
            t = u + d
            if t > n:
                v = u
                r = -10.0  # REWARD_INVALID
            elif t == n:
                continue  # Reaching goal terminates the episode, cannot be part of a cycle
            elif t in ladders:
                dest = ladders[t]
                if dest == n:
                    continue  # Terminal ladder
                v = dest
                r = 20.0  # REWARD_LADDER
            elif t in snakes:
                dest = snakes[t]
                if dest == n:
                    continue  # Terminal snake
                v = dest
                r = -20.0  # REWARD_SNAKE
            else:
                v = t
                r = -1.0  # REWARD_NORMAL

            # weight = -reward; finding negative cycle is equivalent to finding positive reward cycle
            edges.append((u, v, -r, r, d))

    dist: Dict[int, float] = {u: 0.0 for u in nodes}
    pred: Dict[int, Optional[Tuple[int, float, int]]] = {u: None for u in nodes}

    # Relax edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w, r, d in edges:
            if dist[u] + w < dist[v] - 1e-9:
                dist[v] = dist[u] + w
                pred[v] = (u, r, d)
                updated = True
        if not updated:
            return False, None, 0.0

    # Check for negative weight cycle
    for u, v, w, r, d in edges:
        if dist[u] + w < dist[v] - 1e-9:
            # Trace back to isolate cycle nodes
            curr = v
            for _ in range(len(nodes)):
                assert pred[curr] is not None
                curr = pred[curr][0]

            cycle_nodes = [curr]
            cycle_reward = 0.0
            while True:
                step_info = pred[cycle_nodes[-1]]
                assert step_info is not None
                prev_node, r_step, _ = step_info
                cycle_reward += r_step
                cycle_nodes.append(prev_node)
                if prev_node == curr:
                    break
            cycle_nodes.reverse()
            return True, cycle_nodes, cycle_reward

    return False, None, 0.0


def _validate_int_coord(val: Any, role: str) -> int:
    """Validate that coordinate is strictly an integer (reject bools, floats, strings)."""
    if isinstance(val, bool):
        raise TypeError(f"{role} cannot be a boolean (got {val!r})")
    if not isinstance(val, (int, np.integer)):
        raise TypeError(
            f"{role} must be an integer, got {val!r} of type {type(val).__name__}"
        )
    return int(val)


@dataclass
class BoardConfig:
    """Configuration class for the Smart Dice Board Game.

    Attributes:
        n: Total number of squares on the board (from 1 to n).
        snakes: Mapping of snake start square to snake tail square (start > tail).
        ladders: Mapping of ladder base square to ladder top square (base < top).
        max_steps: Maximum number of allowed steps in an episode before truncation.
        name: A human-readable identifier for this configuration.
        allow_positive_reward_cycles: If False, raises ValueError if a positive reward cycle exists.
    """

    n: int
    snakes: Dict[int, int] = field(default_factory=dict)
    ladders: Dict[int, int] = field(default_factory=dict)
    max_steps: int = 1000
    name: str = "custom"
    allow_positive_reward_cycles: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate board parameters, types, coordinate rules, and cycle safety."""
        self.n = _validate_int_coord(self.n, "Board size n")
        if self.n < 2:
            raise ValueError(f"Board size n must be an integer >= 2, got {self.n}")

        self.max_steps = _validate_int_coord(self.max_steps, "max_steps")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {self.max_steps}")

        # Validate ladders (types and bounds)
        sanitized_ladders: Dict[int, int] = {}
        for start, end in self.ladders.items():
            start_int = _validate_int_coord(start, "Ladder start square")
            end_int = _validate_int_coord(end, "Ladder end square")
            if not (1 < start_int < end_int <= self.n):
                raise ValueError(
                    f"Invalid ladder ({start_int} -> {end_int}): must satisfy 1 < start < end <= {self.n}"
                )
            sanitized_ladders[start_int] = end_int

        # Validate snakes (types and bounds)
        sanitized_snakes: Dict[int, int] = {}
        for start, end in self.snakes.items():
            start_int = _validate_int_coord(start, "Snake start square")
            end_int = _validate_int_coord(end, "Snake end square")
            if not (1 <= end_int < start_int < self.n):
                raise ValueError(
                    f"Invalid snake ({start_int} -> {end_int}): must satisfy 1 <= end < start < {self.n}"
                )
            sanitized_snakes[start_int] = end_int

        # Ensure disjoint start squares for snakes and ladders
        overlap = set(sanitized_snakes.keys()).intersection(set(sanitized_ladders.keys()))
        if overlap:
            raise ValueError(f"Squares cannot be both snake and ladder starts: {overlap}")

        self.snakes = sanitized_snakes
        self.ladders = sanitized_ladders

        # Check for positive reward cycles
        if not self.allow_positive_reward_cycles:
            has_cycle, cycle_nodes, cycle_reward = detect_positive_reward_cycle(
                self.n, self.snakes, self.ladders
            )
            if has_cycle and cycle_nodes is not None:
                cycle_str = " -> ".join(map(str, cycle_nodes))
                raise ValueError(
                    f"Board '{self.name}' contains a positive reward cycle: "
                    f"{cycle_str} (net reward: {cycle_reward:+.1f}). "
                    f"Positive reward cycles must be prevented to avoid infinite learning loops."
                )


# Configuration 1: N=30, 3 snakes, 3 ladders (Loop-safe)
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

# Configuration 2: N=50, 5 snakes, 5 ladders (Loop-safe)
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

# Configuration 3: N=100, 8 snakes, 8 ladders (Loop-safe default)
BOARD_CONFIG_100 = BoardConfig(
    n=100,
    snakes={
        31: 9,
        45: 16,
        58: 32,
        67: 43,
        77: 65,
        86: 66,
        93: 74,
        97: 85,
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

# Documented assignment board containing the +40 exploit cycle:
# 84 -> 52 -> 76 -> 96 -> 40 -> 62 -> 84 (worth +40 every 6 moves)
BOARD_CONFIG_100_ASSIGNMENT = BoardConfig(
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
    name="board100_assignment",
    allow_positive_reward_cycles=True,
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
    "board100_assignment": BOARD_CONFIG_100_ASSIGNMENT,
    "board_100_assignment": BOARD_CONFIG_100_ASSIGNMENT,
    "exploit": BOARD_CONFIG_100_ASSIGNMENT,
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

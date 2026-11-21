"""Smart Dice Board Game Gymnasium Environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces

from smart_dice.envs.board_config import BoardConfig, BOARD_CONFIG_30, get_board_config


class DiscreteActionSpace(spaces.Discrete):
    """Discrete action space for dice values 1 through 6.

    Inherits from gymnasium.spaces.Discrete with n=6 and start=1, while
    supporting flexible equality with standard Discrete(6) spaces.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(n=6, start=1, seed=seed)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, spaces.Discrete) and other.n == 6:
            return True
        return super().__eq__(other)


class DiscreteObservationSpace(spaces.Discrete):
    """Discrete observation space representing board positions 1 through N.

    Inherits from gymnasium.spaces.Discrete with n=N and start=1, while
    supporting flexible equality with standard Discrete(N) spaces.
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        super().__init__(n=n, start=1, seed=seed)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, spaces.Discrete) and other.n == self.n:
            return True
        return super().__eq__(other)


class SmartDiceEnv(gym.Env):
    """Smart Dice Board Game Gymnasium Environment.

    A controllable board game where an agent chooses a dice roll (1 to 6)
    to navigate from square 1 to square N on a board containing snakes and ladders.

    State:
        Current square position on the board, an integer in {1, 2, ..., N}.

    Actions:
        Discrete choice of dice roll: {1, 2, 3, 4, 5, 6}.

    Reward Structure:
        - Reach the final square N: +100
        - Land on a ladder: +20 (agent teleports forward to ladder top)
        - Land on a normal square: -1
        - Land on a snake: -20 (agent teleports backward to snake tail)
        - Invalid move beyond board (target > N): -10 (agent stays in place, episode continues)

    Termination:
        - Episode terminates when reaching square N.
        - Episode truncates when step count reaches max_steps.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Reward constants
    REWARD_GOAL: float = 100.0
    REWARD_LADDER: float = 20.0
    REWARD_NORMAL: float = -1.0
    REWARD_SNAKE: float = -20.0
    REWARD_INVALID: float = -10.0

    def __init__(
        self,
        config: Optional[Union[BoardConfig, str, int]] = None,
        n: Optional[int] = None,
        snakes: Optional[Dict[int, int]] = None,
        ladders: Optional[Dict[int, int]] = None,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the SmartDiceEnv.

        Config can be supplied via a BoardConfig instance (or preset identifier),
        or directly via n, snakes, ladders, and max_steps keyword arguments.
        """
        super().__init__()

        if config is not None:
            resolved_config = get_board_config(config)
            self.n = n if n is not None else resolved_config.n
            self.snakes = dict(snakes) if snakes is not None else dict(resolved_config.snakes)
            self.ladders = dict(ladders) if ladders is not None else dict(resolved_config.ladders)
            self.max_steps = max_steps if max_steps is not None else resolved_config.max_steps
            self.config_name = resolved_config.name
        else:
            self.n = n if n is not None else BOARD_CONFIG_30.n
            self.snakes = dict(snakes) if snakes is not None else dict(BOARD_CONFIG_30.snakes)
            self.ladders = dict(ladders) if ladders is not None else dict(BOARD_CONFIG_30.ladders)
            self.max_steps = max_steps if max_steps is not None else BOARD_CONFIG_30.max_steps
            self.config_name = "custom"

        # Validate configuration
        self.config = BoardConfig(
            n=self.n,
            snakes=self.snakes,
            ladders=self.ladders,
            max_steps=self.max_steps,
            name=self.config_name,
        )

        self.render_mode = render_mode

        # Define action space: discrete choices 1..6
        self.action_space = DiscreteActionSpace()

        # Define observation space: board squares 1..N
        self.observation_space = DiscreteObservationSpace(self.n)

        # Environment state variables
        self.current_pos: int = 1
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment to starting square 1 with optional seed."""
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.current_pos = 1
        self._step_count = 0

        info: Dict[str, Any] = {
            "target_square": 1,
            "final_square": 1,
            "invalid_move": False,
            "ladder_applied": False,
            "snake_applied": False,
            "step_count": 0,
        }

        if self.render_mode == "human":
            self.render()

        return self.current_pos, info

    def step(self, action: Union[int, Any]) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute one dice roll action.

        Args:
            action: Selected dice roll value in {1, 2, 3, 4, 5, 6}.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Validate action
        action_int = int(action)
        if action_int < 1 or action_int > 6:
            raise ValueError(
                f"Invalid action {action}. Expected dice value in {{1, 2, 3, 4, 5, 6}}."
            )

        self._step_count += 1
        target_square = self.current_pos + action_int

        invalid_move = False
        ladder_applied = False
        snake_applied = False
        terminated = False

        if target_square > self.n:
            # Rule: moves beyond N are invalid: state unchanged, reward = -10, does not terminate
            final_square = self.current_pos
            reward = self.REWARD_INVALID
            invalid_move = True
        elif target_square == self.n:
            # Reached goal
            final_square = self.n
            reward = self.REWARD_GOAL
            terminated = True
        elif target_square in self.ladders:
            # Landed on ladder
            final_square = self.ladders[target_square]
            reward = self.REWARD_LADDER
            ladder_applied = True
            if final_square == self.n:
                terminated = True
        elif target_square in self.snakes:
            # Landed on snake
            final_square = self.snakes[target_square]
            reward = self.REWARD_SNAKE
            snake_applied = True
        else:
            # Normal square
            final_square = target_square
            reward = self.REWARD_NORMAL

        self.current_pos = final_square
        truncated = (self._step_count >= self.max_steps) and not terminated

        info: Dict[str, Any] = {
            "target_square": target_square,
            "final_square": final_square,
            "invalid_move": invalid_move,
            "ladder_applied": ladder_applied,
            "snake_applied": snake_applied,
            "step_count": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return self.current_pos, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current board state in ASCII format."""
        lines = []
        lines.append(f"Board (N={self.n}, Step={self._step_count}/{self.max_steps}):")
        status = f"Current Position: [{self.current_pos}/{self.n}]"
        if self.current_pos == self.n:
            status += " (GOAL REACHED!)"
        lines.append(status)

        # Overview of active elements
        ladder_strs = [f"{s}->{e}" for s, e in sorted(self.ladders.items())]
        snake_strs = [f"{s}->{e}" for s, e in sorted(self.snakes.items())]
        lines.append(f"Ladders: {', '.join(ladder_strs)}")
        lines.append(f"Snakes:  {', '.join(snake_strs)}")

        board_str = "\n".join(lines)
        if self.render_mode == "human":
            print(board_str)
            return None
        return board_str

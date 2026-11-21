"""BFS Shortest-Path Baseline for the Deterministic Smart Dice Board Game."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from smart_dice.envs.board_config import BoardConfig, get_board_config
from smart_dice.envs.smart_dice_env import SmartDiceEnv


@dataclass
class TransitionStep:
    """Detailed information about a single transition along a path."""

    step: int
    from_state: int
    action: int
    target_square: int
    to_state: int
    reward: float
    transition_type: str  # 'normal', 'ladder', 'snake', 'goal', 'invalid'


@dataclass
class BFSResult:
    """Result of running BFS shortest path on a board."""

    board_name: str
    n: int
    shortest_path_length: int  # number of dice rolls
    states: List[int]  # sequence of states [1, s_1, ..., N]
    actions: List[int]  # sequence of actions [a_1, ..., a_k]
    transitions: List[TransitionStep] = field(default_factory=list)
    total_reward: float = 0.0
    optimal_policy: Dict[int, int] = field(default_factory=dict)
    visited_states_count: int = 0


def _simulate_transition(
    current_pos: int,
    action: int,
    n: int,
    snakes: Dict[int, int],
    ladders: Dict[int, int],
) -> Tuple[int, float, str, int]:
    """Simulate a deterministic board step.

    Returns:
        (next_pos, reward, transition_type, target_square)
    """
    target = current_pos + action
    if target > n:
        return current_pos, SmartDiceEnv.REWARD_INVALID, "invalid", target
    if target == n:
        return n, SmartDiceEnv.REWARD_GOAL, "goal", target
    if target in ladders:
        return ladders[target], SmartDiceEnv.REWARD_LADDER, "ladder", target
    if target in snakes:
        return snakes[target], SmartDiceEnv.REWARD_SNAKE, "snake", target
    return target, SmartDiceEnv.REWARD_NORMAL, "normal", target


def bfs_shortest_path(config: Union[BoardConfig, str, int]) -> BFSResult:
    """Compute the shortest path (minimum rolls) from square 1 to square N using BFS.

    Args:
        config: BoardConfig instance or preset identifier ('30', '50', '100').

    Returns:
        BFSResult with the shortest action sequence, visited states, transitions,
        and cumulative reward.
    """
    board_config = get_board_config(config)
    n = board_config.n
    snakes = board_config.snakes
    ladders = board_config.ladders

    # parent[s] = (predecessor_state, action_taken, reward, trans_type, target_square)
    parent: Dict[int, Optional[Tuple[int, int, float, str, int]]] = {1: None}
    dist: Dict[int, int] = {1: 0}
    queue: deque[int] = deque([1])

    while queue:
        curr = queue.popleft()
        if curr == n:
            break

        # Actions are 1 through 6
        for action in range(1, 7):
            next_state, reward, trans_type, target = _simulate_transition(
                curr, action, n, snakes, ladders
            )

            # Ignore self-loops from invalid moves beyond board
            if next_state == curr and trans_type == "invalid":
                continue

            if next_state not in parent:
                parent[next_state] = (curr, action, reward, trans_type, target)
                dist[next_state] = dist[curr] + 1
                queue.append(next_state)

    if n not in parent:
        raise RuntimeError(f"Goal square {n} is unreachable from square 1 on board '{board_config.name}'!")

    # Reconstruct shortest path from goal back to start
    rev_transitions: List[TransitionStep] = []
    curr = n
    step_num = dist[n]

    while curr != 1:
        prev_info = parent[curr]
        assert prev_info is not None
        p_state, action, reward, trans_type, target = prev_info
        rev_transitions.append(
            TransitionStep(
                step=step_num,
                from_state=p_state,
                action=action,
                target_square=target,
                to_state=curr,
                reward=reward,
                transition_type=trans_type,
            )
        )
        curr = p_state
        step_num -= 1

    transitions = list(reversed(rev_transitions))
    states = [1] + [t.to_state for t in transitions]
    actions = [t.action for t in transitions]
    total_reward = sum(t.reward for t in transitions)

    # Build optimal policy mapping for states on the shortest path
    optimal_policy = {t.from_state: t.action for t in transitions}

    return BFSResult(
        board_name=board_config.name,
        n=n,
        shortest_path_length=len(actions),
        states=states,
        actions=actions,
        transitions=transitions,
        total_reward=total_reward,
        optimal_policy=optimal_policy,
        visited_states_count=len(parent),
    )


def solve_bfs(config: Union[BoardConfig, str, int]) -> BFSResult:
    """Alias for bfs_shortest_path."""
    return bfs_shortest_path(config)


def verify_bfs_path(env: SmartDiceEnv, actions: List[int]) -> Tuple[bool, float, List[int]]:
    """Execute the BFS action path in the environment to verify correctness.

    Args:
        env: The SmartDiceEnv instance.
        actions: Sequence of dice actions from BFS.

    Returns:
        Tuple of (reached_goal, cumulative_reward, states_visited).
    """
    obs, _ = env.reset()
    states = [obs]
    cumulative_reward = 0.0

    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward
        states.append(obs)
        if terminated:
            return True, cumulative_reward, states
        if truncated:
            return False, cumulative_reward, states

    return obs == env.n, cumulative_reward, states

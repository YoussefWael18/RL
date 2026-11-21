"""Tests for SmartDiceEnv environment dynamics, rewards, and compliance."""

import pytest
import gymnasium as gym

from smart_dice.envs.board_config import BoardConfig, BOARD_CONFIG_30
from smart_dice.envs.smart_dice_env import SmartDiceEnv


@pytest.fixture
def simple_env():
    """Create a minimal deterministic environment for precise step testing."""
    config = BoardConfig(
        n=20,
        snakes={12: 5},
        ladders={4: 15},
        max_steps=50,
        name="test_board",
    )
    return SmartDiceEnv(config=config)


def test_initial_state(simple_env):
    """Test start position is 1, step count is 0, and reset info matches requirements."""
    obs, info = simple_env.reset()
    assert obs == 1
    assert simple_env.current_pos == 1
    assert simple_env._step_count == 0
    assert info["target_square"] == 1
    assert info["final_square"] == 1
    assert info["invalid_move"] is False
    assert info["ladder_applied"] is False
    assert info["snake_applied"] is False
    assert info["step_count"] == 0


def test_normal_movement(simple_env):
    """Test moving to a normal square gives -1 reward and advances position."""
    simple_env.reset()
    # From 1, roll 1 -> target 2 (normal square)
    obs, reward, terminated, truncated, info = simple_env.step(1)
    assert obs == 2
    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 2
    assert info["final_square"] == 2
    assert info["invalid_move"] is False
    assert info["ladder_applied"] is False
    assert info["snake_applied"] is False
    assert info["step_count"] == 1

    # From 2, roll 4 -> target 6 (normal square)
    obs, reward, terminated, truncated, info = simple_env.step(4)
    assert obs == 6
    assert reward == -1.0
    assert terminated is False
    assert info["target_square"] == 6
    assert info["final_square"] == 6


def test_ladder_transition(simple_env):
    """Test landing on a ladder moves agent forward and rewards +20."""
    simple_env.reset()
    # From 1, roll 3 -> target 4 (ladder base 4 -> 15)
    obs, reward, terminated, truncated, info = simple_env.step(3)
    assert obs == 15
    assert reward == 20.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 4
    assert info["final_square"] == 15
    assert info["ladder_applied"] is True
    assert info["snake_applied"] is False
    assert info["invalid_move"] is False


def test_snake_transition(simple_env):
    """Test landing on a snake moves agent backward and rewards -20."""
    simple_env.reset()
    # We want to land on square 12 (snake 12 -> 5).
    # Step 1: from 1 roll 5 -> square 6 (normal, reward -1)
    obs1, _, _, _, _ = simple_env.step(5)
    assert obs1 == 6

    # Step 2: from 6 roll 6 -> target 12 (snake 12 -> 5)
    obs2, reward, terminated, truncated, info = simple_env.step(6)
    assert obs2 == 5
    assert reward == -20.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 12
    assert info["final_square"] == 5
    assert info["snake_applied"] is True
    assert info["ladder_applied"] is False
    assert info["invalid_move"] is False


def test_invalid_move_beyond_n(simple_env):
    """Test moves beyond N keep state unchanged, reward -10, episode does not terminate."""
    simple_env.reset()
    # Teleport to 18 to test boundary (or take valid steps)
    simple_env.current_pos = 18

    # N=20. From 18, roll 3 -> target 21 > 20 (invalid)
    obs, reward, terminated, truncated, info = simple_env.step(3)
    assert obs == 18
    assert simple_env.current_pos == 18
    assert reward == -10.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 21
    assert info["final_square"] == 18
    assert info["invalid_move"] is True
    assert info["ladder_applied"] is False
    assert info["snake_applied"] is False

    # Second invalid move from 18: roll 6 -> target 24 > 20
    obs2, reward2, terminated2, truncated2, info2 = simple_env.step(6)
    assert obs2 == 18
    assert reward2 == -10.0
    assert terminated2 is False
    assert info2["invalid_move"] is True


def test_episode_termination_at_n(simple_env):
    """Test landing exactly on square N ends episode with +100 reward."""
    simple_env.reset()
    simple_env.current_pos = 18

    # N=20. From 18, roll 2 -> target 20 == N
    obs, reward, terminated, truncated, info = simple_env.step(2)
    assert obs == 20
    assert reward == 100.0
    assert terminated is True
    assert truncated is False
    assert info["target_square"] == 20
    assert info["final_square"] == 20
    assert info["invalid_move"] is False
    assert info["ladder_applied"] is False
    assert info["snake_applied"] is False


def test_seeded_reproducibility():
    """Test deterministic seeded resets and action-space sampling reproducibility."""
    env1 = SmartDiceEnv(config="30")
    env2 = SmartDiceEnv(config="30")

    # Fixed action sequence should produce identical results
    obs1, _ = env1.reset(seed=12345)
    obs2, _ = env2.reset(seed=12345)
    assert obs1 == obs2 == 1

    actions = [2, 6, 1, 4, 3, 5]
    for a in actions:
        out1 = env1.step(a)
        out2 = env2.step(a)
        assert out1 == out2

    # Action space sample reproducibility with seeded spaces
    env1.action_space.seed(999)
    env2.action_space.seed(999)
    samples1 = [env1.action_space.sample() for _ in range(20)]
    samples2 = [env2.action_space.sample() for _ in range(20)]
    assert samples1 == samples2


def test_max_steps_truncation():
    """Test episode truncates when reaching max_steps limit without reaching goal."""
    config = BoardConfig(n=30, snakes={}, ladders={}, max_steps=5, name="trunc_test")
    env = SmartDiceEnv(config=config)
    env.reset()

    # Take 4 steps: should not truncate
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(1)
        assert not terminated
        assert not truncated

    # 5th step hits max_steps
    _, _, terminated, truncated, info = env.step(1)
    assert not terminated
    assert truncated is True
    assert info["step_count"] == 5


def test_invalid_action_rejected(simple_env):
    """Test that actions outside {1, 2, 3, 4, 5, 6} raise ValueError."""
    simple_env.reset()
    with pytest.raises(ValueError):
        simple_env.step(0)
    with pytest.raises(ValueError):
        simple_env.step(7)
    with pytest.raises(ValueError):
        simple_env.step(-1)


def test_gymnasium_spaces():
    """Test action space and observation space properties."""
    env = SmartDiceEnv(config="30")
    # Action space
    assert env.action_space.n == 6
    for valid_a in [1, 2, 3, 4, 5, 6]:
        assert env.action_space.contains(valid_a)
    assert not env.action_space.contains(0)
    assert not env.action_space.contains(7)
    assert env.action_space == gym.spaces.Discrete(6)

    # Observation space
    assert env.observation_space.n == 30
    assert env.observation_space.contains(1)
    assert env.observation_space.contains(30)
    assert not env.observation_space.contains(0)
    assert not env.observation_space.contains(31)
    assert env.observation_space == gym.spaces.Discrete(30)

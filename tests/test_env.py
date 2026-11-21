"""Tests for SmartDiceEnv environment dynamics, rewards, and compliance."""

import pytest
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

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
    assert info["dice_value"] == 0
    assert info["invalid_move"] is False
    assert info["ladder_applied"] is False
    assert info["snake_applied"] is False
    assert info["step_count"] == 0
    assert simple_env.observation_space.contains(obs)


def test_observation_space_contains_after_reset_and_every_step(simple_env):
    """Test that after reset and every step, env.observation_space.contains(obs) is True."""
    obs, _ = simple_env.reset()
    assert simple_env.observation_space.contains(obs)

    # Perform a trajectory of varying actions
    for action in [0, 2, 4, 1, 3, 5, 0, 1]:
        obs, _, terminated, truncated, _ = simple_env.step(action)
        assert simple_env.observation_space.contains(obs), f"Observation {obs} not in observation_space!"
        assert 1 <= obs <= simple_env.n
        if terminated or truncated:
            break


def test_valid_actions_0_to_5():
    """Test action space is Discrete(6) and valid actions are integers 0..5."""
    env = SmartDiceEnv(config="30")
    assert env.action_space == gym.spaces.Discrete(6)
    assert env.action_space.n == 6

    for valid_a in range(6):
        assert env.action_space.contains(valid_a)

    assert not env.action_space.contains(-1)
    assert not env.action_space.contains(6)


def test_reject_invalid_action_types_and_ranges(simple_env):
    """Regression test: reject floats, strings, bools, negative values, and values above 5."""
    simple_env.reset()

    # Floats must be rejected
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(1.0)
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(0.0)
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(2.5)

    # Strings must be rejected
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step("0")
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step("1")
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step("invalid")

    # Bools must be rejected
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(True)
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(False)

    # Out-of-range integers
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(-1)
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(6)
    with pytest.raises(ValueError, match="Invalid action"):
        simple_env.step(100)

    # Valid integers and numpy integers must be accepted
    obs1, _, _, _, _ = simple_env.step(0)
    assert obs1 == 2
    obs2, _, _, _, _ = simple_env.step(np.int64(2))
    assert obs2 == 5
    obs3, _, _, _, _ = simple_env.step(np.int32(1))
    assert obs3 == 7


def test_action_dice_mapping(simple_env):
    """Test action 0 maps to dice 1 and action 5 maps to dice 6."""
    simple_env.reset()
    # From 1, action 0 -> dice 1 -> target 2
    obs, reward, _, _, info = simple_env.step(0)
    assert info["dice_value"] == 1
    assert info["target_square"] == 2
    assert obs == 2

    simple_env.reset()
    # From 1, action 5 -> dice 6 -> target 7
    obs, reward, _, _, info = simple_env.step(5)
    assert info["dice_value"] == 6
    assert info["target_square"] == 7
    assert obs == 7


def test_normal_movement(simple_env):
    """Test moving to a normal square gives -1 reward and advances position."""
    simple_env.reset()
    # From 1, action 0 (dice 1) -> target 2 (normal square)
    obs, reward, terminated, truncated, info = simple_env.step(0)
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
    assert simple_env.observation_space.contains(obs)

    # From 2, action 3 (dice 4) -> target 6 (normal square)
    obs, reward, terminated, truncated, info = simple_env.step(3)
    assert obs == 6
    assert reward == -1.0
    assert terminated is False
    assert info["target_square"] == 6
    assert info["final_square"] == 6
    assert simple_env.observation_space.contains(obs)


def test_ladder_transition(simple_env):
    """Test landing on a ladder moves agent forward and rewards +20."""
    simple_env.reset()
    # From 1, action 2 (dice 3) -> target 4 (ladder base 4 -> 15)
    obs, reward, terminated, truncated, info = simple_env.step(2)
    assert obs == 15
    assert reward == 20.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 4
    assert info["final_square"] == 15
    assert info["ladder_applied"] is True
    assert info["snake_applied"] is False
    assert info["invalid_move"] is False
    assert simple_env.observation_space.contains(obs)


def test_snake_transition(simple_env):
    """Test landing on a snake moves agent backward and rewards -20."""
    simple_env.reset()
    # Step 1: from 1, action 4 (dice 5) -> square 6 (normal, reward -1)
    obs1, _, _, _, _ = simple_env.step(4)
    assert obs1 == 6

    # Step 2: from 6, action 5 (dice 6) -> target 12 (snake 12 -> 5)
    obs2, reward, terminated, truncated, info = simple_env.step(5)
    assert obs2 == 5
    assert reward == -20.0
    assert terminated is False
    assert truncated is False
    assert info["target_square"] == 12
    assert info["final_square"] == 5
    assert info["snake_applied"] is True
    assert info["ladder_applied"] is False
    assert info["invalid_move"] is False
    assert simple_env.observation_space.contains(obs2)


def test_invalid_move_beyond_n(simple_env):
    """Test moves beyond N keep state unchanged, reward -10, episode does not terminate."""
    simple_env.reset()
    simple_env.current_pos = 18

    # N=20. From 18, action 2 (dice 3) -> target 21 > 20 (invalid)
    obs, reward, terminated, truncated, info = simple_env.step(2)
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
    assert simple_env.observation_space.contains(obs)


def test_episode_termination_at_n(simple_env):
    """Test landing exactly on square N ends episode with +100 reward."""
    simple_env.reset()
    simple_env.current_pos = 18

    # N=20. From 18, action 1 (dice 2) -> target 20 == N
    obs, reward, terminated, truncated, info = simple_env.step(1)
    assert obs == 20
    assert reward == 100.0
    assert terminated is True
    assert truncated is False
    assert info["target_square"] == 20
    assert info["final_square"] == 20
    assert simple_env.observation_space.contains(obs)


def test_ladder_landing_exactly_on_n_terminates():
    """Test that a ladder leading directly to N correctly terminates the episode."""
    config = BoardConfig(n=25, snakes={}, ladders={10: 25}, max_steps=100, name="ladder_to_goal")
    env = SmartDiceEnv(config=config)
    env.reset()
    env.current_pos = 6

    # From 6, action 3 (dice 4) -> lands on square 10 (ladder 10 -> 25)
    obs, reward, terminated, truncated, info = env.step(3)
    assert obs == 25
    assert info["target_square"] == 10
    assert info["final_square"] == 25
    assert info["ladder_applied"] is True
    assert terminated is True
    assert reward == 100.0
    assert env.observation_space.contains(obs)


def test_snake_landing_exactly_on_n_terminates():
    """Test that a snake transition ending on N correctly terminates the episode."""
    env = SmartDiceEnv(n=20, snakes={}, ladders={}, max_steps=100)
    # Inject a snake pointing to N on the environment to test termination logic
    env.snakes = {15: 20}
    env.reset()
    env.current_pos = 10

    # From 10, action 4 (dice 5) -> target 15 -> snake lands on 20 (N)
    obs, reward, terminated, truncated, info = env.step(4)
    assert obs == 20
    assert info["snake_applied"] is True
    assert terminated is True
    assert reward == 100.0
    assert env.observation_space.contains(obs)


def test_seeded_reproducibility():
    """Test deterministic seeded resets and action-space sampling reproducibility."""
    env1 = SmartDiceEnv(config="30")
    env2 = SmartDiceEnv(config="30")

    obs1, _ = env1.reset(seed=12345)
    obs2, _ = env2.reset(seed=12345)
    assert obs1 == obs2 == 1

    actions = [1, 5, 0, 3, 2, 4]
    for a in actions:
        out1 = env1.step(a)
        out2 = env2.step(a)
        assert out1 == out2

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

    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated

    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated is True
    assert info["step_count"] == 5


def test_sb3_smoke_test():
    """Verify that env passes SB3 env checker and a tiny PPO training run executes cleanly."""
    env = SmartDiceEnv(config="30")
    check_env(env)

    # Run a tiny 64-step SB3 PPO smoke test
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, verbose=0)
    model.learn(total_timesteps=64)
    assert model.num_timesteps >= 64

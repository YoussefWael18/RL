"""Tests for board configurations, cycle detection, coordinate validation, and presets."""

import pytest
import numpy as np

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


def test_board_config_30():
    """Verify requirements for Configuration 1: N=30, 3 snakes, 3 ladders."""
    cfg = BOARD_CONFIG_30
    assert cfg.n == 30
    assert len(cfg.snakes) == 3
    assert len(cfg.ladders) == 3

    for start, end in cfg.ladders.items():
        assert 1 < start < end <= 30

    for start, end in cfg.snakes.items():
        assert 1 <= end < start < 30

    assert set(cfg.snakes.keys()).isdisjoint(set(cfg.ladders.keys()))

    env = SmartDiceEnv(config=cfg)
    assert env.n == 30
    assert len(env.snakes) == 3
    assert len(env.ladders) == 3


def test_board_config_50():
    """Verify requirements for Configuration 2: N=50, 5 snakes, 5 ladders."""
    cfg = BOARD_CONFIG_50
    assert cfg.n == 50
    assert len(cfg.snakes) == 5
    assert len(cfg.ladders) == 5

    for start, end in cfg.ladders.items():
        assert 1 < start < end <= 50

    for start, end in cfg.snakes.items():
        assert 1 <= end < start < 50

    assert set(cfg.snakes.keys()).isdisjoint(set(cfg.ladders.keys()))

    env = SmartDiceEnv(config=cfg)
    assert env.n == 50
    assert len(env.snakes) == 5
    assert len(env.ladders) == 5


def test_board_config_100():
    """Verify requirements for Configuration 3: N=100, 8 snakes, 8 ladders."""
    cfg = BOARD_CONFIG_100
    assert cfg.n == 100
    assert len(cfg.snakes) == 8
    assert len(cfg.ladders) == 8

    for start, end in cfg.ladders.items():
        assert 1 < start < end <= 100

    for start, end in cfg.snakes.items():
        assert 1 <= end < start < 100

    assert set(cfg.snakes.keys()).isdisjoint(set(cfg.ladders.keys()))

    env = SmartDiceEnv(config=cfg)
    assert env.n == 100
    assert len(env.snakes) == 8
    assert len(env.ladders) == 8


def test_presets_are_loop_safe():
    """Verify that all default presets (30, 50, 100) are free from positive reward cycles."""
    for cfg in [BOARD_CONFIG_30, BOARD_CONFIG_50, BOARD_CONFIG_100]:
        has_cycle, cycle_nodes, reward = detect_positive_reward_cycle(cfg.n, cfg.snakes, cfg.ladders)
        assert not has_cycle, f"Preset '{cfg.name}' unexpectedly contains a positive cycle: {cycle_nodes}"


def test_assignment_exploit_positive_reward_cycle():
    """Regression test: detect the exact +40 positive reward cycle on the assignment board."""
    cfg = BOARD_CONFIG_100_ASSIGNMENT
    has_cycle, cycle_nodes, reward = detect_positive_reward_cycle(cfg.n, cfg.snakes, cfg.ladders)

    assert has_cycle is True
    assert cycle_nodes == [84, 52, 76, 96, 40, 62, 84]
    assert reward == pytest.approx(40.0)


def test_board_config_validation_rejects_positive_reward_cycle():
    """Test that creating a BoardConfig with a positive cycle raises ValueError by default."""
    with pytest.raises(ValueError, match="positive reward cycle"):
        BoardConfig(
            n=BOARD_CONFIG_100_ASSIGNMENT.n,
            snakes=BOARD_CONFIG_100_ASSIGNMENT.snakes,
            ladders=BOARD_CONFIG_100_ASSIGNMENT.ladders,
            name="unprotected_exploit_board",
            allow_positive_reward_cycles=False,
        )


def test_reject_non_integer_board_coordinates():
    """Regression test: require coordinates to be strictly integers, rejecting bools, floats, strings."""
    # Ladder float key / destination
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={}, ladders={3.0: 12})
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={}, ladders={3: 12.5})

    # Ladder string key / destination
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={}, ladders={"3": 12})
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={}, ladders={3: "12"})

    # Ladder bool key / destination
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=30, snakes={}, ladders={True: 12})
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=30, snakes={}, ladders={3: False})

    # Snake float key / destination
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={14.0: 4}, ladders={})
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={14: 4.0}, ladders={})

    # Snake string key / destination
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={"14": 4}, ladders={})
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={14: "4"}, ladders={})

    # Snake bool key / destination
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=30, snakes={True: 4}, ladders={})
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=30, snakes={14: False}, ladders={})

    # Non-integer n
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30.0, snakes={}, ladders={})
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=True, snakes={}, ladders={})
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n="30", snakes={}, ladders={})

    # Non-integer max_steps
    with pytest.raises(TypeError, match="must be an integer"):
        BoardConfig(n=30, snakes={}, ladders={}, max_steps=50.5)
    with pytest.raises(TypeError, match="cannot be a boolean"):
        BoardConfig(n=30, snakes={}, ladders={}, max_steps=False)


def test_get_board_config_helper():
    """Test retrieving board configurations by string and int keys."""
    assert get_board_config(30) is BOARD_CONFIG_30
    assert get_board_config("30") is BOARD_CONFIG_30
    assert get_board_config("board_30") is BOARD_CONFIG_30
    assert get_board_config(50) is BOARD_CONFIG_50
    assert get_board_config("50") is BOARD_CONFIG_50
    assert get_board_config(100) is BOARD_CONFIG_100
    assert get_board_config("100") is BOARD_CONFIG_100
    assert get_board_config(BOARD_CONFIG_30) is BOARD_CONFIG_30
    assert get_board_config("board100_assignment") is BOARD_CONFIG_100_ASSIGNMENT

    with pytest.raises(KeyError):
        get_board_config("invalid_board_id")


def test_config_validation_errors():
    """Test that out-of-bounds configurations raise appropriate ValueErrors."""
    # Invalid N (< 2)
    with pytest.raises(ValueError):
        BoardConfig(n=1, snakes={}, ladders={})

    # Ladder start >= end (backward ladder)
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={}, ladders={15: 10})

    # Snake start <= end (forward snake)
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={5: 15}, ladders={})

    # Snake start at goal N
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={30: 10}, ladders={})

    # Ladder start at square 1
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={}, ladders={1: 10})

    # Ladder end > N
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={}, ladders={25: 35})

    # Overlapping snake and ladder start
    with pytest.raises(ValueError):
        BoardConfig(n=30, snakes={15: 5}, ladders={15: 25})

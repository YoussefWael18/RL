"""Tests for the three required board configurations and config validation."""

import pytest

from smart_dice.envs.board_config import (
    BoardConfig,
    BOARD_CONFIG_30,
    BOARD_CONFIG_50,
    BOARD_CONFIG_100,
    BOARD_CONFIGS,
    get_board_config,
)
from smart_dice.envs.smart_dice_env import SmartDiceEnv


def test_board_config_30():
    """Verify requirements for Configuration 1: N=30, 3 snakes, 3 ladders."""
    cfg = BOARD_CONFIG_30
    assert cfg.n == 30
    assert len(cfg.snakes) == 3
    assert len(cfg.ladders) == 3

    # Verify all ladders go forward within bounds
    for start, end in cfg.ladders.items():
        assert 1 < start < end <= 30

    # Verify all snakes go backward within bounds
    for start, end in cfg.snakes.items():
        assert 1 <= end < start < 30

    # Verify start squares are disjoint
    assert set(cfg.snakes.keys()).isdisjoint(set(cfg.ladders.keys()))

    # Verify environment initializes cleanly
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

    with pytest.raises(KeyError):
        get_board_config("invalid_board_id")


def test_config_validation_errors():
    """Test that invalid board configurations raise appropriate ValueErrors."""
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

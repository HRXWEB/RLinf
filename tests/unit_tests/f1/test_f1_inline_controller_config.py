# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused Task 1 coverage for inline F1 Controller 0.2.0 configuration."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from f1_robot_controller import create_controller as create_real_controller

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlinf.envs.realworld.f1 import F1RobotConfig, F1RobotEnv  # noqa: E402
from rlinf.envs.realworld.f1 import f1_robot_env as f1_env_module  # noqa: E402


def valid_motion_envelope() -> dict[str, object]:
    """Return a closed Controller 0.2.0 motion envelope mapping."""

    arm = {
        "tcp": {
            "max_delta": {
                "position_m": 0.005,
                "orientation_deg": 1.0,
            },
            "workspace_bounds": {
                "position_m": {
                    "x": {"min": -1.0, "max": 1.0},
                    "y": {"min": -1.0, "max": 1.0},
                    "z": {"min": 0.0, "max": 1.0},
                },
                "orientation_deg": {
                    "rx": {"min": -180.0, "max": 180.0},
                    "ry": {"min": -180.0, "max": 180.0},
                    "rz": {"min": -180.0, "max": 180.0},
                },
            },
        }
    }
    return {
        "left_arm": arm,
        "right_arm": arm,
        "gripper_percent_closed": [0.0, 100.0],
    }


def fake_controller_config() -> dict[str, object]:
    """Return an inline fake backend config before env envelope injection."""

    return {
        "backend": "fake",
        "control_period_s": 0.001,
        "max_observation_age_s": 0.25,
        "max_observation_skew_s": 0.05,
        "fake": {
            "seed": 7,
            "image_height": 32,
            "image_width": 48,
            "command_latency_s": 0.0,
            "command_history_limit": 32,
        },
    }


def valid_action_scale() -> dict[str, float]:
    return {
        "tcp_position_m": 0.005,
        "tcp_orientation_deg": 1.0,
        "gripper_percent_closed": 10.0,
    }


def valid_env_cfg() -> dict[str, object]:
    return {
        "controller": fake_controller_config(),
        "action_scale": valid_action_scale(),
        "motion_envelope": valid_motion_envelope(),
        "max_num_steps": 10,
    }


def test_f1_config_is_complete_without_artifact_paths() -> None:
    cfg = F1RobotConfig(**valid_env_cfg())

    assert cfg.controller["backend"] == "fake"
    assert cfg.controller["motion_envelope"] == valid_motion_envelope()
    assert cfg.motion_envelope == valid_motion_envelope()
    assert cfg.action_scale == valid_action_scale()
    assert cfg.max_num_steps == 10


@pytest.mark.parametrize(
    "legacy_field",
    [
        "is_dummy",
        "architecture_smoke",
        "phase2_handoff",
        "command_capability",
    ],
)
def test_f1_config_rejects_removed_fields(legacy_field: str) -> None:
    payload = valid_env_cfg()
    payload[legacy_field] = True

    with pytest.raises(TypeError):
        F1RobotConfig(**payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("controller", "controller.json"),
        ("motion_envelope", "motion-envelope.json"),
    ],
)
def test_f1_config_requires_inline_mappings(field_name: str, value: object) -> None:
    payload = valid_env_cfg()
    payload[field_name] = value

    with pytest.raises(TypeError, match=field_name):
        F1RobotConfig(**payload)


def test_f1_config_copies_and_injects_motion_envelope_without_opening_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def fail_if_opened(_config: Mapping[str, Any]) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("F1RobotConfig must not open the controller")

    monkeypatch.setattr(f1_env_module, "create_controller", fail_if_opened)
    controller = fake_controller_config()
    envelope = valid_motion_envelope()

    cfg = F1RobotConfig(
        controller=controller,
        action_scale=valid_action_scale(),
        motion_envelope=envelope,
        max_num_steps=10,
    )
    controller["backend"] = "ros2"
    envelope["gripper_percent_closed"] = [10.0, 90.0]

    assert opened is False
    assert cfg.controller["backend"] == "fake"
    assert cfg.controller["motion_envelope"] == valid_motion_envelope()
    assert cfg.motion_envelope == valid_motion_envelope()


def test_f1_env_uses_injected_single_argument_controller_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls: list[Mapping[str, Any]] = []

    def controller_factory(config: Mapping[str, Any]) -> object:
        factory_calls.append(config)
        return create_real_controller(config)

    monkeypatch.setattr(f1_env_module, "create_controller", controller_factory)

    env = F1RobotEnv(F1RobotConfig(**valid_env_cfg()))
    try:
        assert factory_calls == [env.config.controller]
        assert factory_calls[0]["motion_envelope"] == valid_motion_envelope()
    finally:
        env.close()


def test_f1_env_scales_normalized_tcp_and_gripper_action() -> None:
    env = F1RobotEnv.__new__(F1RobotEnv)
    env.config = F1RobotConfig(**valid_env_cfg())

    np.testing.assert_allclose(
        env._physical_policy_delta(np.ones(14, dtype=np.float32)),
        np.array(
            [
                0.005,
                0.005,
                0.005,
                1.0,
                1.0,
                1.0,
                10.0,
                0.005,
                0.005,
                0.005,
                1.0,
                1.0,
                1.0,
                10.0,
            ],
            dtype=np.float64,
        ),
    )

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for immutable F1 replay dataset schemas."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import rlinf.data.embodied.f1_schema as f1_schema
from rlinf.data.embodied.f1_schema import (
    F1_ACTION_LAYOUT,
    F1_STATE_LAYOUT,
    F1_TRAJECTORY_FIELD_MAPPING,
    F1_TRANSITION_SCHEMA_VERSION,
    f1_action_bounds_from_motion_envelope,
    validate_f1_dataset_manifest,
    validate_f1_transition,
)

EXPECTED_DESCRIPTOR_FIELDS = {
    "schema_version",
    "task_id",
    "observation_layout",
    "action_layout",
    "camera_roles",
    "action_scale",
    "control_period_s",
    "source_type",
}


def _action_scale() -> dict[str, float]:
    return {
        "tcp_position_m": 0.005,
        "tcp_orientation_deg": 1.0,
        "gripper_percent_closed": 10.0,
    }


def _action_high() -> list[float]:
    return [0.01, 0.01, 0.01, 2.0, 2.0, 2.0, 5.0] * 2


def _manifest(source_type: str = "demo") -> dict[str, object]:
    return f1_schema.build_f1_replay_descriptor(
        task_id="F1DualArmPegInsertionEnv-v1",
        action_scale=_action_scale(),
        control_period_s=0.1,
        source_type=source_type,
    )


def _transition(source_type: str = "demo") -> dict[str, object]:
    return {
        "schema_version": F1_TRANSITION_SCHEMA_VERSION,
        "episode_id": "episode-0001",
        "step_index": 7,
        "source_type": source_type,
        "observation": {
            "state": [0.0] * 16,
            "images": {
                "head_color": {
                    "shape": [720, 1280, 3],
                    "dtype": "uint8",
                    "timestamp_ns": 10,
                },
                "left_wrist_color": {
                    "shape": [480, 848, 3],
                    "dtype": "uint8",
                    "timestamp_ns": 11,
                },
                "right_wrist_color": {
                    "shape": [480, 848, 3],
                    "dtype": "uint8",
                    "timestamp_ns": 12,
                },
            },
            "timestamp_ns": 9,
        },
        "action": [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0] * 2,
        "physical_delta_commanded": [
            -0.005,
            -0.0025,
            0.0,
            0.25,
            0.5,
            0.75,
            10.0,
        ]
        * 2,
        "absolute_target_commanded": {
            "left_arm": {
                "tcp_pose": {
                    "position_m": [0.295, 0.1975, 0.4],
                    "orientation_deg": [0.25, 0.5, 0.75],
                },
                "gripper_percent_closed": 60.0,
            },
            "right_arm": {
                "tcp_pose": {
                    "position_m": [0.295, -0.2025, 0.4],
                    "orientation_deg": [0.25, 0.5, 0.75],
                },
                "gripper_percent_closed": 60.0,
            },
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "timestamps": {
            "observation_ns": 9,
            "action_ns": 13,
            "reward_ns": 14,
        },
        "fault": None,
        "safety_abort": False,
    }


def _motion_envelope(
    *,
    left_position: list[float] | None = None,
    left_orientation: list[float] | None = None,
    left_gripper: float = 5.0,
    right_position: list[float] | None = None,
    right_orientation: list[float] | None = None,
    right_gripper: float = 6.0,
) -> dict[str, object]:
    """Build a minimal action envelope with deliberately asymmetric arms."""

    return {
        "action_envelope": {
            "left_arm": {
                "tcp": {
                    "max_delta": {
                        "position_m": left_position or [0.01, 0.02, 0.03],
                        "orientation_deg": left_orientation or [1.0, 2.0, 3.0],
                    }
                },
                "gripper": {"max_delta_percent_closed": left_gripper},
            },
            "right_arm": {
                "tcp": {
                    "max_delta": {
                        "position_m": right_position or [0.04, 0.05, 0.06],
                        "orientation_deg": right_orientation or [4.0, 5.0, 6.0],
                    }
                },
                "gripper": {"max_delta_percent_closed": right_gripper},
            },
        }
    }


def test_accepts_valid_manifest_and_transition() -> None:
    manifest = _manifest()
    transition = _transition()

    assert validate_f1_dataset_manifest(manifest) == manifest
    assert validate_f1_transition(transition, manifest) == transition


def test_descriptor_is_derived_from_runtime_config() -> None:
    descriptor = f1_schema.build_f1_replay_descriptor(
        task_id="F1DualArmPegInsertionEnv-v1",
        action_scale=_action_scale(),
        control_period_s=0.1,
        source_type="online",
    )

    assert set(descriptor) == EXPECTED_DESCRIPTOR_FIELDS
    assert descriptor["task_id"] == "F1DualArmPegInsertionEnv-v1"
    assert descriptor["action_scale"] == _action_scale()
    assert not any(
        token in json.dumps(descriptor)
        for token in ("sha256", "commit", "approval", "handoff", "path")
    )


def test_descriptor_schema_is_closed_and_rejects_provenance_fields() -> None:
    descriptor = f1_schema.build_f1_replay_descriptor(
        task_id="F1DualArmPegInsertionEnv-v1",
        action_scale=_action_scale(),
        control_period_s=0.1,
        source_type="demo",
    )
    descriptor["manifest_path"] = "/tmp/f1.json"

    with pytest.raises(ValueError, match="unexpected"):
        f1_schema.validate_f1_replay_descriptor(descriptor)


def test_transition_separates_policy_action_from_command_audit() -> None:
    manifest = _manifest()
    transition = _transition()

    assert validate_f1_transition(transition, manifest) == transition

    transition = _transition()
    transition.pop("physical_delta_commanded")
    with pytest.raises(ValueError, match="physical_delta_commanded"):
        validate_f1_transition(transition, manifest)

    transition = _transition()
    physical_delta = transition["physical_delta_commanded"]
    assert isinstance(physical_delta, list)
    physical_delta[0] = 0.25
    with pytest.raises(ValueError, match="physical_delta_commanded\\[0\\]"):
        validate_f1_transition(transition, manifest)


def test_transition_rejects_out_of_range_normalized_action() -> None:
    transition = _transition()
    action = transition["action"]
    assert isinstance(action, list)
    action[0] = 1.01

    with pytest.raises(ValueError, match="action\\[0\\]"):
        validate_f1_transition(transition, _manifest())


def test_action_layout_matches_env_14d_tcp_delta_contract() -> None:
    assert F1_ACTION_LAYOUT == (
        "left_dx_m",
        "left_dy_m",
        "left_dz_m",
        "left_droll_deg",
        "left_dpitch_deg",
        "left_dyaw_deg",
        "left_gripper_delta_percent_closed",
        "right_dx_m",
        "right_dy_m",
        "right_dz_m",
        "right_droll_deg",
        "right_dpitch_deg",
        "right_dyaw_deg",
        "right_gripper_delta_percent_closed",
    )


def test_state_layout_matches_flattened_env_state_order() -> None:
    assert F1_STATE_LAYOUT == (
        "left_joint_position[0]",
        "left_joint_position[1]",
        "left_joint_position[2]",
        "left_joint_position[3]",
        "left_joint_position[4]",
        "left_joint_position[5]",
        "left_joint_position[6]",
        "left_gripper[0]",
        "right_joint_position[0]",
        "right_joint_position[1]",
        "right_joint_position[2]",
        "right_joint_position[3]",
        "right_joint_position[4]",
        "right_joint_position[5]",
        "right_joint_position[6]",
        "right_gripper[0]",
    )


def test_action_bounds_are_derived_from_motion_envelope_not_constants() -> None:
    low, high = f1_action_bounds_from_motion_envelope(_motion_envelope())

    assert high == [
        0.01,
        0.02,
        0.03,
        1.0,
        2.0,
        3.0,
        5.0,
        0.04,
        0.05,
        0.06,
        4.0,
        5.0,
        6.0,
        6.0,
    ]
    assert low == [-value for value in high]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "f1-replay-descriptor-v0"),
        ("camera_roles", ["head_color", "missing"]),
        ("observation_layout", {"state": ["joint"] * 16}),
        ("action_layout", ["action"] * 14),
        ("control_period_s", "0.1"),
        ("source_type", "approved"),
        ("action_scale", {"tcp_position_m": -1.0}),
    ],
)
def test_rejects_manifest_contract_mismatches(field: str, value: object) -> None:
    manifest = _manifest()
    manifest[field] = value

    with pytest.raises(ValueError, match=field):
        validate_f1_dataset_manifest(manifest)


def test_rejects_closed_manifest_extra_fields() -> None:
    manifest = _manifest()
    manifest["silent_migration"] = "never"

    with pytest.raises(ValueError, match="unexpected"):
        validate_f1_dataset_manifest(manifest)


def test_rejects_camera_shape_and_dtype_mismatch() -> None:
    manifest = _manifest()
    observation_layout = manifest["observation_layout"]
    assert isinstance(observation_layout, dict)
    cameras = observation_layout["cameras"]
    assert isinstance(cameras, dict)
    left_wrist = cameras["left_wrist_color"]
    assert isinstance(left_wrist, dict)
    left_wrist["shape"] = [480, 640, 3]

    with pytest.raises(ValueError, match="observation_layout.cameras"):
        validate_f1_dataset_manifest(manifest)


def test_rejects_action_scale_mismatch() -> None:
    manifest = _manifest()
    action_scale = manifest["action_scale"]
    assert isinstance(action_scale, dict)
    action_scale["gripper_percent_closed"] = 0.0

    with pytest.raises(ValueError, match="action_scale.gripper_percent_closed"):
        validate_f1_dataset_manifest(manifest)


def test_rejects_non_finite_control_period() -> None:
    manifest = _manifest()
    manifest["control_period_s"] = float("nan")
    with pytest.raises(ValueError, match="control_period_s"):
        validate_f1_dataset_manifest(manifest)


def test_rejects_transition_action_outside_manifest_bounds() -> None:
    transition = _transition()
    action = transition["action"]
    assert isinstance(action, list)
    action[3] = 2.1

    with pytest.raises(ValueError, match="action\\[3\\]"):
        validate_f1_transition(transition, _manifest())


def test_descriptor_compatibility_is_exact_fieldwise() -> None:
    expected = validate_f1_dataset_manifest(_manifest())
    candidate = validate_f1_dataset_manifest(_manifest())
    assert (
        f1_schema.validate_f1_replay_descriptor_compatibility(candidate, expected)
        is None
    )

    candidate = validate_f1_dataset_manifest(_manifest())
    action_scale = candidate["action_scale"]
    assert isinstance(action_scale, dict)
    action_scale["tcp_position_m"] = 0.01
    with pytest.raises(ValueError, match="action_scale"):
        f1_schema.validate_f1_replay_descriptor_compatibility(candidate, expected)


def test_transition_rejects_type_coercion_and_unknown_fields() -> None:
    transition = _transition()
    transition["reward"] = "1.0"
    with pytest.raises(ValueError, match="reward"):
        validate_f1_transition(transition, _manifest())

    transition = _transition()
    transition["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_f1_transition(transition, _manifest())


@pytest.mark.parametrize("source_type", ["demo", "online"])
def test_fault_and_safety_abort_must_be_quarantined(source_type: str) -> None:
    transition = _transition(source_type=source_type)
    transition["fault"] = {"code": "controller_fault", "message": "stale"}
    with pytest.raises(ValueError, match="quarantine"):
        validate_f1_transition(transition, _manifest(source_type=source_type))

    transition = _transition(source_type=source_type)
    transition["safety_abort"] = True
    with pytest.raises(ValueError, match="quarantine"):
        validate_f1_transition(transition, _manifest(source_type=source_type))


def test_quarantine_transition_may_record_fault() -> None:
    transition = _transition(source_type="quarantine")
    transition["fault"] = {"code": "controller_fault", "message": "stale"}
    transition["safety_abort"] = True

    validate_f1_transition(transition, _manifest(source_type="quarantine"))


def test_transition_checks_episode_step_identity_and_camera_contract() -> None:
    transition = _transition()
    transition["step_index"] = -1
    with pytest.raises(ValueError, match="step_index"):
        validate_f1_transition(transition, _manifest())

    transition = _transition()
    observation = transition["observation"]
    assert isinstance(observation, dict)
    images = observation["images"]
    assert isinstance(images, dict)
    images["left_wrist_color"] = {
        "shape": [480, 848, 3],
        "dtype": "float32",
        "timestamp_ns": 10,
    }
    with pytest.raises(ValueError, match="left_wrist_color.dtype"):
        validate_f1_transition(transition, _manifest())


def test_transition_mapping_freezes_trajectory_integration_points() -> None:
    assert F1_TRAJECTORY_FIELD_MAPPING == {
        "observation": "curr_obs",
        "action": "actions",
        "physical_delta_commanded": "f1_transitions",
        "absolute_target_commanded": "f1_transitions",
        "reward": "rewards",
        "terminated": "terminations",
        "truncated": "truncations",
        "safety_abort": "f1_transitions",
        "fault": "f1_transitions",
    }


def test_validation_cli_is_read_only(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    transition_path = tmp_path / "transition.jsonl"
    descriptor_path.write_text(
        json.dumps(_manifest(), sort_keys=True),
        encoding="utf-8",
    )
    transition_path.write_text(json.dumps(_transition()) + "\n", encoding="utf-8")
    before_descriptor = descriptor_path.read_bytes()
    before_transition = transition_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            "toolkits/f1/validate_f1_dataset.py",
            "--descriptor",
            str(descriptor_path),
            "--transitions",
            str(transition_path),
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "validated 1 transitions" in result.stdout
    assert descriptor_path.read_bytes() == before_descriptor
    assert transition_path.read_bytes() == before_transition


def test_validation_cli_rejects_obsolete_expected_manifest_flag(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    expected_path = tmp_path / "expected.json"
    descriptor_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    expected_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "toolkits/f1/validate_f1_dataset.py",
            "--descriptor",
            str(descriptor_path),
            "--expected-manifest",
            str(expected_path),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --expected-manifest" in result.stderr

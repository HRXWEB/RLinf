# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Closed F1 replay dataset and transition schemas.

This module is intentionally dependency-light and side-effect-free so it can
validate offline artifacts before any ROS, Ray, or robot process starts.

The transition schema maps directly onto ``Trajectory`` fields in
``rlinf.data.embodied_io_struct``:

* ``observation`` -> ``curr_obs`` for the current step.
* ``action`` -> ``actions`` using the canonical 14D policy action layout:
  per-arm TCP deltas in metres/degrees followed by gripper delta percent.
* ``reward`` -> ``rewards``.
* ``terminated`` -> ``terminations``.
* ``truncated`` -> ``truncations``.
* ``safety_abort`` and non-null ``fault`` -> quarantine-only metadata, never
  normal ``demo`` or ``online`` replay.

The validator rejects unknown fields and exact-type mismatches instead of
migrating or coercing data silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, Any]

F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION = "f1-replay-descriptor-v1"
F1_DATASET_MANIFEST_SCHEMA_VERSION = F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION
F1_OBSERVATION_SCHEMA_VERSION = "f1-observation-v1"
F1_ACTION_SCHEMA_VERSION = "f1-action-v1"
F1_TRANSITION_SCHEMA_VERSION = "f1-transition-v1"

F1_SOURCE_TYPES = frozenset({"demo", "online", "quarantine"})

F1_CAMERA_ROLES = ("head_color", "left_wrist_color", "right_wrist_color")
F1_CAMERA_SPECS: Mapping[str, Mapping[str, object]] = {
    "head_color": {"shape": [720, 1280, 3], "dtype": "uint8"},
    "left_wrist_color": {"shape": [480, 848, 3], "dtype": "uint8"},
    "right_wrist_color": {"shape": [480, 848, 3], "dtype": "uint8"},
}

F1_STATE_LAYOUT = (
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

F1_ACTION_LAYOUT = (
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

F1_GRIPPER_SEMANTICS = {
    "unit": "percent_closed",
    "open": 0,
    "closed": 100,
    "action": "delta_percent_closed",
}

F1_DEFAULT_ACTION_SCALE_VECTOR = (
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
)

F1_TRAJECTORY_FIELD_MAPPING = {
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

_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "observation_layout",
        "action_layout",
        "camera_roles",
        "action_scale",
        "control_period_s",
        "source_type",
    }
)
_CAMERA_FIELDS = frozenset({"shape", "dtype"})
_TRANSITION_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "step_index",
        "source_type",
        "observation",
        "action",
        "physical_delta_commanded",
        "absolute_target_commanded",
        "reward",
        "terminated",
        "truncated",
        "timestamps",
        "fault",
        "safety_abort",
    }
)
_OBSERVATION_FIELDS = frozenset({"state", "images", "timestamp_ns"})
_IMAGE_FIELDS = frozenset({"shape", "dtype", "timestamp_ns"})
_TIMESTAMP_FIELDS = frozenset({"observation_ns", "action_ns", "reward_ns"})
_FAULT_FIELDS = frozenset({"code", "message"})
_ABSOLUTE_COMMAND_FIELDS = frozenset({"left_arm", "right_arm"})
_ABSOLUTE_ARM_FIELDS = frozenset({"tcp_pose", "gripper_percent_closed"})
_TCP_POSE_FIELDS = frozenset({"position_m", "orientation_deg"})


def f1_action_bounds_from_motion_envelope(
    motion_envelope: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Derive the canonical 14D F1 action bounds from a motion envelope.

    Args:
        motion_envelope: Mapping containing ``action_envelope.left_arm`` and
            ``action_envelope.right_arm`` entries with per-arm TCP
            ``max_delta.position_m``, ``max_delta.orientation_deg``, and
            ``gripper.max_delta_percent_closed`` values.

    Returns:
        ``(action_low, action_high)`` in ``F1_ACTION_LAYOUT`` order.

    Raises:
        ValueError: If the envelope is missing required values or contains
            non-finite/non-positive bounds.
    """

    envelope = _require_object(motion_envelope, "motion_envelope")
    action_envelope = _require_object(
        envelope.get("action_envelope"), "motion_envelope.action_envelope"
    )
    high: list[float] = []
    for arm_name in ("left_arm", "right_arm"):
        arm = _require_object(
            action_envelope.get(arm_name),
            f"motion_envelope.action_envelope.{arm_name}",
        )
        tcp = _require_object(
            arm.get("tcp"),
            f"motion_envelope.action_envelope.{arm_name}.tcp",
        )
        max_delta = _require_object(
            tcp.get("max_delta"),
            f"motion_envelope.action_envelope.{arm_name}.tcp.max_delta",
        )
        position = _validate_positive_numeric_vector(
            max_delta.get("position_m"),
            3,
            f"motion_envelope.action_envelope.{arm_name}.tcp.max_delta.position_m",
        )
        orientation = _validate_positive_numeric_vector(
            max_delta.get("orientation_deg"),
            3,
            f"motion_envelope.action_envelope.{arm_name}.tcp.max_delta.orientation_deg",
        )
        gripper = _require_object(
            arm.get("gripper"),
            f"motion_envelope.action_envelope.{arm_name}.gripper",
        )
        gripper_delta = _require_positive_number(
            gripper.get("max_delta_percent_closed"),
            (
                "motion_envelope.action_envelope."
                f"{arm_name}.gripper.max_delta_percent_closed"
            ),
        )
        high.extend([*position, *orientation, gripper_delta])
    return [-value for value in high], high


def build_f1_replay_descriptor(
    *,
    task_id: str,
    action_scale: Mapping[str, Any],
    control_period_s: float,
    source_type: str,
) -> dict[str, JsonValue]:
    """Build the in-memory F1 replay descriptor from runtime config."""

    descriptor: dict[str, JsonValue] = {
        "schema_version": F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION,
        "task_id": task_id,
        "observation_layout": {
            "schema_version": F1_OBSERVATION_SCHEMA_VERSION,
            "state": list(F1_STATE_LAYOUT),
            "cameras": {role: dict(F1_CAMERA_SPECS[role]) for role in F1_CAMERA_ROLES},
        },
        "action_layout": list(F1_ACTION_LAYOUT),
        "camera_roles": list(F1_CAMERA_ROLES),
        "action_scale": _validate_action_scale_mapping(action_scale),
        "control_period_s": float(control_period_s),
        "source_type": source_type,
    }
    return validate_f1_replay_descriptor(descriptor)


def validate_f1_replay_descriptor(
    descriptor: Mapping[str, Any],
) -> dict[str, JsonValue]:
    """Validate the closed in-memory F1 replay descriptor schema."""

    data = _require_object(descriptor, "descriptor")
    _require_exact_fields(data, _DESCRIPTOR_FIELDS, "descriptor")
    _require_equal(
        data["schema_version"],
        F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION,
        "schema_version",
    )
    _require_non_empty_string(data["task_id"], "task_id")
    observation_layout = _require_object(
        data["observation_layout"], "observation_layout"
    )
    _require_exact_fields(
        observation_layout,
        frozenset({"schema_version", "state", "cameras"}),
        "observation_layout",
    )
    _require_equal(
        observation_layout["schema_version"],
        F1_OBSERVATION_SCHEMA_VERSION,
        "observation_layout.schema_version",
    )
    _require_equal(observation_layout["state"], list(F1_STATE_LAYOUT), "state_layout")
    _validate_cameras(
        _require_object(observation_layout["cameras"], "observation_layout.cameras"),
        "observation_layout.cameras",
    )
    _require_equal(data["action_layout"], list(F1_ACTION_LAYOUT), "action_layout")
    _require_equal(data["camera_roles"], list(F1_CAMERA_ROLES), "camera_roles")
    data["action_scale"] = _validate_action_scale_mapping(data["action_scale"])
    _require_exact_float(data["control_period_s"], "control_period_s")
    if data["control_period_s"] <= 0.0:
        raise ValueError("control_period_s must be positive")
    _validate_source_type(data["source_type"], "source_type")
    return dict(data)


def validate_f1_dataset_manifest(manifest: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Validate the current F1 replay descriptor.

    The public name is retained for import stability inside this branch; the
    accepted schema is the in-memory descriptor, not an external artifact.
    """

    return validate_f1_replay_descriptor(manifest)


def validate_f1_replay_descriptor_compatibility(
    candidate: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Validate that two F1 replay descriptors match field by field."""

    candidate_data = validate_f1_replay_descriptor(candidate)
    expected_data = validate_f1_replay_descriptor(expected)
    for field in sorted(_DESCRIPTOR_FIELDS):
        if candidate_data[field] != expected_data[field]:
            raise ValueError(f"descriptor field {field} differs")


def validate_f1_transition(
    transition: Mapping[str, Any], manifest: Mapping[str, Any] | None = None
) -> JsonObject:
    """Validate one F1 replay transition.

    Args:
        transition: JSON-compatible transition mapping.
        manifest: Optional dataset manifest. When provided, transition camera
            roles, image shapes/dtypes, and source type must match it exactly.

    Returns:
        The original transition as a plain ``dict`` after validation.

    Raises:
        ValueError: If the transition does not match the frozen schema, or if
            fault/safety-abort data is marked as normal demo/online replay.
    """

    data = _require_object(transition, "transition")
    manifest_data = (
        validate_f1_replay_descriptor(manifest) if manifest is not None else None
    )
    _require_exact_fields(data, _TRANSITION_FIELDS, "transition")
    _require_equal(
        data["schema_version"], F1_TRANSITION_SCHEMA_VERSION, "schema_version"
    )
    _require_non_empty_string(data["episode_id"], "episode_id")
    _require_non_negative_int(data["step_index"], "step_index")
    _validate_source_type(data["source_type"], "source_type")
    if (
        manifest_data is not None
        and data["source_type"] != manifest_data["source_type"]
    ):
        raise ValueError("source_type must match manifest source_type")
    _validate_observation(
        _require_object(data["observation"], "observation"),
        manifest_data,
    )
    policy_action = _validate_numeric_vector(
        data["action"], len(F1_ACTION_LAYOUT), "action"
    )
    _validate_action_within_bounds(
        policy_action,
        [-1.0] * len(F1_ACTION_LAYOUT),
        [1.0] * len(F1_ACTION_LAYOUT),
        path="action",
    )
    physical_delta = _validate_numeric_vector(
        data["physical_delta_commanded"],
        len(F1_ACTION_LAYOUT),
        "physical_delta_commanded",
    )
    if manifest_data is not None:
        scale = _action_scale_vector_from_mapping(manifest_data["action_scale"])
        for index, (actual, normalized, factor) in enumerate(
            zip(physical_delta, policy_action, scale, strict=True)
        ):
            expected = normalized * factor
            if abs(actual - expected) > 1e-9:
                raise ValueError(
                    "physical_delta_commanded"
                    f"[{index}] must equal action[{index}] * action_scale[{index}]"
                )
    _validate_absolute_target_command(
        _require_object(data["absolute_target_commanded"], "absolute_target_commanded")
    )
    _require_number(data["reward"], "reward")
    _require_bool(data["terminated"], "terminated")
    _require_bool(data["truncated"], "truncated")
    _validate_timestamps(_require_object(data["timestamps"], "timestamps"))
    _validate_fault(data["fault"])
    _require_bool(data["safety_abort"], "safety_abort")
    if data["source_type"] != "quarantine" and (
        data["fault"] is not None or data["safety_abort"]
    ):
        raise ValueError("fault and safety_abort transitions must be quarantine")
    return data


def _validate_cameras(cameras: Mapping[str, Any], path: str) -> None:
    _require_exact_fields(cameras, frozenset(F1_CAMERA_ROLES), path)
    for role in F1_CAMERA_ROLES:
        camera_path = f"{path}.{role}"
        camera = _require_object(cameras[role], camera_path)
        _require_exact_fields(camera, _CAMERA_FIELDS, camera_path)
        spec = F1_CAMERA_SPECS[role]
        _require_equal(camera["shape"], spec["shape"], f"{camera_path}.shape")
        _require_equal(camera["dtype"], spec["dtype"], f"{camera_path}.dtype")


def _validate_action_bounds(action_low: Any, action_high: Any) -> None:
    lows = _validate_numeric_vector(action_low, len(F1_ACTION_LAYOUT), "action_low")
    highs = _validate_numeric_vector(action_high, len(F1_ACTION_LAYOUT), "action_high")
    for index, (low, high) in enumerate(zip(lows, highs, strict=True)):
        if low >= high:
            raise ValueError(
                f"action_low[{index}] must be less than action_high[{index}]"
            )


def _validate_action_scale_mapping(value: Any) -> dict[str, JsonValue]:
    scale = _require_object(value, "action_scale")
    _require_exact_fields(
        scale,
        frozenset(
            {
                "tcp_position_m",
                "tcp_orientation_deg",
                "gripper_percent_closed",
            }
        ),
        "action_scale",
    )
    return {
        "tcp_position_m": _require_positive_number(
            scale["tcp_position_m"], "action_scale.tcp_position_m"
        ),
        "tcp_orientation_deg": _require_positive_number(
            scale["tcp_orientation_deg"], "action_scale.tcp_orientation_deg"
        ),
        "gripper_percent_closed": _require_positive_number(
            scale["gripper_percent_closed"],
            "action_scale.gripper_percent_closed",
        ),
    }


def _action_scale_vector_from_mapping(value: Any) -> list[float]:
    scale = _validate_action_scale_mapping(value)
    return [
        scale["tcp_position_m"],
        scale["tcp_position_m"],
        scale["tcp_position_m"],
        scale["tcp_orientation_deg"],
        scale["tcp_orientation_deg"],
        scale["tcp_orientation_deg"],
        scale["gripper_percent_closed"],
        scale["tcp_position_m"],
        scale["tcp_position_m"],
        scale["tcp_position_m"],
        scale["tcp_orientation_deg"],
        scale["tcp_orientation_deg"],
        scale["tcp_orientation_deg"],
        scale["gripper_percent_closed"],
    ]


def _validate_action_within_bounds(
    action: list[float],
    action_low: list[float],
    action_high: list[float],
    *,
    path: str = "action",
) -> None:
    for index, (value, low, high) in enumerate(
        zip(action, action_low, action_high, strict=True)
    ):
        if value < low or value > high:
            raise ValueError(f"{path}[{index}] must be within [{low!r}, {high!r}]")


def _validate_absolute_target_command(command: Mapping[str, Any]) -> None:
    _require_exact_fields(
        command, _ABSOLUTE_COMMAND_FIELDS, "absolute_target_commanded"
    )
    for arm_name in ("left_arm", "right_arm"):
        arm = _require_object(
            command[arm_name], f"absolute_target_commanded.{arm_name}"
        )
        _require_exact_fields(
            arm, _ABSOLUTE_ARM_FIELDS, f"absolute_target_commanded.{arm_name}"
        )
        tcp_pose = _require_object(
            arm["tcp_pose"], f"absolute_target_commanded.{arm_name}.tcp_pose"
        )
        _require_exact_fields(
            tcp_pose,
            _TCP_POSE_FIELDS,
            f"absolute_target_commanded.{arm_name}.tcp_pose",
        )
        _validate_numeric_vector(
            tcp_pose["position_m"],
            3,
            f"absolute_target_commanded.{arm_name}.tcp_pose.position_m",
        )
        _validate_numeric_vector(
            tcp_pose["orientation_deg"],
            3,
            f"absolute_target_commanded.{arm_name}.tcp_pose.orientation_deg",
        )
        gripper = _require_number(
            arm["gripper_percent_closed"],
            f"absolute_target_commanded.{arm_name}.gripper_percent_closed",
        )
        if gripper < 0.0 or gripper > 100.0:
            raise ValueError(
                f"absolute_target_commanded.{arm_name}.gripper_percent_closed must be within [0, 100]"
            )


def _validate_observation(
    observation: Mapping[str, Any], manifest: Mapping[str, Any] | None
) -> None:
    _require_exact_fields(observation, _OBSERVATION_FIELDS, "observation")
    _validate_numeric_vector(
        observation["state"], len(F1_STATE_LAYOUT), "observation.state"
    )
    images = _require_object(observation["images"], "observation.images")
    camera_roles = (
        tuple(manifest["camera_roles"]) if manifest is not None else F1_CAMERA_ROLES
    )
    _require_exact_fields(images, frozenset(camera_roles), "observation.images")
    for role in camera_roles:
        image_path = f"observation.images.{role}"
        image = _require_object(images[role], image_path)
        _require_exact_fields(image, _IMAGE_FIELDS, image_path)
        _validate_image_shape(image["shape"], f"{role}.shape")
        _require_equal(image["dtype"], "uint8", f"{role}.dtype")
        _require_non_negative_int(image["timestamp_ns"], f"{image_path}.timestamp_ns")
    _require_non_negative_int(observation["timestamp_ns"], "observation.timestamp_ns")


def _validate_image_shape(value: Any, path: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{path} must be an HxWxC shape")
    for index, dim in enumerate(value):
        _require_positive_int(dim, f"{path}[{index}]")


def _validate_timestamps(timestamps: Mapping[str, Any]) -> None:
    _require_exact_fields(timestamps, _TIMESTAMP_FIELDS, "timestamps")
    for field in sorted(_TIMESTAMP_FIELDS):
        _require_non_negative_int(timestamps[field], f"timestamps.{field}")


def _validate_fault(fault: Any) -> None:
    if fault is None:
        return
    fault_data = _require_object(fault, "fault")
    _require_exact_fields(fault_data, _FAULT_FIELDS, "fault")
    _require_non_empty_string(fault_data["code"], "fault.code")
    _require_non_empty_string(fault_data["message"], "fault.message")


def _validate_source_type(value: Any, path: str) -> None:
    _require_non_empty_string(value, path)
    if value not in F1_SOURCE_TYPES:
        raise ValueError(f"{path} must be one of {sorted(F1_SOURCE_TYPES)}")


def _require_object(value: Any, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected_fields: frozenset[str], path: str
) -> None:
    actual_fields = set(value)
    missing = expected_fields - actual_fields
    unexpected = actual_fields - expected_fields
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{path} unexpected fields: {sorted(unexpected)}")


def _require_equal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise ValueError(f"{path} must be {expected!r}")


def _require_non_empty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")


def _require_hex_string(value: Any, length: int, path: str) -> None:
    _require_non_empty_string(value, path)
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{path} must be a lowercase {length}-character hex string")


def _require_exact_float(value: Any, path: str) -> None:
    if not isinstance(value, float):
        raise ValueError(f"{path} must be a float")
    if not isfinite(value):
        raise ValueError(f"{path} must be finite")


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{path} must be finite")
    return normalized


def _require_positive_number(value: Any, path: str) -> float:
    normalized = _require_number(value, path)
    if normalized <= 0.0:
        raise ValueError(f"{path} must be positive")
    return normalized


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a bool")


def _require_non_negative_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")


def _require_positive_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value <= 0:
        raise ValueError(f"{path} must be positive")


def _validate_numeric_vector(
    value: Any, expected_length: int, path: str
) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if len(value) != expected_length:
        raise ValueError(f"{path} must contain {expected_length} values")
    normalized: list[float] = []
    for index, item in enumerate(value):
        normalized.append(_require_number(item, f"{path}[{index}]"))
    return normalized


def _validate_positive_numeric_vector(
    value: Any, expected_length: int, path: str
) -> list[float]:
    values = _validate_numeric_vector(value, expected_length, path)
    for index, item in enumerate(values):
        if item <= 0.0:
            raise ValueError(f"{path}[{index}] must be positive")
    return values

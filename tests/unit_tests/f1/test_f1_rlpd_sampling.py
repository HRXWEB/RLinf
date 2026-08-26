# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for F1 replay admission and RLPD replay sampling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import rlinf.data.embodied.f1_schema as f1_schema
from rlinf.data.embodied.f1_schema import (
    F1_TRANSITION_SCHEMA_VERSION,
)
from rlinf.data.embodied_buffer_dataset import ReplayBufferDataset
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.data.f1_replay_admission import F1ReplayAdmission


class _FakeBuffer:
    def __init__(self, source_id: int) -> None:
        self.source_id = source_id
        self.sample_sizes: list[int] = []

    def is_ready(self, min_size: int) -> bool:
        return True

    def sample(self, num_chunks: int) -> dict[str, torch.Tensor]:
        self.sample_sizes.append(num_chunks)
        return {"source": torch.full((num_chunks,), self.source_id)}


def _manifest(source_type: str = "online") -> dict[str, object]:
    return _descriptor(source_type)


def _descriptor(source_type: str = "online") -> dict[str, object]:
    return f1_schema.build_f1_replay_descriptor(
        task_id="F1DualArmPegInsertionEnv-v1",
        action_scale={
            "tcp_position_m": 0.005,
            "tcp_orientation_deg": 1.0,
            "gripper_percent_closed": 10.0,
        },
        control_period_s=0.1,
        source_type=source_type,
    )


def _transition(source_type: str = "online") -> dict[str, object]:
    return {
        "schema_version": F1_TRANSITION_SCHEMA_VERSION,
        "episode_id": "episode-0001",
        "step_index": 0,
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
        "action": [0.0] * 14,
        "physical_delta_commanded": [0.0] * 14,
        "absolute_target_commanded": {
            "left_arm": {
                "tcp_pose": {
                    "position_m": [0.3, 0.2, 0.4],
                    "orientation_deg": [0.0, 0.0, 0.0],
                },
                "gripper_percent_closed": 50.0,
            },
            "right_arm": {
                "tcp_pose": {
                    "position_m": [0.3, -0.2, 0.4],
                    "orientation_deg": [0.0, 0.0, 0.0],
                },
                "gripper_percent_closed": 50.0,
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


def _audited_transition(source_type: str = "online") -> dict[str, object]:
    transition = _transition(source_type)
    transition["action"] = [0.5] * 14
    transition["physical_delta_commanded"] = [
        0.0025,
        0.0025,
        0.0025,
        0.5,
        0.5,
        0.5,
        5.0,
    ] * 2
    transition["absolute_target_commanded"] = {
        "left_arm": {
            "tcp_pose": {
                "position_m": [0.3025, 0.2025, 0.4025],
                "orientation_deg": [0.5, 0.5, 0.5],
            },
            "gripper_percent_closed": 55.0,
        },
        "right_arm": {
            "tcp_pose": {
                "position_m": [0.3025, -0.1975, 0.4025],
                "orientation_deg": [0.5, 0.5, 0.5],
            },
            "gripper_percent_closed": 55.0,
        },
    }
    return transition


def _trajectory(
    *,
    manifest: dict[str, object] | None = None,
    transitions: list[dict[str, object]] | None = None,
) -> Trajectory:
    return Trajectory(
        actions=torch.zeros((1, 1, 14)),
        rewards=torch.zeros((1, 1, 1)),
        f1_manifest=manifest,
        f1_transitions=transitions,
    )


def test_non_f1_trajectories_keep_backward_compatible_admission() -> None:
    admission = F1ReplayAdmission(enabled=False, expected_descriptor=None)

    accepted = admission.admit_online([Trajectory(rewards=torch.zeros((1, 1, 1)))])

    assert len(accepted) == 1
    assert admission.metrics() == {"replay/quarantine_rejected": 0.0}


def test_f1_admission_rejects_missing_or_incompatible_descriptor() -> None:
    expected = _descriptor("online")
    admission = F1ReplayAdmission(enabled=True, expected_descriptor=expected)

    with pytest.raises(ValueError, match="descriptor"):
        admission.admit_online([_trajectory(transitions=[_transition("online")])])

    incompatible = deepcopy(expected)
    incompatible["control_period_s"] = 0.2
    with pytest.raises(ValueError, match="control_period_s"):
        admission.admit_online(
            [
                _trajectory(
                    manifest=incompatible,
                    transitions=[_transition("online")],
                )
            ]
        )


def test_f1_admission_rejects_quarantine_fault_and_safety_abort() -> None:
    expected = _descriptor("online")
    admission = F1ReplayAdmission(enabled=True, expected_descriptor=expected)

    quarantined_manifest = _descriptor("quarantine")
    quarantined_transition = _transition("quarantine")
    quarantined_transition["safety_abort"] = True
    with pytest.raises(ValueError, match="quarantine"):
        admission.admit_online(
            [
                _trajectory(
                    manifest=quarantined_manifest,
                    transitions=[quarantined_transition],
                )
            ]
        )

    faulted_transition = _transition("online")
    faulted_transition["fault"] = {"code": "controller_fault", "message": "stale"}
    with pytest.raises(ValueError, match="quarantine"):
        admission.admit_online(
            [_trajectory(manifest=expected, transitions=[faulted_transition])]
        )

    assert admission.metrics()["replay/quarantine_rejected"] == 2.0


def test_f1_admission_rejects_missing_transitions_fail_closed() -> None:
    admission = F1ReplayAdmission(
        enabled=True, expected_descriptor=_descriptor("online")
    )

    with pytest.raises(ValueError, match="transitions"):
        admission.admit_online([_trajectory(manifest=_descriptor("online"))])


def test_f1_admission_rejects_action_audit_mismatch_fail_closed() -> None:
    descriptor = _descriptor("online")
    transition = _transition("online")
    transition["action"] = [0.5] * 14
    trajectory = _trajectory(manifest=descriptor, transitions=[transition])

    with pytest.raises(ValueError, match="action"):
        F1ReplayAdmission(
            enabled=True,
            expected_descriptor=descriptor,
        ).admit_online([trajectory])


def test_f1_admission_rejects_invalid_transition_fail_closed() -> None:
    descriptor = _descriptor("online")
    transition = _transition("online")
    observation = transition["observation"]
    assert isinstance(observation, dict)
    images = observation["images"]
    assert isinstance(images, dict)
    images["head_color"]["bytes"] = "not allowed"

    with pytest.raises(ValueError, match="unexpected"):
        F1ReplayAdmission(
            enabled=True,
            expected_descriptor=descriptor,
        ).admit_online([_trajectory(manifest=descriptor, transitions=[transition])])


def test_f1_rollout_result_preserves_env_step_metadata_for_replay_admission() -> None:
    manifest = _manifest("online")
    transition = _transition("online")
    rollout = EmbodiedRolloutResult(max_episode_length=4)

    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.zeros((1, 14)),
            rewards=torch.zeros((1, 1)),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.zeros((1, 1), dtype=torch.bool),
            dones=torch.zeros((1, 1), dtype=torch.bool),
            env_infos={
                "f1_manifest": manifest,
                "f1_transitions": [transition],
            },
        )
    )
    trajectory = rollout.to_trajectory()

    accepted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_online([trajectory])

    assert accepted == [trajectory]
    assert trajectory.f1_manifest == manifest
    assert trajectory.f1_transitions == [transition]


def test_f1_rollout_unwraps_vector_info_before_splitting_for_replay() -> None:
    """Gym vector-info object arrays remain trajectory-level F1 metadata."""
    manifest = _manifest("online")
    transition = _audited_transition("online")
    vector_manifest = np.empty(1, dtype=object)
    vector_manifest[0] = manifest
    vector_transitions = np.empty(1, dtype=object)
    vector_transitions[0] = [transition]
    rollout = EmbodiedRolloutResult(max_episode_length=1)

    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.full((1, 14), 0.5),
            rewards=torch.zeros((1, 1)),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.ones((1, 1), dtype=torch.bool),
            dones=torch.ones((1, 1), dtype=torch.bool),
            env_infos={
                "f1_manifest": vector_manifest,
                "_f1_manifest": np.array([True]),
                "f1_transitions": vector_transitions,
                "_f1_transitions": np.array([True]),
            },
        )
    )

    trajectories = rollout.to_splited_trajectories(split_size=1)
    admitted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_online(trajectories)

    assert admitted == trajectories
    assert trajectories[0].f1_manifest == manifest
    assert trajectories[0].f1_transitions == [transition]


def test_f1_admission_keeps_replay_actions_normalized() -> None:
    manifest = _manifest("online")
    transition = _audited_transition("online")
    normalized_action = torch.full((1, 1, 14), 0.5)
    trajectory = Trajectory(
        actions=normalized_action,
        rewards=torch.zeros((1, 1, 1)),
        f1_manifest=manifest,
        f1_transitions=[transition],
    )

    accepted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_online([trajectory])

    assert accepted == [trajectory]
    assert torch.equal(trajectory.actions, normalized_action)
    assert trajectory.f1_transitions[0]["physical_delta_commanded"][6] == 5.0
    assert (
        trajectory.f1_transitions[0]["absolute_target_commanded"]["left_arm"][
            "gripper_percent_closed"
        ]
        == 55.0
    )


def test_f1_admission_rejects_action_audit_mismatch() -> None:
    manifest = _manifest("online")
    transition = _audited_transition("online")
    trajectory = Trajectory(
        actions=torch.zeros((1, 1, 14)),
        rewards=torch.zeros((1, 1, 1)),
        f1_manifest=manifest,
        f1_transitions=[transition],
    )

    with pytest.raises(ValueError, match="action"):
        F1ReplayAdmission(
            enabled=True,
            expected_descriptor=manifest,
        ).admit_online([trajectory])


@pytest.mark.parametrize("split_method", ["equal", "sizes"])
def test_f1_split_trajectories_keep_actions_and_audit_metadata_aligned(
    split_method: str,
) -> None:
    manifest = _manifest("online")
    left_transition = _transition("online")
    right_transition = deepcopy(left_transition)
    right_action = [0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5] * 2
    right_transition["action"] = right_action
    right_transition["physical_delta_commanded"] = [
        0.0025,
        0.0025,
        0.0025,
        0.25,
        0.25,
        0.25,
        5.0,
    ] * 2
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[0.0] * 14, right_action]),
            rewards=torch.zeros((2, 1)),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.ones((2, 1), dtype=torch.bool),
            dones=torch.ones((2, 1), dtype=torch.bool),
            env_infos={
                "f1_manifest": manifest,
                "f1_transitions": [left_transition, right_transition],
            },
        )
    )

    trajectories = (
        rollout.to_splited_trajectories(2)
        if split_method == "equal"
        else rollout.to_splited_trajectories_by_sizes([1, 1])
    )
    admitted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_online(trajectories)

    assert admitted == trajectories
    assert trajectories[0].f1_transitions == [left_transition]
    assert trajectories[1].f1_transitions == [right_transition]


def test_f1_async_sac_allows_one_trailing_bootstrap_action() -> None:
    """Only executed actions require physical transition audit metadata."""

    manifest = _manifest("online")
    first_transition = _transition("online")
    second_transition = deepcopy(first_transition)
    second_transition["step_index"] = 1
    first_action = torch.tensor(first_transition["action"])
    second_action = torch.tensor(second_transition["action"])
    bootstrap_action = torch.full((14,), 0.75)
    rollout = EmbodiedRolloutResult(max_episode_length=2)

    rollout.append_step_result(ChunkStepResult(actions=first_action.unsqueeze(0)))
    rollout.append_step_result(
        ChunkStepResult(
            actions=second_action.unsqueeze(0),
            rewards=torch.zeros((1, 1)),
            env_infos={"f1_manifest": manifest, "f1_transitions": [first_transition]},
        )
    )
    rollout.append_step_result(
        ChunkStepResult(
            actions=bootstrap_action.unsqueeze(0),
            rewards=torch.zeros((1, 1)),
            env_infos={"f1_manifest": manifest, "f1_transitions": [second_transition]},
        )
    )

    trajectory = rollout.to_splited_trajectories(split_size=1)[0]
    admitted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_online([trajectory])

    assert admitted == [trajectory]
    assert trajectory.actions.shape == (3, 1, 14)
    assert trajectory.rewards.shape == (2, 1, 1)
    assert trajectory.f1_transitions == [first_transition, second_transition]


def test_f1_rollout_collects_terminal_transition_from_final_info() -> None:
    """Gymnasium auto-reset moves the terminal step metadata into final_info."""

    manifest = _manifest("online")
    transition = _transition("online")
    vector_manifest = np.empty(1, dtype=object)
    vector_manifest[0] = manifest
    vector_transitions = np.empty(1, dtype=object)
    vector_transitions[0] = [transition]
    rollout = EmbodiedRolloutResult(max_episode_length=1)

    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.zeros((1, 14)),
            rewards=torch.zeros((1, 1)),
            env_infos={
                "f1_manifest": vector_manifest,
                "_f1_manifest": np.array([True]),
                "final_info": {
                    "f1_manifest": vector_manifest,
                    "_f1_manifest": np.array([True]),
                    "f1_transitions": vector_transitions,
                    "_f1_transitions": np.array([True]),
                },
                "_final_info": np.array([True]),
            },
        )
    )

    trajectory = rollout.to_trajectory()

    assert trajectory.f1_manifest == manifest
    assert trajectory.f1_transitions == [transition]


def test_f1_intervention_extraction_keeps_selected_audit_metadata_aligned() -> None:
    manifest = _manifest("demo")
    action_values = (0.001, 0.002, 0.003, 0.004)
    transitions = []
    actions = []
    for step_index, value in enumerate(action_values):
        action = [value, value, value, 0.0, 0.0, 0.0, 0.0] * 2
        transition = _transition("demo")
        transition["step_index"] = step_index // 2
        transition["action"] = action
        transition["physical_delta_commanded"] = [
            value * 0.005,
            value * 0.005,
            value * 0.005,
            0.0,
            0.0,
            0.0,
            0.0,
        ] * 2
        transitions.append(transition)
        actions.append(action)
    intervene_flags = torch.zeros((2, 2, 14), dtype=torch.bool)
    intervene_flags[0, 0] = True
    intervene_flags[1, 1] = True
    trajectory = Trajectory(
        actions=torch.tensor(actions).reshape(2, 2, 14),
        intervene_flags=intervene_flags,
        rewards=torch.zeros((2, 2, 1)),
        f1_manifest=manifest,
        f1_transitions=transitions,
    )

    extracted = trajectory.extract_intervene_traj()
    assert extracted is not None
    admitted = F1ReplayAdmission(
        enabled=True,
        expected_descriptor=manifest,
    ).admit_demo(extracted)

    assert admitted == extracted
    assert extracted[0].f1_transitions == [transitions[0]]
    assert extracted[1].f1_transitions == [transitions[3]]


def test_f1_split_trajectories_do_not_alias_command_audit_metadata() -> None:
    manifest = _manifest("online")
    left_transition = _audited_transition("online")
    right_transition = deepcopy(left_transition)
    rollout = EmbodiedRolloutResult(max_episode_length=10)
    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.full((2, 14), 0.5),
            rewards=torch.zeros((2, 1)),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.zeros((2, 1), dtype=torch.bool),
            dones=torch.zeros((2, 1), dtype=torch.bool),
            env_infos={
                "f1_manifest": manifest,
                "f1_transitions": [left_transition, right_transition],
            },
        )
    )

    left, right = rollout.to_splited_trajectories_by_sizes([1, 1])
    assert left.actions.shape == (1, 1, 14)
    assert right.actions.shape == (1, 1, 14)

    left.f1_transitions[0]["physical_delta_commanded"][0] = 99.0

    assert right.f1_transitions[0]["physical_delta_commanded"][0] == 0.0025
    assert rollout.f1_transitions[0]["physical_delta_commanded"][0] == 0.0025


def test_f1_rollout_result_does_not_fabricate_faulted_replay_metadata() -> None:
    manifest = _manifest("online")
    faulted_transition = _transition("online")
    faulted_transition["fault"] = {
        "code": "controller_fault",
        "message": "dispatch timeout",
    }
    rollout = EmbodiedRolloutResult(max_episode_length=4)

    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.zeros((1, 14)),
            rewards=torch.zeros((1, 1)),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.ones((1, 1), dtype=torch.bool),
            dones=torch.ones((1, 1), dtype=torch.bool),
            env_infos={
                "f1_manifest": manifest,
                "f1_transitions": [faulted_transition],
            },
        )
    )

    with pytest.raises(ValueError, match="quarantine"):
        F1ReplayAdmission(
            enabled=True,
            expected_descriptor=manifest,
        ).admit_online([rollout.to_trajectory()])


def test_rlpd_demo_ratio_is_configurable_and_preserves_odd_batch_size() -> None:
    replay_buffer = _FakeBuffer(source_id=0)
    demo_buffer = _FakeBuffer(source_id=1)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        batch_size=7,
        min_replay_buffer_size=1,
        min_demo_buffer_size=1,
        demo_fraction=3 / 7,
    )

    batch = next(iter(dataset))

    assert replay_buffer.sample_sizes == [4]
    assert demo_buffer.sample_sizes == [3]
    assert batch["source"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert dataset.sample_metrics() == {
        "replay/online_fraction": 4 / 7,
        "replay/demo_fraction": 3 / 7,
    }


def test_pure_online_sampling_keeps_existing_batch_size() -> None:
    replay_buffer = _FakeBuffer(source_id=0)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=None,
        batch_size=7,
        min_replay_buffer_size=1,
        min_demo_buffer_size=0,
        demo_fraction=0.75,
    )

    batch = next(iter(dataset))

    assert replay_buffer.sample_sizes == [7]
    assert batch["source"].shape == (7,)
    assert dataset.sample_metrics() == {
        "replay/online_fraction": 1.0,
        "replay/demo_fraction": 0.0,
    }


@pytest.mark.parametrize(
    ("demo_fraction", "expected_online", "expected_demo", "expected_source"),
    [
        (0.0, 5, 0, [0, 0, 0, 0, 0]),
        (1.0, 0, 5, [1, 1, 1, 1, 1]),
    ],
)
def test_rlpd_demo_ratio_supports_zero_or_all_demo_without_sampling_zero_chunks(
    demo_fraction: float,
    expected_online: int,
    expected_demo: int,
    expected_source: list[int],
) -> None:
    replay_buffer = _FakeBuffer(source_id=0)
    demo_buffer = _FakeBuffer(source_id=1)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        batch_size=5,
        min_replay_buffer_size=1,
        min_demo_buffer_size=1,
        demo_fraction=demo_fraction,
    )

    batch = next(iter(dataset))

    expected_online_samples = [expected_online] if expected_online else []
    expected_demo_samples = [expected_demo] if expected_demo else []
    assert replay_buffer.sample_sizes == expected_online_samples
    assert demo_buffer.sample_sizes == expected_demo_samples
    assert batch["source"].tolist() == expected_source


def test_f1_rlpd_config_is_real_ros2_and_uses_schema_contracts() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples/embodiment/config/realworld_f1_peg_rlpd_cnn_async.yaml"
    )
    text = config_path.read_text(encoding="utf-8")

    assert "loss_type: embodied_sac" in text
    assert "action_dim: 14" in text
    assert "image_num: 3" in text
    assert "backend: ros2" in text
    assert "action_schema: f1-normalized-action-v1" in text
    assert "max_num_steps: 10" in text
    assert "demo_fraction: 0.4" in text
    assert "load_path: ./data/f1-peg-demo-buffer" in text
    assert "online_manifest_path" not in text
    assert "demo_manifest_path" not in text
    assert "F1_" not in text
    assert "is_dummy" not in text
    assert "architecture_smoke" not in text
    assert "/Users/" not in text

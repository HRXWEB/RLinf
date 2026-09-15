# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration coverage for the F1 Gym registration and Hydra config."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import create_controller, load_controller_config
from gymnasium.envs.registration import registry
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
REAL_CONFIG_NAME = "realworld_f1_peg_rlpd_cnn_async"
RIGHT_ARM_CONFIG_NAME = "realworld_f1_right_arm_peg_rlpd_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
RIGHT_ARM_ENV_ID = "F1RightArmPegInsertionEnv-v0"
LEGACY_ENV_ID = "F1DualArmPegInsertionEnv-v0"
FORBIDDEN_RUNTIME_TERMS = (
    "phase2_handoff",
    "gate_1_status",
    "runtime_lock",
    "site_approval_id",
    "operator_control",
    "architecture_smoke",
)
F1_DOCS = (
    ROOT / "docs" / "source-en" / "rst_source" / "examples" / "embodied" / "f1.rst",
    ROOT / "docs" / "source-zh" / "rst_source" / "examples" / "embodied" / "f1.rst",
    ROOT / "docs" / "source-en" / "rst_source" / "guides" / "f1_runtime.rst",
    ROOT / "docs" / "source-zh" / "rst_source" / "guides" / "f1_runtime.rst",
)


def test_f1_runtime_source_excludes_removed_phase_gate_terms() -> None:
    """F1 runtime code must not depend on removed Phase/Gate artifacts."""

    active_sources = [
        *sorted((ROOT / "toolkits" / "f1").glob("*.py")),
        ROOT / "rlinf" / "envs" / "realworld" / "f1" / "f1_robot_env.py",
    ]

    matches: list[str] = []
    for source in active_sources:
        text = source.read_text(encoding="utf-8")
        for term in FORBIDDEN_RUNTIME_TERMS:
            if term in text:
                matches.append(f"{source.relative_to(ROOT)} contains {term}")
    assert matches == []


def test_f1_docs_describe_the_public_training_workflow() -> None:
    """EN/ZH F1 docs describe the supported training workflow."""

    combined = "\n".join(path.read_text(encoding="utf-8") for path in F1_DOCS)
    required_terms = (
        "controller.backend",
        "action_scale",
        "motion_envelope",
        "f1-controller doctor",
        "f1-controller read-state",
        "RLINF_NODE_RANK=0",
        "RLINF_NODE_RANK=1",
        "ray start",
        "examples/embodiment/train_async.py",
        "realworld_f1_peg_rlpd_cnn_async",
        "OmegaConf.to_container",
        "temporary controller JSON",
        "临时 controller JSON",
        "demo_buffer.load_path",
        "~algorithm.demo_buffer",
        "requirements/install.sh f1-realworld",
        "F1_ROBOT_CONTROLLER_PACKAGE",
    )

    missing = [term for term in required_terms if term not in combined]
    assert missing == []


def test_f1_public_docs_exclude_development_only_workflows() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in F1_DOCS)
    forbidden = (
        "smoke",
        "pending",
        "supervised-online",
        "operator confirmation",
        "realworld_dummy_f1_peg_sac_cnn_async",
        "max_num_steps=1",
    )

    matches = [term for term in forbidden if term.lower() in combined.lower()]
    assert matches == []


def test_f1_docs_exclude_deleted_runtime_gates_and_path_envs() -> None:
    """Docs must not make old artifact gates or F1 path env vars mandatory."""

    forbidden = (
        "Phase",
        "Gate",
        "handoff",
        "runtime-lock",
        "runtime_lock",
        "F1_RUN_DIR",
        "F1_MODEL_PATH",
        "F1_CONTROLLER_CONFIG",
        "F1_MOTION_ENVELOPE",
    )
    matches: list[str] = []
    for path in F1_DOCS:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                matches.append(f"{path.relative_to(ROOT)} contains {term}")

    assert matches == []


def test_f1_tests_do_not_keep_stale_task_xfails() -> None:
    """Task-owned blanket xfails must not hide current F1 coverage."""

    matches: list[str] = []
    xfail_mark = "pytest.mark." + "xfail"
    xfail_call = "pytest." + "xfail("
    for path in sorted((ROOT / "tests" / "unit_tests" / "f1").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if xfail_mark in text or xfail_call in text:
            matches.append(str(path.relative_to(ROOT)))

    assert matches == []


def _controller_json_python_snippet(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "python - \"$controller_json\" <<'PY'"
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    end = next(
        index
        for index, line in enumerate(lines[start:], start=start)
        if line.strip() == "PY"
    )
    snippet_lines = [
        line[3:] if line.startswith("   ") else line for line in lines[start:end]
    ]
    return "\n".join(snippet_lines) + "\n"


def test_documented_controller_json_extraction_snippets_execute_and_load(
    tmp_path,
) -> None:
    """The exact docs snippets emit Controller 0.2.0 strict config JSON."""

    env = {
        **os.environ,
        "EMBODIED_PATH": str(ROOT / "examples" / "embodiment"),
    }
    for path in F1_DOCS:
        output = tmp_path / f"{path.parent.name}-{path.stem}.json"
        result = subprocess.run(
            [sys.executable, "-", str(output)],
            input=_controller_json_python_snippet(path),
            text=True,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert load_controller_config(json.loads(output.read_text(encoding="utf-8")))


def _compose_config(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name=CONFIG_NAME)


def _compose_named_config(monkeypatch: pytest.MonkeyPatch, config_name: str) -> Any:
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name=config_name)


def test_right_arm_training_config_matches_single_camera_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch model dimensions or calibration fields drifting from the Gym API."""

    cfg = _compose_named_config(monkeypatch, RIGHT_ARM_CONFIG_NAME)

    assert cfg.env.train.init_params.id == "F1RightArmPegInsertionEnv-v0"
    assert cfg.env.train.action_dim == 6
    assert cfg.actor.model.state_dim == 6
    assert cfg.actor.model.action_dim == 6
    assert cfg.actor.model.image_num == 1
    assert cfg.rollout.model.action_dim == 6
    assert cfg.rollout.model.image_num == 1
    assert cfg.cluster.component_placement.actor.node_group == "gpu"
    assert cfg.cluster.component_placement.rollout.node_group == "f1"
    assert cfg.cluster.component_placement.env.node_group == "f1"
    assert cfg.algorithm.demo_buffer.load_path == (
        "/data/hrx/datasets/f1_single_arm_peg_insertion_demo_buffer_10hz_v2"
    )
    assert cfg.rollout.enable_torch_compile is False
    assert cfg.env.train.override_cfg.tcp_reference_frame == "right_arm_tcp_pose"
    assert cfg.env.train.override_cfg.target_tcp_pose_m_deg == pytest.approx(
        [
            0.48393488343221785,
            -0.2951324389324688,
            0.06695701800144561,
            134.62856612383422,
            5.558924816706281,
            87.42882189672278,
        ]
    )
    assert cfg.env.train.override_cfg.action_scale.tcp_position_m == pytest.approx(
        0.01649676354589978
    )
    assert cfg.env.train.override_cfg.reset_left_joint_pose_deg == [
        90,
        -90,
        -90,
        -90,
        0,
        0,
        0,
    ]
    assert cfg.env.train.override_cfg.reset_right_joint_pose_deg == [
        -90,
        -90,
        90,
        -90,
        0,
        0,
        0,
    ]
    assert cfg.env.train.override_cfg.controller.ros2.command_topics.joint == (
        "/motion_ctl/joint_ctl"
    )
    assert cfg.env.train.override_cfg.action_scale.tcp_orientation_deg == pytest.approx(
        4.592777960973893
    )
    right_max_delta = cfg.env.train.override_cfg.motion_envelope.right_arm.tcp.max_delta
    assert right_max_delta.position_m == pytest.approx(0.02)
    assert right_max_delta.orientation_deg == pytest.approx(5.0)
    assert cfg.env.train.ignore_terminations is False
    assert cfg.env.train.max_episode_steps == 50
    assert cfg.env.train.override_cfg.max_num_steps == 50


def _approved_motion_envelope() -> dict[str, object]:
    arm = {
        "tcp": {
            "max_delta": {"position_m": 0.005, "orientation_deg": 1.0},
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


def _fake_controller_config() -> dict[str, object]:
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


def _action_scale() -> dict[str, float]:
    return {
        "tcp_position_m": 0.005,
        "tcp_orientation_deg": 1.0,
        "gripper_percent_closed": 10.0,
    }


def _assert_no_removed_f1_runtime_fields(payload: Any) -> None:
    container = json.dumps(payload, sort_keys=True)
    forbidden = (
        "oc.env:F1_",
        "is_dummy",
        "architecture_smoke",
        "operator_control",
        "phase2_handoff",
        "command_capability",
        "handoff",
        "phase",
    )
    assert [term for term in forbidden if term in container] == []


def _motion_envelope_file() -> Path:
    path = Path(tempfile.gettempdir()) / "rlinf_f1_motion_envelope_test.json"
    path.write_text(json.dumps(_approved_motion_envelope()), encoding="utf-8")
    return path


@pytest.fixture
def registered_f1(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    registry.pop(ENV_ID, None)
    registry.pop(RIGHT_ARM_ENV_ID, None)
    module = importlib.import_module("rlinf.envs.realworld.f1.tasks")
    importlib.reload(module)
    try:
        yield module
    finally:
        registry.pop(ENV_ID, None)
        registry.pop(RIGHT_ARM_ENV_ID, None)
        sys.modules.pop("rlinf.envs.realworld.f1.tasks", None)


def _make_env(
    *,
    mode: str,
    is_dummy: bool = True,
    operator_control: Mapping[str, object] | None = None,
    **task_overrides: object,
) -> gym.Env:
    del mode, is_dummy, operator_control
    return gym.make(
        ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": _approved_motion_envelope(),
            "max_num_steps": 10,
            **task_overrides,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={},
    )


def _record_controller_motion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_is_dummy: bool,
) -> tuple[list[str], list[str]]:
    env_module = sys.modules["rlinf.envs.realworld.f1.f1_robot_env"]
    motion_calls: list[str] = []
    close_calls: list[str] = []

    def controller_factory(config: Mapping[str, object]) -> object:
        controller = create_controller(config)
        original_submit = controller.submit_command
        original_reset = controller.submit_reset_command
        original_close = controller.close

        def submit_command(command: Any) -> Any:
            motion_calls.append("policy")
            return original_submit(command)

        def submit_reset_command(command: Any) -> Any:
            motion_calls.append("reset")
            return original_reset(command)

        def close() -> None:
            close_calls.append("close")
            original_close()

        monkeypatch.setattr(controller, "submit_command", submit_command)
        monkeypatch.setattr(controller, "submit_reset_command", submit_reset_command)
        monkeypatch.setattr(controller, "close", close)
        return controller

    monkeypatch.setattr(env_module, "create_controller", controller_factory)
    return motion_calls, close_calls


def test_hydra_composes_the_f1_train_and_eval_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_config(monkeypatch)
    container = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.cluster.num_nodes == 1
    assert dict(cfg.cluster.component_placement) == {
        "actor": 0,
        "env": 0,
        "rollout": 0,
    }
    assert cfg.runner.max_steps == 2
    assert cfg.algorithm.adv_type == "embodied_sac"
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert "demo_buffer" not in cfg.algorithm
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.entropy_tuning.target_entropy == -14
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.state_dim == 16
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.image_num == 3
    assert cfg.runner.only_eval is False
    assert cfg.rollout.model.precision == cfg.actor.model.precision
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    _assert_no_removed_f1_runtime_fields(container)

    for section in (cfg.env.train, cfg.env.eval):
        assert section.env_type == "realworld"
        assert section.action_dim == 14
        assert section.init_params.id == ENV_ID
        assert section.init_params.registration_module == (
            "rlinf.envs.realworld.f1.tasks"
        )
        assert section.video_cfg.save_video is False
        assert section.auto_reset is True
        assert section.max_episode_steps == 10
        assert section.max_steps_per_rollout_epoch == 10
        assert section.main_image_key == "head_color"
        assert "state_order" not in section
        assert section.override_cfg.controller.backend == "fake"
        assert section.override_cfg.action_scale.tcp_position_m == pytest.approx(0.005)
        assert section.override_cfg.action_scale.tcp_orientation_deg == pytest.approx(
            1.0
        )
        assert section.override_cfg.max_num_steps == 10


def test_real_f1_training_config_is_self_contained_ros2_rlpd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose_named_config(monkeypatch, REAL_CONFIG_NAME)
    container = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.component_placement.actor.node_group == "gpu"
    assert cfg.cluster.component_placement.rollout.node_group == "gpu"
    assert cfg.cluster.component_placement.env.node_group == "f1"
    assert cfg.cluster.node_groups[0].label == "gpu"
    assert cfg.cluster.node_groups[0].node_ranks == 0
    assert cfg.cluster.node_groups[1].label == "f1"
    assert cfg.cluster.node_groups[1].node_ranks == 1
    assert cfg.cluster.node_groups[1].env_configs[0].python_interpreter_path == (
        "/opt/rlinf-venv/bin/python"
    )
    assert cfg.runner.max_steps == 1
    assert cfg.runner.weight_sync_interval == 1
    assert cfg.runner.logger.log_path == "./logs/f1-peg-rlpd"
    assert cfg.algorithm.update_epoch == 1
    assert "demo_buffer" in cfg.algorithm
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.env.train.auto_reset is False
    assert cfg.env.train.max_episode_steps == 10
    assert cfg.env.train.init_params.registration_module == (
        "rlinf.envs.realworld.f1.tasks"
    )
    assert cfg.env.train.override_cfg.max_num_steps == 10
    assert cfg.env.train.override_cfg.controller.backend == "ros2"
    assert cfg.env.train.override_cfg.controller.control_period_s == pytest.approx(0.1)
    ros2 = cfg.env.train.override_cfg.controller.ros2
    assert dict(ros2.sensor_topics) == {
        "head_color": "/camera/head/color/image_raw/compressed",
        "left_wrist_color": "/camera/left_wrist/color/image_raw/compressed",
        "right_wrist_color": "/camera/right_wrist/color/image_raw/compressed",
        "joint_state": "/hal/joint_states",
        "left_gripper_position": "/motion_ctl/gripper/left/state",
        "right_gripper_position": "/motion_ctl/gripper/right/state",
    }
    assert dict(ros2.image_transports) == {
        "head_color": "compressed",
        "left_wrist_color": "compressed",
        "right_wrist_color": "compressed",
    }
    assert {source: list(shape) for source, shape in ros2.image_shapes.items()} == {
        "head_color": [720, 1280, 3],
        "left_wrist_color": [480, 848, 3],
        "right_wrist_color": [480, 848, 3],
    }
    assert list(ros2.left_joint_names) == [f"arm_l_j{idx}" for idx in range(1, 8)]
    assert list(ros2.right_joint_names) == [f"arm_r_j{idx}" for idx in range(1, 8)]
    assert ros2.joint_position_scale_to_rad == pytest.approx(0.017453292519943295)
    assert ros2.left_gripper_raw_open == 0.0
    assert ros2.left_gripper_raw_closed == 100.0
    assert ros2.right_gripper_raw_open == 0.0
    assert ros2.right_gripper_raw_closed == 100.0
    assert dict(ros2.command_topics) == {
        "left_tcp": "/motion_ctl/left_arm/tcp_pos_ctl",
        "right_tcp": "/motion_ctl/right_arm/tcp_pos_ctl",
        "left_gripper": "/motion_ctl/gripper/left",
        "right_gripper": "/motion_ctl/gripper/right",
    }
    assert dict(ros2.command_state_topics) == {
        "left_tcp": "/state/left_arm/tcp_pos",
        "right_tcp": "/state/right_arm/tcp_pos",
        "left_gripper": "/motion_ctl/gripper/left/state",
        "right_gripper": "/motion_ctl/gripper/right/state",
    }
    assert ros2.node_name == "f1_robot_controller"
    assert cfg.env.train.override_cfg.action_scale.tcp_position_m == pytest.approx(
        0.005
    )
    assert cfg.env.train.override_cfg.action_scale.tcp_orientation_deg == pytest.approx(
        1.0
    )
    assert (
        cfg.env.train.override_cfg.motion_envelope.left_arm.tcp.max_delta.position_m
        == pytest.approx(0.005)
    )
    assert cfg.env.train.override_cfg.motion_envelope.gripper_percent_closed == [
        0.0,
        100.0,
    ]
    for section in (cfg.env.train, cfg.env.eval):
        envelope = section.override_cfg.motion_envelope
        for arm in (envelope.left_arm, envelope.right_arm):
            assert arm.tcp.workspace_bounds.position_m.z.min == pytest.approx(-1.0)
            assert arm.tcp.workspace_bounds.position_m.z.max == pytest.approx(1.0)
    assert cfg.env.eval.override_cfg.max_num_steps == 10
    assert cfg.env.eval.override_cfg.controller.backend == "ros2"
    assert cfg.rollout.collect_transitions is True
    assert cfg.actor.model.action_dim == 14
    assert cfg.rollout.model.action_dim == 14
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    _assert_no_removed_f1_runtime_fields(container)


def test_real_f1_training_config_constructs_strict_controller_without_ros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical real YAML must satisfy Controller 0.2.0 before ROS opens."""

    from rlinf.envs.realworld.f1 import F1RobotConfig

    cfg = _compose_named_config(monkeypatch, REAL_CONFIG_NAME)
    override_cfg = OmegaConf.to_container(
        cfg.env.train.override_cfg,
        resolve=True,
    )

    robot_config = F1RobotConfig(**override_cfg)
    loaded_controller = load_controller_config(robot_config.controller)

    assert loaded_controller.backend.value == "ros2"
    assert loaded_controller.ros2 is not None
    assert loaded_controller.ros2.motion_enabled is True


def test_documented_controller_json_extraction_matches_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Docs keep one Hydra YAML but feed Controller CLI a controller JSON."""

    cfg = _compose_named_config(monkeypatch, REAL_CONFIG_NAME)
    override_cfg = OmegaConf.to_container(
        cfg.env.train.override_cfg,
        resolve=True,
    )
    controller_payload = dict(override_cfg["controller"])
    controller_payload["motion_envelope"] = override_cfg["motion_envelope"]
    output_path = tmp_path / "controller.json"
    output_path.write_text(
        json.dumps(controller_payload, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    loaded_controller = load_controller_config(
        json.loads(output_path.read_text(encoding="utf-8"))
    )

    assert loaded_controller.backend.value == "ros2"
    assert loaded_controller.ros2 is not None
    assert loaded_controller.ros2.command_topics["left_tcp"] == (
        "/motion_ctl/left_arm/tcp_pos_ctl"
    )


def test_each_gym_make_uses_direct_task_env_without_dynamic_operator_config(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    automatic = _make_env(mode="automatic")
    manual = _make_env(mode="manual")
    try:
        assert gym.spec(ENV_ID).kwargs == {}
        assert automatic.unwrapped.__class__ is manual.unwrapped.__class__
        assert automatic.unwrapped.config.max_num_steps == 10
        assert manual.unwrapped.config.max_num_steps == 10
    finally:
        automatic.close()
        manual.close()


@pytest.mark.parametrize(
    "make_kwargs",
    [
        {},
        {"env_cfg": {}},
        {"env_cfg": {"operator_control": None}},
    ],
    ids=["no-env-cfg", "missing-operator-control", "null-operator-control"],
)
def test_direct_gym_make_without_operator_control_uses_current_state_origin(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    make_kwargs: dict[str, object],
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )

    env = gym.make(
        ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": _approved_motion_envelope(),
            "max_num_steps": 10,
        },
        **make_kwargs,
    )
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert info["session_origin_state"].shape == (16,)
    finally:
        env.close()

    assert motion_calls == []
    assert close_calls == ["close"]


def test_direct_gym_rejects_removed_legacy_override_fields(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    with pytest.raises(TypeError, match="is_dummy"):
        gym.make(
            ENV_ID,
            override_cfg={
                "is_dummy": True,
                "motion_envelope": _approved_motion_envelope(),
            },
        )


def test_direct_fake_env_runs_reset_and_ten_step_horizon_without_ros(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    forbidden_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    env = _make_env(mode="automatic")
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)

        truncations = []
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.zeros(14, dtype=np.float32))
            assert terminated is False
            truncations.append(truncated)
        assert truncations == [False] * 9 + [True]
    finally:
        env.close()

    forbidden_after = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    assert forbidden_after == forbidden_before


def test_composed_config_runs_realworld_env_through_the_ten_step_horizon(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    cfg = _compose_config(monkeypatch)
    from rlinf.envs import get_env_cls

    env_cls = get_env_cls("realworld", cfg.env.train)

    def fail_if_legacy_setup_runs() -> None:
        raise AssertionError("F1 construction must not run legacy node setup")

    monkeypatch.setattr(
        env_cls,
        "realworld_setup",
        staticmethod(fail_if_legacy_setup_runs),
    )
    env = env_cls(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        observation, _ = env.reset(seed=11)
        assert tuple(observation["states"].shape) == (1, 16)
        assert tuple(observation["main_images"].shape) == (1, 128, 128, 3)
        assert tuple(observation["extra_view_images"].shape) == (
            1,
            2,
            128,
            128,
            3,
        )

        truncations = []
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(
                np.zeros((1, 14), dtype=np.float32)
            )
            assert terminated.tolist() == [False]
            truncations.append(truncated.tolist())
        assert truncations == [[False]] * 9 + [[True]]
        assert env.elapsed_steps.tolist() == [0]
        assert not any(
            name.startswith(
                (
                    "rlinf.envs.realworld.dosw1",
                    "rlinf.envs.realworld.franka",
                    "rlinf.envs.realworld.gim_arm",
                    "rlinf.envs.realworld.xsquare",
                )
            )
            for name in sys.modules
        )
    finally:
        env.close()


def test_manual_mode_never_approves_robot_reset_automatically(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    motion_calls, _ = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )
    env = _make_env(mode="manual")
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert motion_calls == []
    finally:
        env.close()


def test_stale_operator_env_cfg_cannot_spoof_direct_backend_config(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch, expected_is_dummy=True
    )
    env = gym.make(
        ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": _approved_motion_envelope(),
            "max_num_steps": 10,
        },
        env_cfg={"backend_is_fake": False},
    )
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert motion_calls == []
    finally:
        env.close()
    assert close_calls == ["close"]


@pytest.mark.parametrize(
    "operator_control",
    [
        {"mode": "unknown", "timeout_s": 0.0},
        {"mode": "automatic", "timeout_s": 0.0, "unexpected": True},
    ],
    ids=["unknown-mode", "unknown-field"],
)
def test_stale_operator_control_is_ignored_by_direct_gym(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operator_control: dict[str, object],
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )

    env = _make_env(mode="automatic", operator_control=operator_control)
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
    finally:
        env.close()

    assert motion_calls == []
    assert close_calls == ["close"]


def test_unknown_task_override_still_fails_closed(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    with pytest.raises(TypeError, match="unexpected_task_field"):
        _make_env(mode="automatic", unexpected_task_field=True)

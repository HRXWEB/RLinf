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

"""Fault-boundary integration coverage for the configured F1 environment."""

import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from f1_robot_controller import ControllerConfig, create_controller
from gymnasium.envs.registration import registry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
FORBIDDEN_MODULE_PREFIXES = ("rclpy", "cv_bridge")


@pytest.fixture
def configured_f1_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, Any, Any]]:
    """Build the composed RealWorld/F1/wrapper stack with installed Fake."""

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=CONFIG_NAME)

    realworld = importlib.import_module("rlinf.envs.realworld")
    tasks_module = importlib.import_module("rlinf.envs.realworld.f1.tasks")
    if ENV_ID not in registry:
        importlib.reload(tasks_module)
    env_module = importlib.import_module("rlinf.envs.realworld.f1.f1_robot_env")
    controllers = []

    def capture_installed_fake(
        *,
        is_dummy: bool,
        config: ControllerConfig,
    ) -> Any:
        controller = create_controller(is_dummy=is_dummy, config=config)
        controllers.append(controller)
        return controller

    monkeypatch.setattr(env_module, "create_controller", capture_installed_fake)
    env = realworld.RealWorldEnv(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        env.reset()
        assert len(controllers) == 1
        wrapper = env.env.envs[0]
        yield env, wrapper, controllers[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("inject_fault", "expected_cause"),
    [
        (
            lambda controller: controller.fault_injector.inject_stale_sensor(
                "head_color"
            ),
            "ObservationUnavailableError",
        ),
        (
            lambda controller: controller.fault_injector.inject_missing_sensor(
                "left_joint_position"
            ),
            "ObservationUnavailableError",
        ),
        (
            lambda controller: controller.fault_injector.reject_next_command(
                "injected command rejection"
            ),
            "CommandRejectedError",
        ),
        (
            lambda controller: controller.fault_injector.fail_executor_once(
                "injected executor fault"
            ),
            "ObservationUnavailableError",
        ),
    ],
    ids=("stale-camera", "missing-joint-state", "command-rejection", "executor-fault"),
)
def test_installed_fake_faults_fail_closed_without_exposing_a_transition(
    configured_f1_env: tuple[Any, Any, Any],
    inject_fault: Callable[[Any], None],
    expected_cause: str,
) -> None:
    """Catch a fault being downgraded into a replayable Gym transition."""

    env, wrapper, controller = configured_f1_env
    wrappers = importlib.import_module("rlinf.envs.realworld.f1.wrappers")
    inject_fault(controller)
    transition_sentinel = object()
    transition: object = transition_sentinel

    with pytest.raises(wrappers.EpisodeFaultError) as raised:
        transition = env.step(np.zeros((1, 16), dtype=np.float32))

    error = raised.value
    assert transition is transition_sentinel
    assert error.transition_valid is False
    assert error.quarantine is True
    assert wrapper.state is wrappers.EpisodeState.FAULT
    causes = []
    cause = error.__cause__
    while cause is not None:
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    assert expected_cause in causes
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def test_gpu_e2e_config_freezes_the_two_step_f1_sac_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch the GPU gate losing a required training or schema observable."""

    e2e_config_root = ROOT / "tests" / "e2e_tests" / "embodied"
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    with initialize_config_dir(config_dir=str(e2e_config_root)):
        cfg = compose(config_name="realworld_f1_dummy_sac_cnn")

    assert cfg.runner.max_steps == 2
    assert cfg.runner.weight_sync_interval == 1
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.update_epoch >= 1
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.state_dim == 16
    assert cfg.actor.model.action_dim == 16
    assert cfg.actor.model.image_num == 3
    assert cfg.algorithm.entropy_tuning.target_entropy == -16
    assert cfg.env.train.init_params.id == ENV_ID
    assert cfg.env.train.operator_control.mode == "automatic"
    assert cfg.env.train.override_cfg.is_dummy is True
    assert cfg.rollout.collect_transitions is True
    assert cfg.actor.model.model_path == "/path/to/RLinf-ResNet10-pretrained"


def test_blocking_weight_sync_counts_only_completed_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a failed or unfinished weight sync being reported as successful."""

    runner_module_name = "_f1_async_runner_under_test"
    embodied_runner_module = ModuleType("rlinf.runners.embodied_runner")

    class StubEmbodiedRunner:
        sync_error: Exception | None = None

        def update_rollout_weights(self) -> None:
            if self.sync_error is not None:
                raise self.sync_error

    embodied_runner_module.EmbodiedRunner = StubEmbodiedRunner
    scheduler_module = ModuleType("rlinf.scheduler")
    scheduler_module.Channel = object
    scheduler_module.WorkerGroupFuncResult = object
    metric_module = ModuleType("rlinf.utils.metric_utils")
    metric_module.compute_evaluate_metrics = lambda results: results
    runner_utils_module = ModuleType("rlinf.utils.runner_utils")
    runner_utils_module.check_progress = lambda *_args, **_kwargs: (False,) * 3
    monkeypatch.setitem(
        sys.modules,
        "rlinf.runners.embodied_runner",
        embodied_runner_module,
    )
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", scheduler_module)
    monkeypatch.setitem(sys.modules, "rlinf.utils.metric_utils", metric_module)
    monkeypatch.setitem(sys.modules, "rlinf.utils.runner_utils", runner_utils_module)
    module_path = ROOT / "rlinf" / "runners" / "async_embodied_runner.py"
    spec = importlib.util.spec_from_file_location(runner_module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, runner_module_name, module)
    spec.loader.exec_module(module)
    runner = module.AsyncEmbodiedRunner.__new__(module.AsyncEmbodiedRunner)
    runner._weight_sync_success_total = 0

    runner.update_rollout_weights(no_wait=False)
    assert runner._weight_sync_success_total == 1

    runner.sync_error = RuntimeError("injected sync failure")
    with pytest.raises(RuntimeError, match="injected sync failure"):
        runner.update_rollout_weights(no_wait=False)
    assert runner._weight_sync_success_total == 1

    class CompletedHandle:
        def done(self) -> bool:
            return True

        def wait(self) -> None:
            return None

    runner.logger = SimpleNamespace(info=lambda _message: None)
    runner._pending_rollout_weight_sync = (CompletedHandle(), CompletedHandle())
    assert runner._cleanup_pending_rollout_weight_sync(no_wait=True) is True
    assert runner._weight_sync_success_total == 2

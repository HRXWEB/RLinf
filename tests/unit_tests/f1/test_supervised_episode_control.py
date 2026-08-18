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

"""Behavioral coverage for supervised F1 episode control."""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import registry

ROOT = Path(__file__).resolve().parents[3]
F1_PACKAGE_DIR = ROOT / "rlinf" / "envs" / "realworld" / "f1"
WRAPPERS_DIR = ROOT / "rlinf" / "envs" / "realworld" / "f1" / "wrappers"
WRAPPERS_PACKAGE = "_f1_supervised_wrappers_under_test"
TASKS_PACKAGE = "rlinf.envs.realworld.f1.tasks"
TASK_ENV_ID = "F1DualArmPegInsertionEnv-v1"


def _namespace_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    return package


@pytest.fixture
def wrappers_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Load only the F1 wrappers package, without importing the RLinf stack."""

    package_init = WRAPPERS_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        WRAPPERS_PACKAGE,
        package_init,
        submodule_search_locations=[str(WRAPPERS_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the F1 wrappers package")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, WRAPPERS_PACKAGE, module)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(f"{WRAPPERS_PACKAGE}.supervised_episode_control", None)


class RecordingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Small Gym environment recording the wrapper boundary."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.on_reset: Any = None
        self.reset_error: BaseException | None = None
        self.step_error: BaseException | None = None
        self.close_error: BaseException | None = None
        self.terminated = False
        self.truncated = False
        self.on_step: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.reset_calls += 1
        if self.on_reset is not None:
            self.on_reset()
        if self.reset_error is not None:
            raise self.reset_error
        return np.array([0.25], dtype=np.float32), {
            "seed": seed,
            "options": options,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_calls += 1
        if self.on_step is not None:
            self.on_step()
        if self.step_error is not None:
            raise self.step_error
        return (
            np.array(action, dtype=np.float32, copy=True),
            0.0,
            self.terminated,
            self.truncated,
            {"step_calls": self.step_calls},
        )

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_operator_gate_delivers_explicit_events_without_polling_sleep(
    wrappers_module: ModuleType,
) -> None:
    gate = wrappers_module.OperatorGate()
    waiter_started = Event()
    received: list[object] = []

    def wait_for_start() -> None:
        waiter_started.set()
        received.append(
            gate.wait_for_event(
                wrappers_module.EpisodeState.WAITING_START,
                timeout_s=1.0,
            )
        )

    waiter = Thread(target=wait_for_start)
    waiter.start()
    assert waiter_started.wait(timeout=1.0)
    gate.submit(wrappers_module.OperatorEvent.START)
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert received == [wrappers_module.OperatorEvent.START]
    assert gate.poll(wrappers_module.EpisodeState.RUNNING) is None


def test_automatic_gate_requires_fake_backend_and_never_auto_acknowledges_fault(
    wrappers_module: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="Fake"):
        wrappers_module.AutomaticOperatorGate(backend_is_fake=False)

    gate = wrappers_module.AutomaticOperatorGate(backend_is_fake=True)

    assert gate.poll(wrappers_module.EpisodeState.WAITING_RESET) is (
        wrappers_module.OperatorEvent.RESET_APPROVED
    )
    assert gate.poll(wrappers_module.EpisodeState.WAITING_START) is (
        wrappers_module.OperatorEvent.START
    )
    assert gate.poll(wrappers_module.EpisodeState.FAULT) is None


def test_happy_path_exposes_linear_states_and_only_steps_while_running(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=1.0,
    )
    reset_states: list[object] = []
    env.on_reset = lambda: reset_states.append(wrapper.state)

    assert wrapper.state is wrappers_module.EpisodeState.INITIALIZING
    with pytest.raises(RuntimeError, match="RUNNING"):
        wrapper.step(np.zeros(1, dtype=np.float32))
    assert env.step_calls == 0

    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    observation, info = wrapper.reset(seed=7, options={"scene": "fixture"})

    assert reset_states == [wrappers_module.EpisodeState.RESETTING_ROBOT]
    assert wrapper.state is wrappers_module.EpisodeState.RUNNING
    np.testing.assert_array_equal(observation, np.array([0.25], dtype=np.float32))
    assert info == {"seed": 7, "options": {"scene": "fixture"}}

    transition = wrapper.step(np.array([0.5], dtype=np.float32))

    assert env.step_calls == 1
    assert transition[4] == {"step_calls": 1}
    assert wrapper.state is wrappers_module.EpisodeState.RUNNING


def _start_episode(
    wrappers_module: ModuleType,
    env: RecordingEnv,
    *,
    operator_timeout_s: float = 1.0,
) -> tuple[object, object]:
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=operator_timeout_s,
    )
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    wrapper.reset()
    return wrapper, gate


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(True, False), (False, True)],
    ids=["task-termination", "horizon"],
)
def test_real_terminal_transition_returns_to_waiting_reset(
    wrappers_module: ModuleType,
    terminated: bool,
    truncated: bool,
) -> None:
    env = RecordingEnv()
    env.terminated = terminated
    env.truncated = truncated
    wrapper, _ = _start_episode(wrappers_module, env)

    transition = wrapper.step(np.zeros(1, dtype=np.float32))

    assert transition[2:4] == (terminated, truncated)
    assert env.step_calls == 1
    assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET


def test_task_abort_is_a_trainable_failure_without_a_fake_transition(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    wrapper, gate = _start_episode(wrappers_module, env)
    gate.submit(wrappers_module.OperatorEvent.TASK_ABORT)

    with pytest.raises(wrappers_module.EpisodeAbortedError) as raised:
        wrapper.step(np.zeros(1, dtype=np.float32))

    assert raised.value.event is wrappers_module.OperatorEvent.TASK_ABORT
    assert raised.value.quarantine is False
    assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET
    assert env.step_calls == 0


def test_safety_abort_faults_without_a_transition_and_requires_explicit_ack(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    wrapper, gate = _start_episode(wrappers_module, env, operator_timeout_s=0.0)
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)

    with pytest.raises(wrappers_module.EpisodeFaultError) as raised:
        wrapper.step(np.zeros(1, dtype=np.float32))

    assert raised.value.event is wrappers_module.OperatorEvent.SAFETY_ABORT
    assert raised.value.quarantine is True
    assert wrapper.state is wrappers_module.EpisodeState.FAULT
    assert env.step_calls == 0

    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    with pytest.raises(RuntimeError, match="invalid"):
        wrapper.reset()
    assert wrapper.state is wrappers_module.EpisodeState.FAULT
    assert env.reset_calls == 1

    gate.submit(wrappers_module.OperatorEvent.ACK_FAULT)
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    wrapper.reset()

    assert wrapper.state is wrappers_module.EpisodeState.RUNNING
    assert env.reset_calls == 2


def test_automatic_gate_cannot_recover_a_fault_without_an_explicit_ack(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.AutomaticOperatorGate(backend_is_fake=True)
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    wrapper.reset()
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)
    with pytest.raises(wrappers_module.EpisodeFaultError):
        wrapper.step(np.zeros(1, dtype=np.float32))

    with pytest.raises(TimeoutError, match="ack_fault"):
        wrapper.reset()

    assert wrapper.state is wrappers_module.EpisodeState.FAULT
    assert env.reset_calls == 1


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_underlying_failures_enter_fault_and_preserve_the_cause(
    wrappers_module: ModuleType,
    operation: str,
) -> None:
    env = RecordingEnv()
    failure = ValueError(f"{operation} exploded")
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )

    if operation == "reset":
        env.reset_error = failure
        gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    else:
        gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
        gate.submit(wrappers_module.OperatorEvent.START)
        wrapper.reset()
        env.step_error = failure

    with pytest.raises(wrappers_module.EpisodeFaultError) as raised:
        if operation == "reset":
            wrapper.reset()
        else:
            wrapper.step(np.zeros(1, dtype=np.float32))

    assert raised.value.__cause__ is failure
    assert raised.value.event is None
    assert raised.value.quarantine is True
    assert wrapper.state is wrappers_module.EpisodeState.FAULT


def test_each_operator_wait_times_out_in_place_without_an_idle_transition(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )

    with pytest.raises(TimeoutError, match="reset_approved"):
        wrapper.reset(seed=3)
    assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET
    assert env.reset_calls == 0
    assert env.step_calls == 0

    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    with pytest.raises(TimeoutError, match="start"):
        wrapper.reset(seed=3, options={"reset": 1})
    assert wrapper.state is wrappers_module.EpisodeState.WAITING_START
    assert env.reset_calls == 1
    assert env.step_calls == 0

    gate.submit(wrappers_module.OperatorEvent.START)
    observation, info = wrapper.reset(seed=99, options={"must": "not rerun"})

    np.testing.assert_array_equal(observation, np.array([0.25], dtype=np.float32))
    assert info == {"seed": 3, "options": {"reset": 1}}
    assert wrapper.state is wrappers_module.EpisodeState.RUNNING
    assert env.reset_calls == 1


def test_illegal_event_is_rejected_without_changing_state_or_touching_robot(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    gate.submit(wrappers_module.OperatorEvent.START)

    with pytest.raises(RuntimeError, match="invalid"):
        wrapper.reset()

    assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_close_is_idempotent_and_closed_is_terminal(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )

    wrapper.close()
    wrapper.close()

    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.reset()
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.step(np.zeros(1, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        gate.submit(wrappers_module.OperatorEvent.START)


def test_resetting_state_is_visible_at_the_linearized_robot_reset_boundary(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=1.0,
    )
    entered_reset = Event()
    release_reset = Event()
    result: list[object] = []

    def hold_reset() -> None:
        entered_reset.set()
        if not release_reset.wait(timeout=1.0):
            raise TimeoutError("test did not release robot reset")

    def run_reset() -> None:
        try:
            result.append(wrapper.reset())
        except BaseException as error:
            result.append(error)

    env.on_reset = hold_reset
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    reset_thread = Thread(target=run_reset)
    reset_thread.start()

    assert entered_reset.wait(timeout=1.0)
    assert wrapper.state is wrappers_module.EpisodeState.RESETTING_ROBOT
    assert env.step_calls == 0
    release_reset.set()
    reset_thread.join(timeout=1.0)

    assert not reset_thread.is_alive()
    assert len(result) == 1
    assert not isinstance(result[0], BaseException)
    assert wrapper.state is wrappers_module.EpisodeState.RUNNING


def test_gate_close_wakes_a_waiter_and_discards_queued_events(
    wrappers_module: ModuleType,
) -> None:
    gate = wrappers_module.OperatorGate()
    waiter_started = Event()
    received: list[object] = []

    def wait_for_event() -> None:
        waiter_started.set()
        received.append(
            gate.wait_for_event(
                wrappers_module.EpisodeState.WAITING_RESET,
                timeout_s=1.0,
            )
        )

    waiter = Thread(target=wait_for_event)
    waiter.start()
    assert waiter_started.wait(timeout=1.0)
    gate.close()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert received == [None]

    queued_gate = wrappers_module.OperatorGate()
    queued_gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    queued_gate.close()
    assert queued_gate.poll(wrappers_module.EpisodeState.WAITING_RESET) is None

    automatic_gate = wrappers_module.AutomaticOperatorGate(backend_is_fake=True)
    automatic_gate.close()
    assert automatic_gate.poll(wrappers_module.EpisodeState.WAITING_RESET) is None
    assert automatic_gate.poll(wrappers_module.EpisodeState.WAITING_START) is None


@pytest.mark.parametrize("timeout_s", [True, -1.0, np.inf])
def test_gate_rejects_unbounded_or_ambiguous_waits(
    wrappers_module: ModuleType,
    timeout_s: object,
) -> None:
    gate = wrappers_module.OperatorGate()

    with pytest.raises((TypeError, ValueError), match="timeout_s"):
        gate.wait_for_event(
            wrappers_module.EpisodeState.WAITING_RESET,
            timeout_s=timeout_s,
        )


def test_automatic_gate_validates_timeout_even_for_immediate_events(
    wrappers_module: ModuleType,
) -> None:
    gate = wrappers_module.AutomaticOperatorGate(backend_is_fake=True)

    with pytest.raises(ValueError, match="timeout_s"):
        gate.wait_for_event(
            wrappers_module.EpisodeState.WAITING_RESET,
            timeout_s=np.inf,
        )


def test_reset_is_rejected_while_running_without_touching_the_robot(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    wrapper, _ = _start_episode(wrappers_module, env)

    with pytest.raises(RuntimeError, match="RUNNING"):
        wrapper.reset()

    assert wrapper.state is wrappers_module.EpisodeState.RUNNING
    assert env.reset_calls == 1
    assert env.step_calls == 0


def test_safety_abort_while_waiting_start_enters_fault_without_policy_step(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    with pytest.raises(TimeoutError, match="start"):
        wrapper.reset()
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)

    with pytest.raises(wrappers_module.EpisodeFaultError) as raised:
        wrapper.reset()

    assert raised.value.event is wrappers_module.OperatorEvent.SAFETY_ABORT
    assert wrapper.state is wrappers_module.EpisodeState.FAULT
    assert env.reset_calls == 1
    assert env.step_calls == 0


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_base_exceptions_fail_closed_without_being_swallowed(
    wrappers_module: ModuleType,
    operation: str,
) -> None:
    env = RecordingEnv()
    interruption = KeyboardInterrupt(f"interrupt {operation}")
    gate = wrappers_module.OperatorGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    if operation == "reset":
        env.reset_error = interruption
        gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    else:
        gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
        gate.submit(wrappers_module.OperatorEvent.START)
        wrapper.reset()
        env.step_error = interruption

    with pytest.raises(KeyboardInterrupt) as raised:
        if operation == "reset":
            wrapper.reset()
        else:
            wrapper.step(np.zeros(1, dtype=np.float32))

    assert raised.value is interruption
    assert wrapper.state is wrappers_module.EpisodeState.FAULT


def test_close_failure_is_propagated_but_cannot_reopen_or_double_close(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    failure = ValueError("close exploded")
    env.close_error = failure
    wrapper, gate = _start_episode(wrappers_module, env)

    with pytest.raises(ValueError, match="close exploded") as raised:
        wrapper.close()
    wrapper.close()

    assert raised.value is failure
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        gate.submit(wrappers_module.OperatorEvent.START)


def test_concurrent_terminal_steps_issue_only_one_robot_action(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    env.truncated = True
    wrapper, _ = _start_episode(wrappers_module, env)
    entered_step = Event()
    release_step = Event()
    results: list[object] = []

    def hold_step() -> None:
        entered_step.set()
        if not release_step.wait(timeout=1.0):
            raise TimeoutError("test did not release robot step")

    def run_step() -> None:
        try:
            results.append(wrapper.step(np.zeros(1, dtype=np.float32)))
        except BaseException as error:
            results.append(error)

    env.on_step = hold_step
    first = Thread(target=run_step)
    second = Thread(target=run_step)
    first.start()
    assert entered_step.wait(timeout=1.0)
    second.start()
    release_step.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert env.step_calls == 1
    assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET
    assert len(results) == 2
    assert sum(isinstance(result, tuple) for result in results) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1


def test_automatic_gate_drives_the_installed_fake_task_to_its_real_horizon(
    wrappers_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    package_paths = {
        "rlinf": ROOT / "rlinf",
        "rlinf.envs": ROOT / "rlinf" / "envs",
        "rlinf.envs.realworld": ROOT / "rlinf" / "envs" / "realworld",
        "rlinf.envs.realworld.f1": F1_PACKAGE_DIR,
    }
    for name, path in package_paths.items():
        monkeypatch.setitem(sys.modules, name, _namespace_package(name, path))
    tasks_init = F1_PACKAGE_DIR / "tasks" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        TASKS_PACKAGE,
        tasks_init,
        submodule_search_locations=[str(tasks_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the F1 tasks package")
    tasks_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, TASKS_PACKAGE, tasks_module)
    spec.loader.exec_module(tasks_module)

    env = gym.make(
        TASK_ENV_ID,
        override_cfg={"control_period_s": 0.001},
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg=None,
    )
    gate = wrappers_module.AutomaticOperatorGate(
        backend_is_fake=env.unwrapped.config.is_dummy
    )
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    try:
        observation, _ = wrapper.reset()
        assert wrapper.observation_space.contains(observation)
        transitions = [wrapper.step(np.zeros(16, dtype=np.float32)) for _ in range(4)]
        assert [transition[3] for transition in transitions] == [
            False,
            False,
            False,
            True,
        ]
        assert wrapper.state is wrappers_module.EpisodeState.WAITING_RESET
    finally:
        wrapper.close()
        registry.pop(TASK_ENV_ID, None)

    forbidden_after = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    assert forbidden_after == forbidden_before

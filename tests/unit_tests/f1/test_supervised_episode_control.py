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
from threading import Barrier, Event, Lock, Thread
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
        self.close_error_once: BaseException | None = None
        self.on_close: Any = None
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
        if self.on_close is not None:
            self.on_close()
        if self.close_error_once is not None:
            error = self.close_error_once
            self.close_error_once = None
            raise error
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


@pytest.mark.parametrize(
    ("capacity", "error_type"),
    [(True, TypeError), (0, ValueError), (-1, ValueError), (1.0, TypeError)],
)
def test_operator_gate_requires_an_exact_positive_integer_capacity(
    wrappers_module: ModuleType,
    capacity: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="capacity must be"):
        wrappers_module.OperatorGate(capacity=capacity)


def test_gate_sequences_accepted_events_and_never_drops_on_capacity_overflow(
    wrappers_module: ModuleType,
) -> None:
    gate = wrappers_module.OperatorGate(capacity=2)

    assert gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED) == 1
    assert gate.submit(wrappers_module.OperatorEvent.START) == 2
    with pytest.raises(BufferError, match="capacity"):
        gate.submit(wrappers_module.OperatorEvent.TASK_ABORT)

    assert gate.poll(wrappers_module.EpisodeState.WAITING_RESET) is (
        wrappers_module.OperatorEvent.RESET_APPROVED
    )
    assert gate.poll(wrappers_module.EpisodeState.WAITING_START) is (
        wrappers_module.OperatorEvent.START
    )
    assert gate.submit(wrappers_module.OperatorEvent.TASK_ABORT) == 3


def test_concurrent_submitters_cannot_overbook_the_last_gate_slot(
    wrappers_module: ModuleType,
) -> None:
    gate = wrappers_module.OperatorGate(capacity=1)
    start = Barrier(3)
    results_lock = Lock()
    results: list[object] = []

    def submit(event: object) -> None:
        start.wait(timeout=1.0)
        try:
            result: object = gate.submit(event)
        except BaseException as error:
            result = error
        with results_lock:
            results.append(result)

    submitters = [
        Thread(target=submit, args=(wrappers_module.OperatorEvent.TASK_ABORT,)),
        Thread(target=submit, args=(wrappers_module.OperatorEvent.SAFETY_ABORT,)),
    ]
    for submitter in submitters:
        submitter.start()
    start.wait(timeout=1.0)
    for submitter in submitters:
        submitter.join(timeout=1.0)

    assert all(not submitter.is_alive() for submitter in submitters)
    assert len(results) == 2
    assert sum(result == 1 for result in results) == 1
    assert sum(isinstance(result, BufferError) for result in results) == 1
    assert gate.poll(wrappers_module.EpisodeState.RUNNING) in {
        wrappers_module.OperatorEvent.TASK_ABORT,
        wrappers_module.OperatorEvent.SAFETY_ABORT,
    }
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


@pytest.mark.parametrize(
    ("event_name", "error_name", "expected_state", "quarantine"),
    [
        ("TASK_ABORT", "EpisodeAbortedError", "WAITING_RESET", False),
        ("SAFETY_ABORT", "EpisodeFaultError", "FAULT", True),
    ],
    ids=["task-abort", "safety-abort"],
)
def test_abort_accepted_during_in_flight_step_discards_the_real_transition(
    wrappers_module: ModuleType,
    event_name: str,
    error_name: str,
    expected_state: str,
    quarantine: bool,
) -> None:
    env = RecordingEnv()
    wrapper, gate = _start_episode(wrappers_module, env)
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
    step_thread = Thread(target=run_step)
    step_thread.start()
    assert entered_step.wait(timeout=1.0)
    event = getattr(wrappers_module.OperatorEvent, event_name)
    gate.submit(event)
    release_step.set()
    step_thread.join(timeout=1.0)

    assert not step_thread.is_alive()
    assert len(results) == 1
    assert isinstance(results[0], getattr(wrappers_module, error_name))
    assert results[0].event is event
    assert results[0].quarantine is quarantine
    assert wrapper.state is getattr(wrappers_module.EpisodeState, expected_state)
    assert env.step_calls == 1


def test_safety_abort_takes_priority_over_an_earlier_task_abort(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    wrapper, gate = _start_episode(wrappers_module, env)
    gate.submit(wrappers_module.OperatorEvent.TASK_ABORT)
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)

    with pytest.raises(wrappers_module.EpisodeFaultError) as raised:
        wrapper.step(np.zeros(1, dtype=np.float32))

    assert raised.value.event is wrappers_module.OperatorEvent.SAFETY_ABORT
    assert wrapper.state is wrappers_module.EpisodeState.FAULT
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


def test_ack_queued_before_fault_boundary_cannot_recover_the_fault(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    wrapper, gate = _start_episode(wrappers_module, env, operator_timeout_s=0.0)
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)
    gate.submit(wrappers_module.OperatorEvent.ACK_FAULT)

    with pytest.raises(wrappers_module.EpisodeFaultError):
        wrapper.step(np.zeros(1, dtype=np.float32))
    with pytest.raises(TimeoutError, match="ack_fault"):
        wrapper.reset()

    assert wrapper.state is wrappers_module.EpisodeState.FAULT
    assert env.reset_calls == 1

    gate.submit(wrappers_module.OperatorEvent.ACK_FAULT)
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    wrapper.reset()
    assert wrapper.state is wrappers_module.EpisodeState.RUNNING
    assert env.reset_calls == 2


def test_ack_submitted_after_fault_is_visible_can_recover_the_fault(
    wrappers_module: ModuleType,
) -> None:
    fault_published = Event()
    release_locked_boundary = Event()
    ack_attempt = Barrier(2)
    ack_accepted = Event()

    class CoordinatedFaultGate(wrappers_module.OperatorGate):
        def submit(self, event: Any) -> int:
            if event is wrappers_module.OperatorEvent.ACK_FAULT:
                ack_attempt.wait(timeout=1.0)
            sequence = super().submit(event)
            if event is wrappers_module.OperatorEvent.ACK_FAULT:
                ack_accepted.set()
            return sequence

        def cursor(self) -> int:
            fault_published.set()
            if not ack_accepted.wait(timeout=1.0):
                raise TimeoutError("post-fault ACK was not accepted")
            return super().cursor()

        def _linearize_fault_boundary(self, publish: Any) -> bool:
            def publish_and_pause(boundary: int) -> bool:
                result = publish(boundary)
                fault_published.set()
                if not release_locked_boundary.wait(timeout=1.0):
                    raise TimeoutError("test did not release fault boundary")
                return result

            return super()._linearize_fault_boundary(publish_and_pause)

    env = RecordingEnv()
    gate = CoordinatedFaultGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=0.0,
    )
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    wrapper.reset()
    gate.submit(wrappers_module.OperatorEvent.SAFETY_ABORT)
    results: list[object] = []

    def trigger_fault() -> None:
        try:
            wrapper.step(np.zeros(1, dtype=np.float32))
        except BaseException as error:
            results.append(error)

    def acknowledge_visible_fault() -> None:
        gate.submit(wrappers_module.OperatorEvent.ACK_FAULT)

    fault_thread = Thread(target=trigger_fault)
    fault_thread.start()
    observed_fault = fault_published.wait(timeout=1.0)
    assert observed_fault
    assert wrapper.state is wrappers_module.EpisodeState.FAULT

    ack_thread = Thread(target=acknowledge_visible_fault)
    ack_thread.start()
    ack_attempt.wait(timeout=1.0)
    release_locked_boundary.set()
    fault_thread.join(timeout=1.0)
    ack_thread.join(timeout=1.0)

    assert not fault_thread.is_alive()
    assert not ack_thread.is_alive()
    assert ack_accepted.is_set()
    assert len(results) == 1
    assert isinstance(results[0], wrappers_module.EpisodeFaultError)

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


def test_automatic_decision_and_close_are_serialized_by_the_gate_condition(
    wrappers_module: ModuleType,
) -> None:
    decision_entered = Event()
    release_decision = Event()
    close_attempted = Event()
    close_finished = Event()
    results: list[object] = []

    class PausingAutomaticGate(wrappers_module.AutomaticOperatorGate):
        def _automatic_event_locked(self, state: object) -> object:
            decision_entered.set()
            if not release_decision.wait(timeout=1.0):
                raise TimeoutError("test did not release automatic decision")
            return super()._automatic_event_locked(state)

    gate = PausingAutomaticGate(backend_is_fake=True)

    def poll_reset() -> None:
        results.append(gate.poll(wrappers_module.EpisodeState.WAITING_RESET))

    def close_gate() -> None:
        close_attempted.set()
        gate.close()
        close_finished.set()

    poll_thread = Thread(target=poll_reset)
    close_thread = Thread(target=close_gate)
    poll_thread.start()
    assert decision_entered.wait(timeout=1.0)
    close_thread.start()
    assert close_attempted.wait(timeout=1.0)
    decision_was_serialized = not close_finished.is_set()
    release_decision.set()
    poll_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not poll_thread.is_alive()
    assert not close_thread.is_alive()
    assert decision_was_serialized
    assert results == [wrappers_module.OperatorEvent.RESET_APPROVED]
    assert close_finished.is_set()
    assert gate.poll(wrappers_module.EpisodeState.WAITING_RESET) is None


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


def test_close_failure_is_propagated_and_the_resource_close_is_retried(
    wrappers_module: ModuleType,
) -> None:
    env = RecordingEnv()
    failure = ValueError("close exploded")
    env.close_error = failure
    wrapper, gate = _start_episode(wrappers_module, env)

    with pytest.raises(ValueError, match="close exploded") as raised:
        wrapper.close()
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.close_calls == 1

    env.close_error = None
    wrapper.close()
    wrapper.close()

    assert raised.value is failure
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.close_calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        gate.submit(wrappers_module.OperatorEvent.START)


def test_concurrent_close_retries_after_the_first_resource_close_fails(
    wrappers_module: ModuleType,
) -> None:
    first_close_entered = Event()
    release_first_close = Event()
    results_lock = Lock()
    results: list[object] = []
    first_failure = ValueError("first close exploded")
    env = RecordingEnv()
    env.close_error_once = first_failure

    def hold_first_close() -> None:
        if env.close_calls == 1:
            first_close_entered.set()
            if not release_first_close.wait(timeout=1.0):
                raise TimeoutError("test did not release first resource close")

    env.on_close = hold_first_close
    wrapper, _ = _start_episode(wrappers_module, env)

    def run_close() -> None:
        try:
            result: object = wrapper.close()
        except BaseException as error:
            result = error
        with results_lock:
            results.append(result)

    first = Thread(target=run_close)
    second = Thread(target=run_close)
    first.start()
    assert first_close_entered.wait(timeout=1.0)
    second.start()
    release_first_close.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.close_calls == 2
    assert len(results) == 2
    assert sum(result is first_failure for result in results) == 1
    assert sum(result is None for result in results) == 1

    wrapper.close()
    assert env.close_calls == 2


def test_close_wakes_an_operator_wait_before_waiting_for_the_operation_lock(
    wrappers_module: ModuleType,
) -> None:
    wait_entered = Event()
    gate_closed = Event()
    reset_results: list[object] = []
    close_results: list[object] = []

    class SignalingGate(wrappers_module.OperatorGate):
        def _wait_for_accepted(self, *args: Any, **kwargs: Any) -> Any:
            wait_entered.set()
            return super()._wait_for_accepted(*args, **kwargs)

        def close(self) -> None:
            super().close()
            gate_closed.set()

    env = RecordingEnv()
    gate = SignalingGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=5.0,
    )

    def run_reset() -> None:
        try:
            reset_results.append(wrapper.reset())
        except BaseException as error:
            reset_results.append(error)

    def run_close() -> None:
        try:
            close_results.append(wrapper.close())
        except BaseException as error:
            close_results.append(error)

    reset_thread = Thread(target=run_reset)
    close_thread = Thread(target=run_close)
    reset_thread.start()
    assert wait_entered.wait(timeout=1.0)
    close_thread.start()
    close_linearized_early = gate_closed.wait(timeout=1.0)
    if not close_linearized_early:
        gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
        gate.submit(wrappers_module.OperatorEvent.START)
    reset_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_linearized_early
    assert not reset_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(reset_results) == 1
    assert isinstance(reset_results[0], RuntimeError)
    assert "closed" in str(reset_results[0])
    assert close_results == [None]
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.reset_calls == 0
    assert env.close_calls == 1


def test_close_during_underlying_reset_discards_the_reset_result(
    wrappers_module: ModuleType,
) -> None:
    entered_reset = Event()
    release_reset = Event()
    gate_closed = Event()
    reset_results: list[object] = []
    close_results: list[object] = []

    class SignalingGate(wrappers_module.OperatorGate):
        def close(self) -> None:
            super().close()
            gate_closed.set()

    def hold_reset() -> None:
        entered_reset.set()
        if not release_reset.wait(timeout=2.0):
            raise TimeoutError("test did not release robot reset")

    env = RecordingEnv()
    env.on_reset = hold_reset
    gate = SignalingGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=1.0,
    )
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)

    def run_reset() -> None:
        try:
            reset_results.append(wrapper.reset())
        except BaseException as error:
            reset_results.append(error)

    def run_close() -> None:
        try:
            close_results.append(wrapper.close())
        except BaseException as error:
            close_results.append(error)

    reset_thread = Thread(target=run_reset)
    close_thread = Thread(target=run_close)
    reset_thread.start()
    assert entered_reset.wait(timeout=1.0)
    close_thread.start()
    close_linearized_early = gate_closed.wait(timeout=1.0)
    release_reset.set()
    reset_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_linearized_early
    assert not reset_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(reset_results) == 1
    assert isinstance(reset_results[0], RuntimeError)
    assert "closed" in str(reset_results[0])
    assert close_results == [None]
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.reset_calls == 1
    assert env.close_calls == 1


def test_close_during_underlying_step_discards_the_real_transition(
    wrappers_module: ModuleType,
) -> None:
    entered_step = Event()
    release_step = Event()
    gate_closed = Event()
    step_results: list[object] = []
    close_results: list[object] = []

    class SignalingGate(wrappers_module.OperatorGate):
        def close(self) -> None:
            super().close()
            gate_closed.set()

    def hold_step() -> None:
        entered_step.set()
        if not release_step.wait(timeout=2.0):
            raise TimeoutError("test did not release robot step")

    env = RecordingEnv()
    env.on_step = hold_step
    gate = SignalingGate()
    wrapper = wrappers_module.SupervisedEpisodeControlWrapper(
        env,
        gate,
        operator_timeout_s=1.0,
    )
    gate.submit(wrappers_module.OperatorEvent.RESET_APPROVED)
    gate.submit(wrappers_module.OperatorEvent.START)
    wrapper.reset()

    def run_step() -> None:
        try:
            step_results.append(wrapper.step(np.zeros(1, dtype=np.float32)))
        except BaseException as error:
            step_results.append(error)

    def run_close() -> None:
        try:
            close_results.append(wrapper.close())
        except BaseException as error:
            close_results.append(error)

    step_thread = Thread(target=run_step)
    close_thread = Thread(target=run_close)
    step_thread.start()
    assert entered_step.wait(timeout=1.0)
    close_thread.start()
    close_linearized_early = gate_closed.wait(timeout=1.0)
    release_step.set()
    step_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_linearized_early
    assert not step_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(step_results) == 1
    assert isinstance(step_results[0], RuntimeError)
    assert "closed" in str(step_results[0])
    assert close_results == [None]
    assert wrapper.state is wrappers_module.EpisodeState.CLOSED
    assert env.step_calls == 1
    assert env.close_calls == 1


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

    env = tasks_module.DualArmPegInsertionEnv(
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

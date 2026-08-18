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

"""Operator-supervised episode boundaries for the F1 robot environment."""

from collections import deque
from enum import Enum
from math import isfinite
from threading import TIMEOUT_MAX, Condition, Lock, RLock
from time import monotonic
from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class EpisodeState(str, Enum):
    """Externally observable states of a supervised F1 episode."""

    INITIALIZING = "initializing"
    WAITING_RESET = "waiting_reset"
    RESETTING_ROBOT = "resetting_robot"
    WAITING_START = "waiting_start"
    RUNNING = "running"
    FAULT = "fault"
    CLOSED = "closed"


class OperatorEvent(str, Enum):
    """Events accepted from an explicit local operator-control adapter."""

    RESET_APPROVED = "reset_approved"
    START = "start"
    TASK_ABORT = "task_abort"
    SAFETY_ABORT = "safety_abort"
    ACK_FAULT = "ack_fault"


class EpisodeControlError(RuntimeError):
    """Base error carrying episode-control disposition metadata."""

    def __init__(
        self,
        message: str,
        *,
        state: EpisodeState,
        event: OperatorEvent | None,
        quarantine: bool,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.event = event
        self.quarantine = quarantine


class EpisodeAbortedError(EpisodeControlError):
    """Task abort that ends the episode without fabricating a transition."""

    def __init__(self, *, state: EpisodeState) -> None:
        super().__init__(
            "operator task abort ended the episode without a transition",
            state=state,
            event=OperatorEvent.TASK_ABORT,
            quarantine=False,
        )


class EpisodeFaultError(EpisodeControlError):
    """Safety or environment fault that must be quarantined and acknowledged."""

    def __init__(
        self,
        message: str,
        *,
        event: OperatorEvent | None,
    ) -> None:
        super().__init__(
            message,
            state=EpisodeState.FAULT,
            event=event,
            quarantine=True,
        )


def _validated_timeout(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be a real number")
    normalized = float(timeout_s)
    if not isfinite(normalized) or normalized < 0.0 or normalized > TIMEOUT_MAX:
        raise ValueError(f"timeout_s must be finite and in [0, {TIMEOUT_MAX}]")
    return normalized


class OperatorGate:
    """Thread-safe local event gate with bounded and nonblocking reads."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._events: deque[OperatorEvent] = deque()
        self._closed = False

    def submit(self, event: OperatorEvent) -> None:
        """Publish one explicit operator event to the episode controller."""

        if not isinstance(event, OperatorEvent):
            raise TypeError("event must be an OperatorEvent")
        with self._condition:
            if self._closed:
                raise RuntimeError("operator gate is closed")
            self._events.append(event)
            self._condition.notify()

    def poll(self, state: EpisodeState) -> OperatorEvent | None:
        """Return the next event immediately, or ``None`` if none is ready."""

        return self.wait_for_event(state, timeout_s=0.0)

    def wait_for_event(
        self,
        state: EpisodeState,
        *,
        timeout_s: float,
    ) -> OperatorEvent | None:
        """Wait at most ``timeout_s`` for the next explicit event."""

        if not isinstance(state, EpisodeState):
            raise TypeError("state must be an EpisodeState")
        timeout = _validated_timeout(timeout_s)
        deadline = monotonic() + timeout
        with self._condition:
            while not self._events and not self._closed:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._events:
                return self._events.popleft()
            return None

    def close(self) -> None:
        """Close the gate and wake any bounded waiter."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._events.clear()
            self._condition.notify_all()

    def _is_closed(self) -> bool:
        with self._condition:
            return self._closed


class AutomaticOperatorGate(OperatorGate):
    """Deterministic reset/start approval for explicit Fake-backend mode."""

    def __init__(self, *, backend_is_fake: bool) -> None:
        if type(backend_is_fake) is not bool:
            raise TypeError("backend_is_fake must be a bool")
        if not backend_is_fake:
            raise ValueError("automatic operator control requires the Fake backend")
        super().__init__()

    def wait_for_event(
        self,
        state: EpisodeState,
        *,
        timeout_s: float,
    ) -> OperatorEvent | None:
        """Emit deterministic reset/start events, but never acknowledge faults."""

        timeout = _validated_timeout(timeout_s)
        explicit_event = super().wait_for_event(state, timeout_s=0.0)
        if explicit_event is not None:
            return explicit_event
        if self._is_closed():
            return None
        if state is EpisodeState.WAITING_RESET:
            return OperatorEvent.RESET_APPROVED
        if state is EpisodeState.WAITING_START:
            return OperatorEvent.START
        return super().wait_for_event(state, timeout_s=timeout)


class SupervisedEpisodeControlWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Allow robot reset and policy actions only in supervised states."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        operator_gate: OperatorGate,
        *,
        operator_timeout_s: float,
    ) -> None:
        if not isinstance(operator_gate, OperatorGate):
            raise TypeError("operator_gate must be an OperatorGate")
        super().__init__(env)
        self._operator_gate = operator_gate
        self._operator_timeout_s = _validated_timeout(operator_timeout_s)
        self._operation_lock = Lock()
        self._state_lock = RLock()
        self._state = EpisodeState.INITIALIZING
        self._pending_reset_result: tuple[ObsType, dict[str, Any]] | None = None
        self._fault_reason: str | None = None

    @property
    def state(self) -> EpisodeState:
        """Return the current linearized episode state."""

        with self._state_lock:
            return self._state

    def _transition(self, state: EpisodeState) -> None:
        with self._state_lock:
            self._state = state

    @property
    def fault_reason(self) -> str | None:
        """Return the reason for the current fault, if any."""

        with self._state_lock:
            return self._fault_reason

    def _enter_fault(self, reason: str) -> None:
        with self._state_lock:
            self._fault_reason = reason
            self._state = EpisodeState.FAULT
        self._pending_reset_result = None

    def _wait_for(self, expected: OperatorEvent) -> None:
        state = self.state
        event = self._operator_gate.wait_for_event(
            state,
            timeout_s=self._operator_timeout_s,
        )
        if event is None:
            raise TimeoutError(
                f"timed out waiting for {expected.value} while in {state.value}"
            )
        if event is OperatorEvent.SAFETY_ABORT:
            reason = f"safety abort while in {state.value}"
            self._enter_fault(reason)
            raise EpisodeFaultError(
                reason,
                event=OperatorEvent.SAFETY_ABORT,
            )
        if event is not expected:
            raise EpisodeControlError(
                f"event {event.value} is invalid while in {state.value}; "
                f"expected {expected.value}",
                state=state,
                event=event,
                quarantine=False,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Wait for reset/start approval and return the measured reset result."""

        with self._operation_lock:
            if self.state is EpisodeState.CLOSED:
                raise RuntimeError("episode control is closed")
            if self.state is EpisodeState.INITIALIZING:
                self._transition(EpisodeState.WAITING_RESET)
            if self.state is EpisodeState.FAULT:
                self._wait_for(OperatorEvent.ACK_FAULT)
                with self._state_lock:
                    self._fault_reason = None
                    self._state = EpisodeState.WAITING_RESET
            if self.state is EpisodeState.RUNNING:
                raise RuntimeError("cannot reset while an episode is RUNNING")
            if self.state is EpisodeState.WAITING_RESET:
                self._wait_for(OperatorEvent.RESET_APPROVED)
                self._transition(EpisodeState.RESETTING_ROBOT)
                try:
                    self._pending_reset_result = self.env.reset(
                        seed=seed,
                        options=options,
                    )
                except BaseException as error:
                    reason = f"robot reset failed: {error}"
                    self._enter_fault(reason)
                    if isinstance(error, Exception):
                        raise EpisodeFaultError(reason, event=None) from error
                    raise
                self._transition(EpisodeState.WAITING_START)
            if self.state is EpisodeState.WAITING_START:
                self._wait_for(OperatorEvent.START)
                self._transition(EpisodeState.RUNNING)
            if self.state is not EpisodeState.RUNNING:
                raise RuntimeError(f"cannot reset while in {self.state.value}")
            if self._pending_reset_result is None:
                raise RuntimeError("reset result is unavailable")
            result = self._pending_reset_result
            self._pending_reset_result = None
            return result

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Apply an action only while the supervised episode is running."""

        with self._operation_lock:
            if self.state is EpisodeState.CLOSED:
                raise RuntimeError("episode control is closed")
            if self.state is not EpisodeState.RUNNING:
                raise RuntimeError(
                    f"policy step requires RUNNING state, got {self.state.value}"
                )
            event = self._operator_gate.poll(self.state)
            if event is OperatorEvent.TASK_ABORT:
                self._transition(EpisodeState.WAITING_RESET)
                raise EpisodeAbortedError(state=self.state)
            if event is OperatorEvent.SAFETY_ABORT:
                reason = "safety abort while in running"
                self._enter_fault(reason)
                raise EpisodeFaultError(
                    reason,
                    event=OperatorEvent.SAFETY_ABORT,
                )
            if event is not None:
                raise EpisodeControlError(
                    f"event {event.value} is invalid while in running",
                    state=self.state,
                    event=event,
                    quarantine=False,
                )
            try:
                transition = self.env.step(action)
                terminated = transition[2]
                truncated = transition[3]
            except BaseException as error:
                reason = f"policy step failed: {error}"
                self._enter_fault(reason)
                if isinstance(error, Exception):
                    raise EpisodeFaultError(reason, event=None) from error
                raise
            if terminated or truncated:
                self._transition(EpisodeState.WAITING_RESET)
            return transition

    def close(self) -> None:
        """Enter the terminal state and close the gate and environment once."""

        with self._operation_lock:
            if self.state is EpisodeState.CLOSED:
                return
            self._transition(EpisodeState.CLOSED)
            self._operator_gate.close()
            self.env.close()

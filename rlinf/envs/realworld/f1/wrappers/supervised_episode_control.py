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
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _AcceptedOperatorEvent:
    sequence: int
    event: OperatorEvent


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

    def __init__(self, *, capacity: int = 64) -> None:
        if type(capacity) is not int:
            raise TypeError("capacity must be an exact positive integer")
        if capacity <= 0:
            raise ValueError("capacity must be an exact positive integer")
        self._condition = Condition()
        self._capacity = capacity
        self._events: deque[_AcceptedOperatorEvent] = deque()
        self._last_sequence = 0
        self._closed = False

    def submit(self, event: OperatorEvent) -> int:
        """Publish one explicit operator event to the episode controller."""

        if not isinstance(event, OperatorEvent):
            raise TypeError("event must be an OperatorEvent")
        with self._condition:
            if self._closed:
                raise RuntimeError("operator gate is closed")
            if len(self._events) >= self._capacity:
                raise BufferError(
                    f"operator gate capacity {self._capacity} is exhausted"
                )
            sequence = self._last_sequence + 1
            self._events.append(_AcceptedOperatorEvent(sequence, event))
            self._last_sequence = sequence
            self._condition.notify()
            return sequence

    def cursor(self) -> int:
        """Return the sequence of the latest accepted explicit event."""

        with self._condition:
            return self._last_sequence

    def poll(self, state: EpisodeState) -> OperatorEvent | None:
        """Return the next event immediately, or ``None`` if none is ready."""

        accepted = self._wait_for_accepted(state, timeout_s=0.0)
        if accepted is None:
            return None
        return accepted.event

    def _poll_control_event(self, state: EpisodeState) -> OperatorEvent | None:
        accepted = self._wait_for_accepted(
            state,
            timeout_s=0.0,
            safety_priority=True,
        )
        if accepted is None:
            return None
        return accepted.event

    def wait_for_event(
        self,
        state: EpisodeState,
        *,
        timeout_s: float,
    ) -> OperatorEvent | None:
        """Wait at most ``timeout_s`` for the next explicit event."""

        accepted = self._wait_for_accepted(state, timeout_s=timeout_s)
        if accepted is None:
            return None
        return accepted.event

    def _wait_for_accepted(
        self,
        state: EpisodeState,
        *,
        timeout_s: float,
        after_sequence: int | None = None,
        safety_priority: bool = False,
    ) -> _AcceptedOperatorEvent | None:
        if not isinstance(state, EpisodeState):
            raise TypeError("state must be an EpisodeState")
        timeout = _validated_timeout(timeout_s)
        if after_sequence is not None and (
            type(after_sequence) is not int or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        deadline = monotonic() + timeout
        with self._condition:
            while True:
                accepted = self._pop_accepted_locked(
                    after_sequence=after_sequence,
                    safety_priority=safety_priority,
                )
                if accepted is not None:
                    return accepted
                if self._closed:
                    return None
                automatic_event = self._automatic_event_locked(state)
                if automatic_event is not None:
                    return _AcceptedOperatorEvent(
                        self._last_sequence,
                        automatic_event,
                    )
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def _pop_accepted_locked(
        self,
        *,
        after_sequence: int | None,
        safety_priority: bool,
    ) -> _AcceptedOperatorEvent | None:
        if after_sequence is not None and self._events:
            self._events = deque(
                accepted
                for accepted in self._events
                if accepted.sequence > after_sequence
            )
        if safety_priority:
            for index, accepted in enumerate(self._events):
                if accepted.event is OperatorEvent.SAFETY_ABORT:
                    del self._events[index]
                    return accepted
        if self._events:
            return self._events.popleft()
        return None

    def _automatic_event_locked(self, state: EpisodeState) -> OperatorEvent | None:
        del state
        return None

    def close(self) -> None:
        """Close the gate and wake any bounded waiter."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._events.clear()
            self._condition.notify_all()


class AutomaticOperatorGate(OperatorGate):
    """Deterministic reset/start approval for explicit Fake-backend mode."""

    def __init__(self, *, backend_is_fake: bool, capacity: int = 64) -> None:
        if type(backend_is_fake) is not bool:
            raise TypeError("backend_is_fake must be a bool")
        if not backend_is_fake:
            raise ValueError("automatic operator control requires the Fake backend")
        super().__init__(capacity=capacity)

    def _automatic_event_locked(self, state: EpisodeState) -> OperatorEvent | None:
        if state is EpisodeState.WAITING_RESET:
            return OperatorEvent.RESET_APPROVED
        if state is EpisodeState.WAITING_START:
            return OperatorEvent.START
        return None


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
        self._fault_event_boundary: int | None = None
        self._resource_closed = False

    @property
    def state(self) -> EpisodeState:
        """Return the current linearized episode state."""

        with self._state_lock:
            return self._state

    def _transition(self, state: EpisodeState) -> None:
        with self._state_lock:
            if self._state is EpisodeState.CLOSED and state is not EpisodeState.CLOSED:
                raise RuntimeError("episode control is closed")
            self._state = state

    def _raise_if_closed(self) -> None:
        if self.state is EpisodeState.CLOSED:
            raise RuntimeError("episode control is closed")

    @property
    def fault_reason(self) -> str | None:
        """Return the reason for the current fault, if any."""

        with self._state_lock:
            return self._fault_reason

    def _enter_fault(self, reason: str) -> bool:
        with self._state_lock:
            if self._state is EpisodeState.CLOSED:
                return False
            self._fault_reason = reason
            self._state = EpisodeState.FAULT
        self._pending_reset_result = None
        boundary = self._operator_gate.cursor()
        with self._state_lock:
            if self._state is not EpisodeState.FAULT:
                return False
            self._fault_event_boundary = boundary
            return True

    def _wait_for(self, expected: OperatorEvent) -> None:
        state = self.state
        after_sequence = None
        if state is EpisodeState.FAULT and expected is OperatorEvent.ACK_FAULT:
            after_sequence = self._fault_event_boundary
        accepted = self._operator_gate._wait_for_accepted(
            state,
            timeout_s=self._operator_timeout_s,
            after_sequence=after_sequence,
            safety_priority=True,
        )
        if accepted is None:
            self._raise_if_closed()
            raise TimeoutError(
                f"timed out waiting for {expected.value} while in {state.value}"
            )
        self._raise_if_closed()
        event = accepted.event
        if event is OperatorEvent.SAFETY_ABORT:
            reason = f"safety abort while in {state.value}"
            if not self._enter_fault(reason):
                self._raise_if_closed()
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
                    if self._state is EpisodeState.CLOSED:
                        raise RuntimeError("episode control is closed")
                    self._fault_reason = None
                    self._fault_event_boundary = None
                    self._state = EpisodeState.WAITING_RESET
            if self.state is EpisodeState.RUNNING:
                raise RuntimeError("cannot reset while an episode is RUNNING")
            if self.state is EpisodeState.WAITING_RESET:
                self._wait_for(OperatorEvent.RESET_APPROVED)
                self._transition(EpisodeState.RESETTING_ROBOT)
                try:
                    reset_result = self.env.reset(
                        seed=seed,
                        options=options,
                    )
                except BaseException as error:
                    if self.state is EpisodeState.CLOSED:
                        raise RuntimeError("episode control is closed") from error
                    reason = f"robot reset failed: {error}"
                    if not self._enter_fault(reason):
                        raise RuntimeError("episode control is closed") from error
                    if isinstance(error, Exception):
                        raise EpisodeFaultError(reason, event=None) from error
                    raise
                with self._state_lock:
                    if self._state is EpisodeState.CLOSED:
                        raise RuntimeError("episode control is closed")
                    self._pending_reset_result = reset_result
                    self._state = EpisodeState.WAITING_START
            if self.state is EpisodeState.WAITING_START:
                self._wait_for(OperatorEvent.START)
                self._transition(EpisodeState.RUNNING)
            with self._state_lock:
                if self._state is EpisodeState.CLOSED:
                    raise RuntimeError("episode control is closed")
                if self._state is not EpisodeState.RUNNING:
                    raise RuntimeError(f"cannot reset while in {self._state.value}")
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
            self._raise_for_running_event()
            try:
                transition = self.env.step(action)
                terminated = transition[2]
                truncated = transition[3]
            except BaseException as error:
                if self.state is EpisodeState.CLOSED:
                    raise RuntimeError("episode control is closed") from error
                reason = f"policy step failed: {error}"
                if not self._enter_fault(reason):
                    raise RuntimeError("episode control is closed") from error
                if isinstance(error, Exception):
                    raise EpisodeFaultError(reason, event=None) from error
                raise
            self._raise_if_closed()
            self._raise_for_running_event()
            with self._state_lock:
                if self._state is EpisodeState.CLOSED:
                    raise RuntimeError("episode control is closed")
                if terminated or truncated:
                    self._state = EpisodeState.WAITING_RESET
                return transition

    def _raise_for_running_event(self) -> None:
        self._raise_if_closed()
        event = self._operator_gate._poll_control_event(self.state)
        self._raise_if_closed()
        if event is OperatorEvent.TASK_ABORT:
            self._transition(EpisodeState.WAITING_RESET)
            raise EpisodeAbortedError(state=self.state)
        if event is OperatorEvent.SAFETY_ABORT:
            reason = "safety abort while in running"
            if not self._enter_fault(reason):
                self._raise_if_closed()
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

    def close(self) -> None:
        """Close logical control immediately and retry resource close as needed."""

        with self._state_lock:
            if self._resource_closed:
                return
            self._state = EpisodeState.CLOSED
            self._pending_reset_result = None
        self._operator_gate.close()
        with self._operation_lock:
            with self._state_lock:
                if self._resource_closed:
                    return
            self.env.close()
            with self._state_lock:
                self._resource_closed = True

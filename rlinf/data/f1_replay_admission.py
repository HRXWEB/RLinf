# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed F1 replay admission for online and demo trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from rlinf.data.embodied.f1_schema import (
    validate_f1_replay_descriptor,
    validate_f1_replay_descriptor_compatibility,
    validate_f1_transition,
)
from rlinf.data.embodied_io_struct import Trajectory


class F1ReplayAdmission:
    """Validate F1 trajectories before they enter replay or demo buffers.

    The descriptor and JSON transition records live on the trajectory boundary.
    They are not flattened by :class:`TrajectoryReplayBuffer`, so sampled tensor
    batches keep the existing low-overhead format.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        expected_descriptor: Mapping[str, Any] | None,
        source_type: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.expected_descriptor = (
            validate_f1_replay_descriptor(expected_descriptor)
            if enabled and expected_descriptor is not None
            else expected_descriptor
        )
        self.source_type = source_type
        self.quarantine_rejected = 0

        if self.enabled and self.expected_descriptor is None:
            raise ValueError("F1 replay admission requires an expected descriptor")

    @classmethod
    def from_config(
        cls,
        *,
        enabled: bool,
        descriptor: Mapping[str, Any] | None,
        source_type: str | None = None,
    ) -> "F1ReplayAdmission":
        """Create admission from the runtime replay descriptor."""

        return cls(
            enabled=enabled,
            expected_descriptor=descriptor,
            source_type=source_type,
        )

    def admit_online(self, trajectories: Iterable[Trajectory]) -> list[Trajectory]:
        """Validate trajectories intended for the online replay buffer."""

        return self.admit(trajectories, source_type=self.source_type or "online")

    def admit_demo(self, trajectories: Iterable[Trajectory]) -> list[Trajectory]:
        """Validate trajectories intended for the demo replay buffer."""

        return self.admit(trajectories, source_type=self.source_type or "demo")

    def admit(
        self, trajectories: Iterable[Trajectory], *, source_type: str
    ) -> list[Trajectory]:
        """Return accepted trajectories, raising before any unsafe insertion."""

        items = list(trajectories)
        if not self.enabled:
            return items

        for trajectory in items:
            self._validate_trajectory(trajectory, source_type=source_type)
        return items

    def metrics(self) -> dict[str, float]:
        """Return replay admission metrics."""

        return {"replay/quarantine_rejected": float(self.quarantine_rejected)}

    def _validate_trajectory(self, trajectory: Trajectory, *, source_type: str) -> None:
        descriptor = getattr(trajectory, "f1_manifest", None)
        if descriptor is None:
            raise ValueError("F1 replay trajectory is missing descriptor")

        candidate_descriptor = validate_f1_replay_descriptor(descriptor)
        if candidate_descriptor["source_type"] == "quarantine":
            self.quarantine_rejected += 1
            raise ValueError("quarantine trajectories cannot enter replay buffers")
        if candidate_descriptor["source_type"] != source_type:
            raise ValueError(
                "F1 replay trajectory source_type must match target buffer "
                f"{source_type!r}"
            )
        validate_f1_replay_descriptor_compatibility(
            candidate_descriptor,
            self.expected_descriptor,
        )

        transitions = getattr(trajectory, "f1_transitions", None)
        if not transitions:
            raise ValueError("F1 replay trajectory is missing transitions")

        audited_actions: list[list[float]] = []
        action_field: str | None = None
        for transition in transitions:
            try:
                validate_f1_transition(transition, candidate_descriptor)
            except ValueError as exc:
                if (
                    "quarantine" in str(exc)
                    or transition.get("source_type") == "quarantine"
                    or transition.get("fault") is not None
                    or transition.get("safety_abort") is True
                ):
                    self.quarantine_rejected += 1
                raise
            action_field = "action"
            audited_actions.append(transition[action_field])

        actions = getattr(trajectory, "actions", None)
        if actions is None:
            raise ValueError("F1 replay trajectory is missing tensor actions")
        action_tensor = torch.as_tensor(actions).detach().cpu().to(torch.float64)
        expected_actions = torch.tensor(audited_actions, dtype=torch.float64)

        rewards = getattr(trajectory, "rewards", None)
        if rewards is None:
            raise ValueError("F1 replay trajectory is missing rewards")
        reward_tensor = torch.as_tensor(rewards)
        if (
            action_tensor.ndim < 2
            or reward_tensor.ndim < 2
            or action_tensor.shape[-1] != expected_actions.shape[-1]
        ):
            raise ValueError("F1 replay tensor actions have an incompatible shape")
        executed_steps, batch_size = reward_tensor.shape[:2]
        if (
            action_tensor.shape[0] < executed_steps
            or action_tensor.shape[1] != batch_size
            or expected_actions.shape[0] != executed_steps * batch_size
        ):
            raise ValueError(
                "F1 replay tensor actions do not align one-to-one with transitions: "
                f"actions={tuple(action_tensor.shape)}, "
                f"rewards={tuple(reward_tensor.shape)}, "
                f"transitions={len(audited_actions)}"
            )
        action_tensor = action_tensor[:executed_steps].reshape(
            -1, action_tensor.shape[-1]
        )
        if action_tensor.shape != expected_actions.shape:
            raise ValueError(
                "F1 replay tensor actions do not align one-to-one with transitions"
            )
        if not torch.allclose(action_tensor, expected_actions, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"F1 replay tensor actions do not match transition {action_field}"
            )

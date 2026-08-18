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

"""Dual-arm peg-insertion task for the F1 real-world environment."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..f1_robot_env import F1RobotConfig, F1RobotEnv


@dataclass
class DualArmPegInsertionConfig(F1RobotConfig):
    """Frozen phase-one configuration for dual-arm peg insertion."""

    reset_joint_target_rad: tuple[float, ...] = (0.0,) * 14
    reset_gripper_target: tuple[float, float] = (0.5, 0.5)
    max_num_steps: int = 4


class DualArmPegInsertionEnv(F1RobotEnv):
    """Coordinate both F1 arms to insert a peg into its matching hole."""

    def __init__(
        self,
        override_cfg: Mapping[str, Any] | None = None,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
        env_cfg: Any = None,
    ) -> None:
        """Create the task from RealWorldEnv-compatible Gym arguments.

        Args:
            override_cfg: Valid overrides for :class:`DualArmPegInsertionConfig`.
            worker_info: Reserved scheduler context for the future real backend.
            hardware_info: Reserved hardware context for the future real backend.
            env_idx: Reserved per-worker environment index.
            env_cfg: Reserved enclosing RealWorldEnv configuration.
        """

        del worker_info, hardware_info, env_idx, env_cfg
        if override_cfg is None:
            config_values: dict[str, Any] = {}
        elif isinstance(override_cfg, Mapping):
            config_values = dict(override_cfg)
        else:
            raise TypeError("override_cfg must be a mapping")
        super().__init__(DualArmPegInsertionConfig(**config_values))

    @property
    def task_description(self) -> str:
        """Return the policy-facing task instruction."""

        return "Use both arms cooperatively to insert the peg into the matching hole."

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        """Compute the phase-one sparse reward from one observation."""

        del observation
        return 0.0

    def _is_success(self, observation: dict[str, Any]) -> bool:
        """Compute phase-one task success from one observation."""

        del observation
        return False

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

"""Real-world environments with dependency-isolated lazy imports."""

from importlib import import_module
from typing import Any

from .f1 import tasks as f1_tasks

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "DualFrankaEnv": (".franka.dual_franka_env", "DualFrankaEnv"),
    "DualFrankaJointEnv": (
        ".franka.tasks.dual_franka_joint_env",
        "DualFrankaJointEnv",
    ),
    "DualFrankaJointRobotConfig": (
        ".franka.tasks.dual_franka_joint_env",
        "DualFrankaJointRobotConfig",
    ),
    "DualFrankaTCPEnv": (
        ".franka.tasks.dual_franka_tcp_env",
        "DualFrankaTCPEnv",
    ),
    "DualFrankaTCPRobotConfig": (
        ".franka.tasks.dual_franka_tcp_env",
        "DualFrankaTCPRobotConfig",
    ),
    "DualFrankaRobotConfig": (
        ".franka.dual_franka_env",
        "DualFrankaRobotConfig",
    ),
    "DOSW1Config": (".dosw1", "DOSW1Config"),
    "DOSW1Env": (".dosw1", "DOSW1Env"),
    "dosw1_tasks": (".dosw1.tasks", None),
    "FrankaEnv": (".franka", "FrankaEnv"),
    "FrankaRobotConfig": (".franka", "FrankaRobotConfig"),
    "FrankaRobotState": (".franka", "FrankaRobotState"),
    "franka_tasks": (".franka.tasks", None),
    "GimArmEnv": (".gim_arm", "GimArmEnv"),
    "GimArmRobotConfig": (".gim_arm", "GimArmRobotConfig"),
    "GimArmRobotState": (".gim_arm", "GimArmRobotState"),
    "gim_arm_tasks": (".gim_arm.tasks", None),
    "Turtle2Env": (".xsquare", "Turtle2Env"),
    "Turtle2RobotConfig": (".xsquare", "Turtle2RobotConfig"),
    "Turtle2RobotState": (".xsquare", "Turtle2RobotState"),
    "xsquare_tasks": (".xsquare.tasks", None),
    "RealWorldEnv": (".realworld_env", "RealWorldEnv"),
}


def __getattr__(name: str) -> Any:
    """Load legacy robot stacks only when their public export is requested."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eagerly and lazily available package attributes."""

    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DualFrankaEnv",
    "DualFrankaJointEnv",
    "DualFrankaJointRobotConfig",
    "DualFrankaTCPEnv",
    "DualFrankaTCPRobotConfig",
    "DualFrankaRobotConfig",
    "DOSW1Config",
    "DOSW1Env",
    "dosw1_tasks",
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "f1_tasks",
    "franka_tasks",
    "GimArmEnv",
    "GimArmRobotConfig",
    "GimArmRobotState",
    "gim_arm_tasks",
    "Turtle2Env",
    "Turtle2RobotConfig",
    "Turtle2RobotState",
    "xsquare_tasks",
    "RealWorldEnv",
]

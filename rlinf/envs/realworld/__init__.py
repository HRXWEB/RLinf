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

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "DualFrankaEnv": ("rlinf.envs.realworld.franka.dual_franka_env", "DualFrankaEnv"),
    "DualFrankaJointEnv": (
        "rlinf.envs.realworld.franka.tasks.dual_franka_joint_env",
        "DualFrankaJointEnv",
    ),
    "DualFrankaJointRobotConfig": (
        "rlinf.envs.realworld.franka.tasks.dual_franka_joint_env",
        "DualFrankaJointRobotConfig",
    ),
    "DualFrankaTCPEnv": (
        "rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env",
        "DualFrankaTCPEnv",
    ),
    "DualFrankaTCPRobotConfig": (
        "rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env",
        "DualFrankaTCPRobotConfig",
    ),
    "DualFrankaRobotConfig": (
        "rlinf.envs.realworld.franka.dual_franka_env",
        "DualFrankaRobotConfig",
    ),
    "DOSW1Config": ("rlinf.envs.realworld.dosw1", "DOSW1Config"),
    "DOSW1Env": ("rlinf.envs.realworld.dosw1", "DOSW1Env"),
    "dosw1_tasks": ("rlinf.envs.realworld.dosw1.tasks", None),
    "FrankaEnv": ("rlinf.envs.realworld.franka", "FrankaEnv"),
    "FrankaRobotConfig": ("rlinf.envs.realworld.franka", "FrankaRobotConfig"),
    "FrankaRobotState": ("rlinf.envs.realworld.franka", "FrankaRobotState"),
    "franka_tasks": ("rlinf.envs.realworld.franka.tasks", None),
    "GimArmEnv": ("rlinf.envs.realworld.gim_arm", "GimArmEnv"),
    "GimArmRobotConfig": ("rlinf.envs.realworld.gim_arm", "GimArmRobotConfig"),
    "GimArmRobotState": ("rlinf.envs.realworld.gim_arm", "GimArmRobotState"),
    "gim_arm_tasks": ("rlinf.envs.realworld.gim_arm.tasks", None),
    "Turtle2Env": ("rlinf.envs.realworld.xsquare", "Turtle2Env"),
    "Turtle2RobotConfig": ("rlinf.envs.realworld.xsquare", "Turtle2RobotConfig"),
    "Turtle2RobotState": ("rlinf.envs.realworld.xsquare", "Turtle2RobotState"),
    "xsquare_tasks": ("rlinf.envs.realworld.xsquare.tasks", None),
    "RealWorldEnv": ("rlinf.envs.realworld.realworld_env", "RealWorldEnv"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


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

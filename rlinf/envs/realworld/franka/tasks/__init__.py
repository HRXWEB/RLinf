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

from typing import Any, Mapping

import gymnasium as gym

from rlinf.envs.realworld.common.wrappers import (
    apply_dual_franka_joint_wrappers,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.franka.dual_franka_env import DualFrankaEnv as DualFrankaEnv
from rlinf.envs.realworld.franka.franka_env import FrankaEnv as FrankaEnv
from rlinf.envs.realworld.franka.tasks.bottle import BottleEnv as BottleEnv
from rlinf.envs.realworld.franka.tasks.dex_pnp import (
    DexpnpEnv as DexpnpEnv,
)
from rlinf.envs.realworld.franka.tasks.dual_franka_joint_env import (
    DualFrankaJointEnv as DualFrankaJointEnv,
)
from rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env import (
    DualFrankaTCPEnv as DualFrankaTCPEnv,
)
from rlinf.envs.realworld.franka.tasks.franka_bin_relocation import (
    FrankaBinRelocationEnv as FrankaBinRelocationEnv,
)
from rlinf.envs.realworld.franka.tasks.peg_insertion_env import (
    PegInsertionEnv as PegInsertionEnv,
)
from rlinf.envs.realworld.registration import register_legacy_task


def create_franka_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = FrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_dual_franka_joint_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = DualFrankaJointEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_dual_franka_joint_wrappers(env, env_cfg)


def create_dual_franka_tcp_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = DualFrankaTCPEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_dual_franka_joint_wrappers(env, env_cfg)


def create_peg_insertion_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = PegInsertionEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_franka_bin_relocation_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = FrankaBinRelocationEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_bottle_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = BottleEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_dexpnp_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = DexpnpEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


register_legacy_task(
    "FrankaEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_franka_env",
)

register_legacy_task(
    "DualFrankaJointEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_dual_franka_joint_env",
)

register_legacy_task(
    "DualFrankaTCPEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_dual_franka_tcp_env",
)

register_legacy_task(
    "PegInsertionEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_peg_insertion_env",
)

register_legacy_task(
    "FrankaBinRelocationEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_franka_bin_relocation_env",
)

register_legacy_task(
    "BottleEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_bottle_env",
)

register_legacy_task(
    "DexpnpEnv-v1",
    "rlinf.envs.realworld.franka.tasks:create_dexpnp_env",
)

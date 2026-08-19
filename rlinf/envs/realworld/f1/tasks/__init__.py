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

"""F1 task environments and Gymnasium registration."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec

from rlinf.envs.realworld.registration import register_exact

from ..wrappers import (
    AutomaticOperatorGate,
    OperatorGate,
    SupervisedEpisodeControlWrapper,
)
from .peg_insertion_env import (
    DualArmPegInsertionConfig,
    DualArmPegInsertionEnv,
)

_ENV_ID = "F1DualArmPegInsertionEnv-v1"
_ENTRY_POINT = "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
_WRAPPER_ENTRY_POINT = "rlinf.envs.realworld.f1.tasks:_make_supervised_episode_control"
_WRAPPER_SPEC = WrapperSpec(
    name="F1SupervisedEpisodeControl",
    entry_point=_WRAPPER_ENTRY_POINT,
    kwargs={},
)
_OPERATOR_CONTROL_FIELDS = frozenset({"mode", "timeout_s"})


def _operator_control_config(env: gym.Env) -> Mapping[str, Any]:
    """Return this Gym instance's required operator-control config."""

    spec = env.unwrapped.spec
    if spec is None:
        raise ValueError("F1 Gym instance has no EnvSpec")
    env_cfg = spec.kwargs.get("env_cfg")
    if env_cfg is None:
        raise ValueError("env_cfg with operator_control is required")
    if not isinstance(env_cfg, Mapping):
        raise ValueError("env_cfg must be a mapping")
    operator_control = env_cfg.get("operator_control")
    if operator_control is None:
        raise ValueError("operator_control is required")
    if not isinstance(operator_control, Mapping):
        raise ValueError("operator_control must be a mapping")
    actual_fields = set(operator_control)
    if actual_fields != _OPERATOR_CONTROL_FIELDS:
        raise ValueError("operator_control must contain exactly mode and timeout_s")
    return operator_control


def _make_supervised_episode_control(env: gym.Env) -> gym.Env:
    """Wrap one configured F1 task with its explicit operator gate."""

    try:
        operator_control = _operator_control_config(env)
        mode = operator_control["mode"]
        if mode == "automatic":
            backend_is_fake = env.unwrapped.config.is_dummy
            gate = AutomaticOperatorGate(backend_is_fake=backend_is_fake)
        elif mode == "manual":
            gate = OperatorGate()
        else:
            raise ValueError("operator_control.mode must be 'automatic' or 'manual'")
        return SupervisedEpisodeControlWrapper(
            env,
            gate,
            operator_timeout_s=operator_control["timeout_s"],
        )
    except BaseException:
        try:
            env.close()
        except BaseException:
            pass
        raise


register_exact(
    _ENV_ID,
    _ENTRY_POINT,
    allowed_entry_points=frozenset({_ENTRY_POINT}),
    additional_wrappers=(_WRAPPER_SPEC,),
)

__all__ = ["DualArmPegInsertionConfig", "DualArmPegInsertionEnv"]

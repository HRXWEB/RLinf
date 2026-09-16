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

from gymnasium.envs.registration import register, registry

from .peg_insertion_env import (
    DualArmPegInsertionConfig,
    DualArmPegInsertionEnv,
)
from .right_arm_fixed_reach_env import (
    RightArmFixedReachConfig,
    RightArmFixedReachEnv,
    RightArmPositionActionWrapper,
    create_right_arm_fixed_reach_env,
)
from .right_arm_peg_insertion_env import (
    RightArmActionWrapper,
    RightArmPegInsertionConfig,
    RightArmPegInsertionEnv,
    create_right_arm_peg_insertion_env,
)

_ENV_ID = "F1DualArmPegInsertionEnv-v1"
_RIGHT_ARM_ENV_ID = "F1RightArmPegInsertionEnv-v0"
_RIGHT_ARM_REACH_ENV_ID = "F1RightArmFixedReachEnv-v0"

if _ENV_ID not in registry:
    register(
        id=_ENV_ID,
        entry_point="rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv",
    )

if _RIGHT_ARM_ENV_ID not in registry:
    register(
        id=_RIGHT_ARM_ENV_ID,
        entry_point=(
            "rlinf.envs.realworld.f1.tasks:create_right_arm_peg_insertion_env"
        ),
    )

if _RIGHT_ARM_REACH_ENV_ID not in registry:
    register(
        id=_RIGHT_ARM_REACH_ENV_ID,
        entry_point="rlinf.envs.realworld.f1.tasks:create_right_arm_fixed_reach_env",
    )

__all__ = [
    "DualArmPegInsertionConfig",
    "DualArmPegInsertionEnv",
    "RightArmActionWrapper",
    "RightArmFixedReachConfig",
    "RightArmFixedReachEnv",
    "RightArmPositionActionWrapper",
    "RightArmPegInsertionConfig",
    "RightArmPegInsertionEnv",
    "create_right_arm_peg_insertion_env",
    "create_right_arm_fixed_reach_env",
]

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

_ENV_ID = "F1DualArmPegInsertionEnv-v1"

if _ENV_ID not in registry:
    register(
        id=_ENV_ID,
        entry_point="rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv",
    )

__all__ = ["DualArmPegInsertionConfig", "DualArmPegInsertionEnv"]

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

from rlinf.envs.realworld.registration import register_exact

from .peg_insertion_env import (
    DualArmPegInsertionConfig,
    DualArmPegInsertionEnv,
)

_ENV_ID = "F1DualArmPegInsertionEnv-v1"
_ENTRY_POINT = "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"


register_exact(
    _ENV_ID,
    _ENTRY_POINT,
    allowed_entry_points=frozenset({_ENTRY_POINT}),
)

__all__ = ["DualArmPegInsertionConfig", "DualArmPegInsertionEnv"]

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

"""Compatibility tests for the real-world no-autoreset vector wrapper."""

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv


class _TerminatingCounterEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 10, shape=(1,), dtype=np.int64)
    action_space = gym.spaces.Discrete(1)

    def __init__(self) -> None:
        self.count = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.count = 0
        return np.array([self.count]), {}

    def step(self, action):
        del action
        self.count += 1
        return np.array([self.count]), 1.0, self.count == 1, False, {}


def test_no_autoreset_sync_vector_env_supports_legacy_gymnasium_buffers() -> None:
    """Keep the F1 runtime's pinned Gymnasium 0.29 buffer contract working."""

    env = NoAutoResetSyncVectorEnv([_TerminatingCounterEnv])
    try:
        env.observations = env._observations
        env._terminateds = env._terminations
        env._truncateds = env._truncations
        del env._observations
        del env._terminations
        del env._truncations
        del env._env_obs

        first_observation, _, first_terminated, _, _ = env.step([0])
        second_observation, _, second_terminated, _, _ = env.step([0])

        assert first_observation.tolist() == [[1]]
        assert first_terminated.tolist() == [True]
        assert second_observation.tolist() == [[2]]
        assert second_terminated.tolist() == [False]
    finally:
        env.close()

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E402

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from toolkits.f1.publish_insertion_workspace import compute_workspace_geometry


def test_compute_workspace_geometry_handles_asymmetric_target_bounds():
    geometry = compute_workspace_geometry(
        target_position=(0.635837088, -0.110543916, 0.688607462),
        lower_offset=(-0.20, -0.05, -0.02),
        upper_offset=(0.20, 0.30, 0.10),
    )

    assert geometry.minimum == pytest.approx((0.435837088, -0.160543916, 0.668607462))
    assert geometry.maximum == pytest.approx((0.835837088, 0.189456084, 0.788607462))
    assert geometry.center == pytest.approx((0.635837088, 0.014456084, 0.728607462))
    assert geometry.scale == pytest.approx((0.40, 0.35, 0.12))


def test_compute_workspace_geometry_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="lower_offset"):
        compute_workspace_geometry(
            target_position=(0.0, 0.0, 0.0),
            lower_offset=(0.1, -0.1, -0.1),
            upper_offset=(0.0, 0.1, 0.1),
        )

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

"""F1 hardware metadata registration and fail-closed validation tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlinf.scheduler.cluster.config import ClusterConfig  # noqa: E402
from rlinf.scheduler.hardware import (  # noqa: E402
    HardwareResource,
    NodeHardwareConfig,
)
from rlinf.scheduler.hardware.robots import F1Config, F1HWInfo  # noqa: E402
from rlinf.scheduler.hardware.robots.f1 import F1Robot  # noqa: E402


def test_f1_config_is_registered_and_enumerates_explicit_metadata_only() -> None:
    """Catch F1 silently requiring env/runtime data during scheduling."""

    node_hw = NodeHardwareConfig(
        type="F1",
        configs=[{"node_rank": 1}],
    )

    assert len(node_hw.configs) == 1
    config = node_hw.configs[0]
    assert isinstance(config, F1Config)
    assert config.node_rank == 1

    resource = F1Robot.enumerate(node_rank=1, configs=node_hw.configs)

    assert isinstance(resource, HardwareResource)
    assert resource.type == "F1"
    assert len(resource.infos) == 1
    info = resource.infos[0]
    assert isinstance(info, F1HWInfo)
    assert info.type == "F1"
    assert info.model == "F1"
    assert info.config is config
    assert F1Robot.enumerate(node_rank=0, configs=node_hw.configs) is None


def test_f1_config_rejects_non_integer_node_rank() -> None:
    """F1 scheduling metadata must identify one unambiguous Ray node."""

    with pytest.raises(AssertionError, match="node_rank.*integer"):
        F1Config(node_rank=True)


def test_f1_enumeration_requires_explicit_configs_and_has_no_side_effect_imports() -> (
    None
):
    """Catch accidental ROS/controller imports or hardware probing during enum."""

    with pytest.raises(AssertionError, match="explicit F1 hardware configurations"):
        F1Robot.enumerate(node_rank=1, configs=None)

    script = textwrap.dedent(
        """
        import json
        import sys

        from rlinf.scheduler.hardware.robots.f1 import F1Config, F1Robot

        config = F1Config(node_rank=1)
        F1Robot.enumerate(node_rank=1, configs=[config])
        forbidden = [
            name for name in sys.modules
            if name == "rclpy"
            or name.startswith("rclpy.")
            or name == "cv_bridge"
            or name.startswith("cv_bridge.")
            or name == "f1_robot_controller"
            or name.startswith("f1_robot_controller.")
        ]
        print(json.dumps(forbidden))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_f1_cluster_config_fails_closed_for_missing_or_wrong_hardware() -> None:
    """Catch F1 metadata assigned outside its enclosing node group."""

    cfg = ClusterConfig.from_dict_cfg(
        DictConfig(
            {
                "num_nodes": 2,
                "component_placement": {"env": {"node_group": "f1", "placement": 0}},
                "node_groups": [
                    {
                        "label": "f1",
                        "node_ranks": "1",
                        "hardware": {
                            "type": "F1",
                            "configs": [{"node_rank": 1}],
                        },
                    }
                ],
            }
        )
    )
    assert cfg.node_groups[0].hardware_type == "F1"
    assert cfg.get_node_hw_configs_by_rank(1) == cfg.node_groups[0].hardware.configs

    with pytest.raises(AssertionError, match="within node_ranks"):
        ClusterConfig.from_dict_cfg(
            DictConfig(
                {
                    "num_nodes": 2,
                    "component_placement": {
                        "env": {"node_group": "f1", "placement": 0}
                    },
                    "node_groups": [
                        {
                            "label": "f1",
                            "node_ranks": "1",
                            "hardware": {
                                "type": "F1",
                                "configs": [{"node_rank": 0}],
                            },
                        }
                    ],
                }
            )
        )

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

"""Pure metadata hardware registration for the F1 real-world robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class F1HWInfo(HardwareInfo):
    """Hardware information for one F1 robot metadata entry."""

    config: "F1Config"


@Hardware.register()
class F1Robot(Hardware):
    """F1 hardware policy that enumerates only explicit scheduler metadata."""

    HW_TYPE = "F1"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["F1Config"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate F1 resources from explicit config only.

        This method intentionally performs no ROS/controller imports, device scans,
        network probes, process inspection, or auto-detection.
        """
        assert configs is not None, (
            "Robot hardware requires explicit F1 hardware configurations"
        )
        robot_configs = [
            config
            for config in configs
            if isinstance(config, F1Config) and config.node_rank == node_rank
        ]
        if not robot_configs:
            return None
        return HardwareResource(
            type=cls.HW_TYPE,
            infos=[
                F1HWInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                for config in robot_configs
            ],
        )


@NodeHardwareConfig.register_hardware_config(F1Robot.HW_TYPE)
@dataclass
class F1Config(HardwareConfig):
    """Scheduler metadata declaring one F1 robot on a Ray node."""

    def __post_init__(self) -> None:
        """Require an exact integer Ray node rank."""
        if type(self.node_rank) is not int:
            raise AssertionError(
                f"'node_rank' in F1 config must be an integer. But got {type(self.node_rank)}."
            )

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

"""F1 canonical two-node placement smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlinf.scheduler.cluster.config import ClusterConfig  # noqa: E402
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo  # noqa: E402
from rlinf.scheduler.hardware import (  # noqa: E402
    Accelerator,
    HardwareInfo,
    HardwareResource,
)
from rlinf.scheduler.hardware.robots.f1 import F1Config, F1Robot  # noqa: E402
from rlinf.utils.placement import HybridComponentPlacement  # noqa: E402

CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_f1_peg_rlpd_cnn_async"


class FakeCluster:
    """Minimal cluster API used by component placement."""

    def __init__(self, nodes: list[NodeInfo], groups: dict[str, NodeGroupInfo]) -> None:
        self._nodes = nodes
        self._groups = groups

    def get_node_group(
        self, label: str | None = NodeGroupInfo.DEFAULT_GROUP_LABEL
    ) -> NodeGroupInfo:
        resolved = NodeGroupInfo.DEFAULT_GROUP_LABEL if label is None else str(label)
        assert resolved in self._groups, (
            f"Node group '{resolved}' not found. Available groups: {list(self._groups)}."
        )
        return self._groups[resolved]

    def get_node_info(self, node_rank: int) -> NodeInfo:
        return self._nodes[node_rank]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_accelerators(self) -> int:
        return sum(node.num_accelerators for node in self._nodes)


def _node(
    node_rank: int,
    *,
    accelerators: int = 0,
    f1_config: F1Config | None = None,
) -> NodeInfo:
    resources: list[HardwareResource] = []
    if accelerators:
        resources.append(
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[
                    HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")
                    for _ in range(accelerators)
                ],
            )
        )
    if f1_config is not None:
        f1_resource = F1Robot.enumerate(node_rank=node_rank, configs=[f1_config])
        assert f1_resource is not None
        resources.append(f1_resource)
    return NodeInfo(
        node_labels=[],
        node_rank=node_rank,
        ray_id=f"node-{node_rank}",
        node_ip=f"10.0.0.{node_rank + 1}",
        num_cpus=32,
        python_interpreter_path="/usr/bin/python3",
        default_env_vars={},
        env_vars={},
        hardware_resources=resources,
    )


def _cluster() -> FakeCluster:
    f1_config = F1Config(node_rank=1)
    nodes = [_node(0, accelerators=1), _node(1, f1_config=f1_config)]
    gpu_group = NodeGroupInfo(label="gpu", nodes=[nodes[0]])
    f1_group = NodeGroupInfo(label="f1", nodes=[nodes[1]], hardware_type="F1")
    default_group = NodeGroupInfo(label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes)
    node_group = NodeGroupInfo(
        label=NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL,
        nodes=nodes,
        ignore_hardware=True,
    )
    return FakeCluster(
        nodes,
        {
            "gpu": gpu_group,
            "f1": f1_group,
            default_group.label: default_group,
            node_group.label: node_group,
        },
    )


def _compose(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        return compose(config_name=CONFIG_NAME)


def test_real_training_config_composes_exact_two_node_f1_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch drift from the canonical rank0 GPU / rank1 F1 topology."""

    cfg = _compose(monkeypatch)
    cluster_cfg = ClusterConfig.from_dict_cfg(cfg.cluster)

    assert cfg.cluster.num_nodes == 2
    assert dict(cfg.cluster.component_placement) == {
        "actor": {"node_group": "gpu", "placement": 0},
        "rollout": {"node_group": "gpu", "placement": 0},
        "env": {"node_group": "f1", "placement": 0},
    }
    assert [(group.label, group.node_ranks) for group in cluster_cfg.node_groups] == [
        ("gpu", [0]),
        ("f1", [1]),
    ]
    assert set(cluster_cfg.node_groups[0].node_ranks).isdisjoint(
        cluster_cfg.node_groups[1].node_ranks
    )
    assert cluster_cfg.node_groups[0].hardware is None
    assert cluster_cfg.node_groups[1].hardware_type == "F1"
    assert len(cluster_cfg.get_node_hw_configs_by_rank(1)) == 1
    assert cluster_cfg.get_node_hw_configs_by_rank(1)[0].node_rank == 1


def test_actor_rollout_rank0_and_env_rank1_f1_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch Env accidentally placed on GPU rank0 or Actor/Rollout on F1 rank1."""

    cfg = _compose(monkeypatch)
    cluster = _cluster()

    placement = HybridComponentPlacement(cfg, cluster)

    actor = placement.get_strategy("actor").get_placement(cluster)
    rollout = placement.get_strategy("rollout").get_placement(cluster)
    env = placement.get_strategy("env").get_placement(cluster)

    assert [p.cluster_node_rank for p in actor] == [0]
    assert [p.node_group_label for p in actor] == ["gpu"]
    assert [p.cluster_node_rank for p in rollout] == [0]
    assert [p.node_group_label for p in rollout] == ["gpu"]
    assert [p.cluster_node_rank for p in env] == [1]
    assert [p.node_group_label for p in env] == ["f1"]
    assert env[0].local_hardware_ranks == [0]
    assert placement.get_world_size("env") == 1


def test_real_training_config_enters_bounded_policy_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch relabeling the real training config as a read-only probe."""

    cfg = _compose(monkeypatch)

    assert "f1_readonly_health_probe" not in cfg
    assert cfg.runner.task_type == "embodied"
    assert cfg.runner.max_steps == 1
    assert cfg.runner.only_eval is False
    assert cfg.rollout.collect_transitions is True
    assert cfg.env.train.auto_reset is False
    assert cfg.env.train.init_params.id == "F1DualArmPegInsertionEnv-v1"
    assert cfg.env.train.max_episode_steps == 10
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert cfg.env.train.init_params.registration_module == (
        "rlinf.envs.realworld.f1.tasks"
    )

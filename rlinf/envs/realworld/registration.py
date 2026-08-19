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

"""Lightweight legacy Gym registration and node-setup compatibility."""

import os
import pathlib
import time
from collections.abc import Callable
from dataclasses import fields
from importlib import import_module
from threading import Lock
from typing import Any, NamedTuple

from gymnasium.envs.registration import (
    EnvSpec,
    WrapperSpec,
    register,
    registry,
)

F1_ENV_ID = "F1DualArmPegInsertionEnv-v1"


class _LegacyRegistration(NamedTuple):
    actual_entry_point: str
    proxy_entry_point: str


_LEGACY_REGISTRATIONS = {
    "DOSW1PickEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.dosw1.tasks:create_dosw1_pick_env",
        "rlinf.envs.realworld.registration:create_dosw1_pick_env",
    ),
    "FrankaEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_franka_env",
        "rlinf.envs.realworld.registration:create_franka_env",
    ),
    "DualFrankaJointEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_dual_franka_joint_env",
        "rlinf.envs.realworld.registration:create_dual_franka_joint_env",
    ),
    "DualFrankaTCPEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_dual_franka_tcp_env",
        "rlinf.envs.realworld.registration:create_dual_franka_tcp_env",
    ),
    "PegInsertionEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_peg_insertion_env",
        "rlinf.envs.realworld.registration:create_peg_insertion_env",
    ),
    "FrankaBinRelocationEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_franka_bin_relocation_env",
        "rlinf.envs.realworld.registration:create_franka_bin_relocation_env",
    ),
    "BottleEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_bottle_env",
        "rlinf.envs.realworld.registration:create_bottle_env",
    ),
    "DexpnpEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.franka.tasks:create_dexpnp_env",
        "rlinf.envs.realworld.registration:create_dexpnp_env",
    ),
    "GimArmPegInsertionEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.gim_arm.tasks:GimArmPegInsertionEnv",
        "rlinf.envs.realworld.registration:create_gim_arm_peg_insertion_env",
    ),
    "ButtonEnv-v1": _LegacyRegistration(
        "rlinf.envs.realworld.xsquare.tasks:create_button_env",
        "rlinf.envs.realworld.registration:create_button_env",
    ),
}
_SETUP_LOCK = Lock()
_setup_complete = False


def register_exact(
    env_id: str,
    entry_point: str,
    *,
    allowed_entry_points: frozenset[str],
    additional_wrappers: tuple[WrapperSpec, ...] = (),
) -> None:
    """Register or validate an exact Gym spec without hiding conflicts."""

    expected = EnvSpec(
        id=env_id,
        entry_point=entry_point,
        additional_wrappers=additional_wrappers,
    )
    existing = registry.get(env_id)
    if existing is None:
        register(
            id=env_id,
            entry_point=entry_point,
            additional_wrappers=additional_wrappers,
        )
        return
    metadata_matches = all(
        getattr(existing, field.name) == getattr(expected, field.name)
        for field in fields(EnvSpec)
        if field.name != "entry_point"
    )
    if existing.entry_point not in allowed_entry_points or not metadata_matches:
        raise RuntimeError(f"{env_id} is already registered incompatibly")


def register_legacy_proxies() -> None:
    """Register dependency-free proxies for every established legacy Gym ID."""

    for env_id, registration in _LEGACY_REGISTRATIONS.items():
        register_exact(
            env_id,
            registration.proxy_entry_point,
            allowed_entry_points=frozenset(
                {
                    registration.proxy_entry_point,
                    registration.actual_entry_point,
                }
            ),
        )
        registry[env_id].entry_point = registration.proxy_entry_point


def register_legacy_task(env_id: str, entry_point: str) -> None:
    """Idempotently bind one legacy task to its exact known registration."""

    registration = _LEGACY_REGISTRATIONS.get(env_id)
    if registration is None or registration.actual_entry_point != entry_point:
        raise RuntimeError(f"{env_id} is not a known exact legacy registration")
    register_exact(
        env_id,
        entry_point,
        allowed_entry_points=frozenset(
            {
                registration.proxy_entry_point,
                registration.actual_entry_point,
            }
        ),
    )


def legacy_node_setup() -> None:
    """Terminate stale ROS master processes for a legacy real-world task."""

    import psutil
    from filelock import FileLock

    node_lock_file = "/tmp/.realworld.lock"
    if not os.path.exists(os.path.dirname(node_lock_file)):
        node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
    node_lock = FileLock(node_lock_file)

    with node_lock:
        ros_proc_names = {"roscore", "rosmaster", "rosout"}
        for proc in psutil.process_iter():
            if proc.name() in ros_proc_names:
                proc.kill()
                time.sleep(0.5)


def ensure_legacy_setup(setup: Callable[[], None] = legacy_node_setup) -> None:
    """Run legacy node setup exactly once in this process."""

    global _setup_complete
    with _SETUP_LOCK:
        if _setup_complete:
            return
        setup()
        _setup_complete = True


def _create_legacy_env(env_id: str, *args: Any, **kwargs: Any) -> Any:
    registration = _LEGACY_REGISTRATIONS[env_id]
    module_name, attribute = registration.actual_entry_point.split(":", 1)
    module = import_module(module_name)
    ensure_legacy_setup()
    factory = getattr(module, attribute)
    return factory(*args, **kwargs)


def create_dosw1_pick_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy DOSW1 pick task through the setup boundary."""

    return _create_legacy_env("DOSW1PickEnv-v1", *args, **kwargs)


def create_franka_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Franka task through the setup boundary."""

    return _create_legacy_env("FrankaEnv-v1", *args, **kwargs)


def create_dual_franka_joint_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy dual-Franka joint task through the setup boundary."""

    return _create_legacy_env("DualFrankaJointEnv-v1", *args, **kwargs)


def create_dual_franka_tcp_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy dual-Franka TCP task through the setup boundary."""

    return _create_legacy_env("DualFrankaTCPEnv-v1", *args, **kwargs)


def create_peg_insertion_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Franka peg task through the setup boundary."""

    return _create_legacy_env("PegInsertionEnv-v1", *args, **kwargs)


def create_franka_bin_relocation_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Franka bin task through the setup boundary."""

    return _create_legacy_env("FrankaBinRelocationEnv-v1", *args, **kwargs)


def create_bottle_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Franka bottle task through the setup boundary."""

    return _create_legacy_env("BottleEnv-v1", *args, **kwargs)


def create_dexpnp_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Franka dex-pnp task through the setup boundary."""

    return _create_legacy_env("DexpnpEnv-v1", *args, **kwargs)


def create_gim_arm_peg_insertion_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy GimArm peg task through the setup boundary."""

    return _create_legacy_env("GimArmPegInsertionEnv-v1", *args, **kwargs)


def create_button_env(*args: Any, **kwargs: Any) -> Any:
    """Create the legacy Turtle2 button task through the setup boundary."""

    return _create_legacy_env("ButtonEnv-v1", *args, **kwargs)

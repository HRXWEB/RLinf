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

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
REALWORLD_CONFIG_MODULES = {
    "ButtonEnv-v1": "rlinf.envs.realworld.xsquare.tasks",
    "BottleEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "DOSW1PickEnv-v1": "rlinf.envs.realworld.dosw1.tasks",
    "DualFrankaJointEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "DualFrankaTCPEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "DexpnpEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "FrankaBinRelocationEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "FrankaEnv-v1": "rlinf.envs.realworld.franka.tasks",
    "PegInsertionEnv-v1": "rlinf.envs.realworld.franka.tasks",
}


def _run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_shipped_realworld_configs_declare_registration_module():
    """Catch shipped real-world configs that cannot bootstrap their task ID."""
    config_paths = sorted(
        set((REPO_ROOT / "examples").glob("**/env/realworld_*.yaml"))
        | set((REPO_ROOT / "examples").glob("**/env/dosw1_pick.yaml"))
    )
    assert config_paths

    actual = {}
    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        if cfg.get("env_type") != "realworld":
            continue
        env_id = cfg.init_params.id
        actual[str(config_path.relative_to(REPO_ROOT))] = (
            env_id,
            cfg.init_params.get("registration_module"),
        )

    expected = {
        str(path.relative_to(REPO_ROOT)): (
            OmegaConf.load(path).init_params.id,
            REALWORLD_CONFIG_MODULES[OmegaConf.load(path).init_params.id],
        )
        for path in config_paths
        if OmegaConf.load(path).get("env_type") == "realworld"
    }
    assert actual == expected


def _import_realworld_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("psutil.process_iter", lambda: ())
    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    return RealWorldEnv


def _make_env_shell(
    tmp_path: Path, registration_module: str | None, monkeypatch: pytest.MonkeyPatch
):
    if registration_module == "selected_realworld_tasks":
        module_path = tmp_path / "selected_realworld_tasks.py"
        module_path.write_text(
            "import builtins\n"
            "builtins.REALWORLD_REGISTRATION_TEST_CALLS.append("
            "('registration_import', __name__)"
            ")\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))

    cfg = OmegaConf.create(
        {
            "init_params": {
                "id": "SelectedRealWorldEnv-v1",
            },
        }
    )
    if registration_module is not None:
        cfg.init_params.registration_module = registration_module

    realworld_env_cls = _import_realworld_env(monkeypatch)
    env = object.__new__(realworld_env_cls)
    env.cfg = cfg
    env.override_cfg = {}
    env.worker_info = SimpleNamespace(hardware_infos=[])
    return env


def test_create_env_imports_configured_registration_module_before_gym_make(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch removing the explicit registration import before env construction."""
    env = _make_env_shell(tmp_path, "selected_realworld_tasks", monkeypatch)
    calls = []
    monkeypatch.setattr(
        "builtins.REALWORLD_REGISTRATION_TEST_CALLS", calls, raising=False
    )

    def fake_make(**kwargs):
        calls.append(
            ("gym.make", kwargs["id"], "selected_realworld_tasks" in sys.modules)
        )
        assert calls == [
            ("registration_import", "selected_realworld_tasks"),
            ("gym.make", "SelectedRealWorldEnv-v1", True),
        ]
        return object()

    module = sys.modules.pop("selected_realworld_tasks", None)
    assert module is None
    monkeypatch.setattr("gymnasium.make", fake_make)

    env._create_env(env_idx=0)


def test_create_env_rejects_missing_registration_module_before_gym_make(
    monkeypatch: pytest.MonkeyPatch,
):
    """Catch falling through to gym.make when registration_module is omitted."""
    env = _make_env_shell(Path(), None, monkeypatch)
    monkeypatch.setattr(
        "gymnasium.make",
        lambda **_kwargs: pytest.fail(
            "gym.make must not run without registration_module"
        ),
    )

    with pytest.raises(ValueError, match="registration_module"):
        env._create_env(env_idx=0)


@pytest.mark.parametrize("registration_module", ["", "   ", 42])
def test_create_env_rejects_invalid_registration_module_before_gym_make(
    registration_module, monkeypatch: pytest.MonkeyPatch
):
    """Catch accepting blank or non-string registration module values."""
    env = _make_env_shell(Path(), None, monkeypatch)
    env.cfg.init_params.registration_module = registration_module
    monkeypatch.setattr(
        "gymnasium.make",
        lambda **_kwargs: pytest.fail(
            "gym.make must not run with invalid module config"
        ),
    )

    with pytest.raises(ValueError, match="registration_module"):
        env._create_env(env_idx=0)


def test_create_env_reports_import_error_before_gym_make(
    monkeypatch: pytest.MonkeyPatch,
):
    """Catch hiding bad registration module paths behind Gym construction errors."""
    env = _make_env_shell(Path(), "rlinf.envs.realworld.no_such_tasks", monkeypatch)
    monkeypatch.setattr(
        "gymnasium.make",
        lambda **_kwargs: pytest.fail("gym.make must not run when module import fails"),
    )

    with pytest.raises(ImportError, match="registration module"):
        env._create_env(env_idx=0)


def test_realworld_package_root_does_not_eagerly_import_robot_stacks():
    """Catch root imports that load sibling robot stacks or perform ROS cleanup."""
    result = _run_fresh_python(
        """
import json
import psutil
import sys

cleanup_calls = []
psutil.process_iter = lambda: cleanup_calls.append("process_iter") or ()

import rlinf.envs.realworld as realworld

print(json.dumps({
    "has_realworld_env": hasattr(realworld, "RealWorldEnv"),
    "cleanup_calls": cleanup_calls,
    "loaded": sorted(
        name for name in sys.modules
        if name in {
            "rlinf.envs.realworld.franka.tasks",
            "rlinf.envs.realworld.xsquare.tasks",
            "rlinf.envs.realworld.gim_arm.tasks",
            "rlinf.envs.realworld.dosw1.tasks",
        }
    ),
}))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "has_realworld_env": True,
        "cleanup_calls": [],
        "loaded": [],
    }


def test_realworld_package_public_exports_are_lazy():
    """Catch removing documented public exports while making the package lazy."""
    result = _run_fresh_python(
        """
import json
import sys
import types

import rlinf.envs.realworld as realworld

before = "rlinf.envs.realworld.xsquare" in sys.modules
xsquare_module = types.ModuleType("rlinf.envs.realworld.xsquare")
xsquare_module.Turtle2Env = type("Turtle2Env", (), {})
sys.modules["rlinf.envs.realworld.xsquare"] = xsquare_module
export_name = realworld.Turtle2Env.__name__
after_selected = "rlinf.envs.realworld.xsquare" in sys.modules
after_sibling = "rlinf.envs.realworld.franka" in sys.modules

print(json.dumps({
    "before": before,
    "export_name": export_name,
    "after_selected": after_selected,
    "after_sibling": after_sibling,
}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "before": False,
        "export_name": "Turtle2Env",
        "after_selected": True,
        "after_sibling": False,
    }


@pytest.mark.parametrize(
    ("alias", "module_name", "cleanup_calls"),
    [
        (
            "franka_tasks",
            "rlinf.envs.realworld.franka.tasks",
            ["cleanup"],
        ),
        (
            "xsquare_tasks",
            "rlinf.envs.realworld.xsquare.tasks",
            ["cleanup"],
        ),
        (
            "dosw1_tasks",
            "rlinf.envs.realworld.dosw1.tasks",
            [],
        ),
        (
            "gim_arm_tasks",
            "rlinf.envs.realworld.gim_arm.tasks",
            [],
        ),
    ],
)
def test_realworld_package_task_aliases_import_task_modules(
    alias: str, module_name: str, cleanup_calls: list[str]
):
    """Catch package-root task aliases resolving against parent packages."""
    result = _run_fresh_python(
        f"""
import json
import sys
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv

torch.Event = object
cleanup_calls = []
RealWorldEnv.realworld_setup = staticmethod(lambda: cleanup_calls.append("cleanup"))

from rlinf.envs.realworld import {alias}

print(json.dumps({{
    "module": {alias}.__name__,
    "cleanup_calls": cleanup_calls,
    "loaded_siblings": sorted(
        name for name in sys.modules
        if name in {{
            "rlinf.envs.realworld.franka.tasks",
            "rlinf.envs.realworld.xsquare.tasks",
            "rlinf.envs.realworld.gim_arm.tasks",
            "rlinf.envs.realworld.dosw1.tasks",
        }}
        and name != {module_name!r}
    ),
}}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "module": module_name,
        "cleanup_calls": cleanup_calls,
        "loaded_siblings": [],
    }


def test_existing_ros1_registration_module_owns_historical_cleanup():
    """Catch moving ROS1 cleanup out of explicitly loaded ROS1 task modules."""
    result = _run_fresh_python(
        """
import json
import sys
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv

torch.Event = object
cleanup_calls = []
RealWorldEnv.realworld_setup = staticmethod(lambda: cleanup_calls.append("cleanup"))

import rlinf.envs.realworld.franka.tasks  # noqa: F401

print(json.dumps({
    "cleanup_calls": cleanup_calls,
    "xsquare_loaded": "rlinf.envs.realworld.xsquare.tasks" in sys.modules,
}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "cleanup_calls": ["cleanup"],
        "xsquare_loaded": False,
    }


def test_non_ros1_registration_module_does_not_run_ros_cleanup():
    """Catch applying historical ROS1 cleanup to unrelated registration modules."""
    result = _run_fresh_python(
        """
import json
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv

torch.Event = object
cleanup_calls = []
RealWorldEnv.realworld_setup = staticmethod(lambda: cleanup_calls.append("cleanup"))

import rlinf.envs.realworld.dosw1.tasks  # noqa: F401

print(json.dumps({"cleanup_calls": cleanup_calls}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cleanup_calls": []}

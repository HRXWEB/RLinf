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

"""Shell compatibility tests for the F1 pre-Ray setup script."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SETUP_SCRIPT = ROOT / "ray_utils" / "realworld" / "f1" / "setup_before_ray.sh"


def _write_runtime_fixture(tmp_path: Path, ros_setup_body: str) -> tuple[Path, Path]:
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "activate").write_text(
        "F1_FAKE_ACTIVATED=1\nexport F1_FAKE_ACTIVATED\n",
        encoding="utf-8",
    )
    fake_python = fake_venv / "bin" / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-" ]; then cat >/dev/null; exit 0; fi\n'
        'if [ "$1" = "-c" ]; then\n'
        '  case "$2" in\n'
        '    *platform.python_version*) printf "3.12.3\\n"; exit 0;;\n'
        '    *ray.__version__*) printf "2.57.0\\n"; exit 0;;\n'
        "    *) exit 0;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    ros_setup = tmp_path / "setup.bash"
    ros_setup.write_text(ros_setup_body, encoding="utf-8")
    return fake_venv, ros_setup


def _nounset_contract(shell_name: str, nounset_enabled: bool) -> str:
    if shell_name == "zsh":
        before = "setopt nounset;" if nounset_enabled else "unsetopt nounset;"
        after = (
            "[[ -o nounset ]]"
            if nounset_enabled
            else "if [[ -o nounset ]]; then exit 13; fi"
        )
    else:
        before = "set -u;" if nounset_enabled else "set +u;"
        after = (
            'case "$-" in *u*) ;; *) exit 13;; esac'
            if nounset_enabled
            else 'case "$-" in *u*) exit 13;; *) ;; esac'
        )
    return f"{before} {after}"


def _run_source(
    *,
    fake_venv: Path,
    nounset_enabled: bool,
    ros_setup: Path,
    shell: tuple[str, ...],
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    shell_name = shell[0]
    nounset_before, nounset_after = _nounset_contract(
        shell_name, nounset_enabled
    ).split(";", 1)
    command = (
        f"{nounset_before}; "
        "export F1_RUNTIME_ROLE=thor; "
        f"export F1_VENV_PATH='{fake_venv}'; "
        "export RLINF_COMM_NET_DEVICES=eth0; "
        f"export F1_ROS_SETUP_PATH='{ros_setup}'; "
        f"source '{SETUP_SCRIPT}'; source_status=$?; "
        "printf 'after-source:%s\\n' \"$source_status\"; "
        f"{nounset_after}; "
        "printf 'parent-alive\\n'"
    )

    return subprocess.run(
        [*shell, "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "shell",
    [
        ("bash", "--noprofile", "--norc"),
        pytest.param(
            ("zsh", "-f"),
            marks=pytest.mark.skipif(
                shutil.which("zsh") is None, reason="zsh is not installed"
            ),
        ),
    ],
)
@pytest.mark.parametrize("nounset_enabled", [False, True])
def test_setup_sources_ros_with_original_nounset_restored(
    tmp_path: Path, shell: tuple[str, ...], nounset_enabled: bool
) -> None:
    """Catch ROS setup aborting when the caller sourced F1 setup with nounset."""
    fake_venv, ros_setup = _write_runtime_fixture(
        tmp_path,
        'printf "ros-setup-entered\\n"\n'
        ': "$F1_FAKE_UNBOUND_FROM_ROS"\n'
        "F1_FAKE_ROS_SOURCED=1\n"
        "export F1_FAKE_ROS_SOURCED\n",
    )

    result = _run_source(
        fake_venv=fake_venv,
        nounset_enabled=nounset_enabled,
        ros_setup=ros_setup,
        shell=shell,
        tmp_path=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "ros-setup-entered" in result.stdout
    assert "F1 runtime ready: role=thor" in result.stdout
    assert "after-source:0" in result.stdout
    assert "parent-alive" in result.stdout


@pytest.mark.parametrize(
    "shell",
    [
        ("bash", "--noprofile", "--norc"),
        pytest.param(
            ("zsh", "-f"),
            marks=pytest.mark.skipif(
                shutil.which("zsh") is None, reason="zsh is not installed"
            ),
        ),
    ],
)
@pytest.mark.parametrize("nounset_enabled", [False, True])
def test_setup_failure_keeps_parent_shell_alive_and_restores_nounset(
    tmp_path: Path, shell: tuple[str, ...], nounset_enabled: bool
) -> None:
    """Catch failed ROS setup killing the caller or leaking nounset changes."""
    fake_venv, ros_setup = _write_runtime_fixture(
        tmp_path,
        'printf "ros-setup-entered\\n"\nreturn 42\n',
    )

    result = _run_source(
        fake_venv=fake_venv,
        nounset_enabled=nounset_enabled,
        ros_setup=ros_setup,
        shell=shell,
        tmp_path=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "ros-setup-entered" in result.stdout
    assert "after-source:1" in result.stdout
    assert "parent-alive" in result.stdout

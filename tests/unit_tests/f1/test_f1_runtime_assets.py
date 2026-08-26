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

"""Static contract tests for the simplified F1 runtime image assets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT = ROOT / "requirements" / "install.sh"
CONSTRAINTS = ROOT / "docker" / "f1" / "constraints" / "realworld-py312.txt"
DOCKERFILE = ROOT / "docker" / "f1" / "Dockerfile"
DOCKERIGNORE = ROOT / "docker" / "f1" / "Dockerfile.dockerignore"
CONTAINER_ENTRYPOINT = ROOT / "docker" / "f1" / "entrypoint.sh"
SETUP_SCRIPT = ROOT / "ray_utils" / "realworld" / "f1" / "setup_before_ray.sh"

RAY_VERSION = "2.57.0"
PYTHON_VERSION = "3.12.3"
CONTROLLER_VERSION = "0.2.0"
ROS_BASE_DIGEST = "2589a8fba5257307857890173c069852c2abf913a0be7970f172478baecb09e4"
UV_IMAGE_DIGEST = "a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _install_function_body() -> str:
    text = _read(INSTALL_SCRIPT)
    return text.split("install_f1_realworld_runtime() {", 1)[1].split(
        "\ninstall_flash_attn() {", 1
    )[0]


def test_f1_runtime_constraints_lock_the_shared_runtime() -> None:
    text = _read(CONSTRAINTS)

    assert f"ray[default]=={RAY_VERSION}" in text
    assert f"f1-robot-controller=={CONTROLLER_VERSION}" not in text
    assert "numpy==1.26.4" in text
    assert "gymnasium==0.29.1" in text
    assert "pillow==12.3.0" in text.lower()
    assert "imageio==2.37.4" in text.lower()
    assert "regex==2026.7.19" in text.lower()
    assert not re.search(r"\bray(?:\[default\])?(?:>=|~=|>|<)", text)
    assert not re.search(r"(?:nvidia-|cuda|cudnn)", text, re.I)


def test_f1_container_entrypoint_sources_ros_and_venv_before_exec(
    tmp_path: Path,
) -> None:
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "activate").write_text(
        "export F1_TEST_VENV=from-venv\n",
        encoding="utf-8",
    )
    ros_setup = tmp_path / "setup.bash"
    ros_setup.write_text(
        "export F1_TEST_ROS_ENV=from-ros-setup\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(CONTAINER_ENTRYPOINT),
            "bash",
            "-c",
            'printf "%s|%s|%s" "$F1_TEST_ROS_ENV" "$F1_TEST_VENV" "$1"',
            "entrypoint-test",
            "argument with spaces",
        ],
        cwd=ROOT,
        env={
            "F1_ROS_SETUP_PATH": str(ros_setup),
            "F1_VENV_PATH": str(fake_venv),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "from-ros-setup|from-venv|argument with spaces"


def test_f1_install_script_runtime_target_is_plain_dependency_install() -> None:
    text = _read(INSTALL_SCRIPT)
    body = _install_function_body()
    help_result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"f1-realworld"' in text
    assert "f1-realworld" in help_result.stdout
    assert f'F1_RUNTIME_PYTHON_VERSION="{PYTHON_VERSION}"' in text
    assert f'F1_RUNTIME_RAY_VERSION="{RAY_VERSION}"' in text
    assert "F1_CONTROLLER_WHEEL" not in body
    assert "F1_CONTROLLER_HANDOFF" not in body
    assert "F1_PHASE1_RLINF_HANDOFF" not in body
    assert "F1_ROS2_MANIFEST" not in body
    assert "F1_RLINF_COMMIT" not in body
    assert "F1_RLINF_SOURCE_SHA256" not in body
    assert "sha256sum" not in body
    assert "runtime_lock.py" not in body
    assert "git -C" not in body
    assert (
        'uv pip install -r "$repo_path/docker/f1/constraints/realworld-py312.txt"'
        in (body)
    )
    assert 'F1_ROBOT_CONTROLLER_PACKAGE="${F1_ROBOT_CONTROLLER_PACKAGE:-}"' in body
    assert "requires F1_ROBOT_CONTROLLER_PACKAGE" in body
    assert 'uv pip install --no-deps "$F1_ROBOT_CONTROLLER_PACKAGE"' in body
    assert 'uv pip install --no-deps -e "$repo_path"' in body
    assert 'uv pip install --no-deps "f1-robot-controller==0.2.0"' not in body
    assert "assert f1_robot_controller.__version__ == '0.2.0'" in body
    assert "shutil.which('f1-controller')" in body
    assert "F1_ROBOT_CONTROLLER_PACKAGE_URL" not in body


def test_thor_dockerfile_has_simplified_layer_contract() -> None:
    text = _read(DOCKERFILE)
    forbidden = (
        "HANDOFF",
        "PHASE1",
        "ROS2_MANIFEST_SHA256",
        "RLINF_SOURCE_SHA256",
        "runtime_lock.py",
        "source-commit.txt",
    )

    assert f"FROM ghcr.io/astral-sh/uv:0.8.15@sha256:{UV_IMAGE_DIGEST}" in text
    assert f"FROM ros:jazzy-ros-base@sha256:{ROS_BASE_DIGEST}" in text
    assert 'test "$TARGETARCH" = "arm64"' in text
    for apt_package in (
        "ros-jazzy-control-msgs",
        "ros-jazzy-cv-bridge",
        "ros-jazzy-diagnostic-msgs",
        "ros-jazzy-geometry-msgs",
        "ros-jazzy-map-msgs",
        "ros-jazzy-nav-msgs",
        "ros-jazzy-rmw-cyclonedds-cpp",
        "ros-jazzy-sensor-msgs",
        "ros-jazzy-tf2-msgs",
    ):
        assert apt_package in text
    for token in forbidden:
        assert token not in text
    assert "COPY . " not in text
    assert "COPY pyproject.toml README.md LICENSE /opt/rlinf/" in text
    assert "COPY rlinf /opt/rlinf/rlinf" in text
    assert "ARG F1_ROBOT_CONTROLLER_PACKAGE_URL" in text
    assert "ARG F1_ROBOT_CONTROLLER_PACKAGE\n" not in text
    assert "F1_ROBOT_CONTROLLER_PACKAGE build arg" not in text
    assert "package URL or pinned git URL" in text
    assert 'uv pip install --no-deps "$F1_ROBOT_CONTROLLER_PACKAGE_URL"' in text
    assert "COPY dist/" not in text
    assert "COPY *.whl" not in text
    assert 'uv pip install --no-deps "f1-robot-controller==0.2.0"' not in text
    assert "requirements/install.sh f1-realworld" in text
    assert "--no-deps" in text
    assert 'ENTRYPOINT ["/opt/rlinf/docker/f1/entrypoint.sh"]' in text
    assert 'CMD ["bash"]' in text
    assert not re.search(r"(?:nvidia|cuda|cudnn|nvcc)", text, re.I)
    assert not re.search(
        r"(?:ros2\s+launch|ros2\s+run|topic\s+pub|service\s+call)", text
    )


def test_thor_dockerfile_keeps_dependency_layers_source_stable() -> None:
    text = _read(DOCKERFILE)
    apt_layer = text.index('RUN test "$TARGETARCH" = "arm64"')
    constraint_copy = text.index(
        "COPY docker/f1/constraints/realworld-py312.txt",
        apt_layer,
    )
    dependency_install = text.index(
        "uv pip install -r /opt/rlinf/docker/f1/constraints/realworld-py312.txt",
        constraint_copy,
    )
    controller_arg = text.index(
        "ARG F1_ROBOT_CONTROLLER_PACKAGE_URL",
        dependency_install,
    )
    controller_install = text.index(
        'uv pip install --no-deps "$F1_ROBOT_CONTROLLER_PACKAGE_URL"',
        controller_arg,
    )
    source_copy = text.index(
        "COPY pyproject.toml README.md LICENSE", controller_install
    )
    rlinf_install = text.index("requirements/install.sh f1-realworld", source_copy)

    assert apt_layer < constraint_copy < dependency_install
    assert dependency_install < controller_arg < controller_install
    assert controller_install < source_copy < rlinf_install


def test_thor_dockerfile_build_time_imports_cover_runtime_modules() -> None:
    text = _read(DOCKERFILE)
    smoke = text[text.index("python -c ") :]

    for module in (
        "PIL",
        "imageio",
        "torch",
        "ray",
        "rlinf",
        "f1_robot_controller",
        "control_msgs",
        "cv_bridge",
        "diagnostic_msgs",
        "geometry_msgs",
        "map_msgs",
        "nav_msgs",
        "rclpy",
        "sensor_msgs",
        "tf2_msgs",
    ):
        assert module in smoke
    assert f"ray.__version__ == '{RAY_VERSION}'" in smoke


def test_thor_docker_context_is_an_explicit_runtime_allowlist() -> None:
    text = _read(DOCKERIGNORE)

    assert text.splitlines()[0] == "**"
    assert "!.git" not in text
    assert "!.venv" not in text
    assert "!logs" not in text
    assert "!.artifacts" not in text
    assert "!pyproject.toml" in text
    assert "!rlinf/**" in text
    assert "!requirements/install.sh" in text
    assert "!docker/f1/entrypoint.sh" in text
    assert "!docker/f1/constraints/realworld-py312.txt" in text
    assert "!ray_utils/realworld/f1/setup_before_ray.sh" in text
    assert "!toolkits/f1/runtime_lock.py" not in text


def test_setup_before_ray_only_prepares_environment_and_ray_identity() -> None:
    text = _read(SETUP_SCRIPT)

    assert f'F1_RUNTIME_PYTHON_VERSION="{PYTHON_VERSION}"' in text
    assert f'F1_RUNTIME_RAY_VERSION="{RAY_VERSION}"' in text
    assert "F1_RUNTIME_ROLE" in text
    assert "F1_VENV_PATH" in text
    assert "RLINF_NODE_RANK" in text
    assert "RLINF_COMM_NET_DEVICES" in text
    assert "platform.python_version()" in text
    assert "ray.__version__" in text
    assert 'source "$ros_setup_path"' in text
    assert "git -C" not in text
    assert "runtime_lock.py" not in text
    assert "runtime-lock.json" not in text
    assert "F1_RLINF_SOURCE_MARKER" not in text
    assert "F1_CONTROLLER_HANDOFF" not in text
    assert "sha256" not in text.lower()
    assert not re.search(
        r"(?:ray\s+start|ros2\s+launch|topic\s+pub|service\s+call)", text
    )


def test_setup_failure_does_not_mutate_existing_or_unset_rank(tmp_path: Path) -> None:
    missing_venv = tmp_path / "missing"
    common = (
        f'F1_RUNTIME_ROLE=gpu F1_VENV_PATH="{missing_venv}" '
        "RLINF_COMM_NET_DEVICES=eth0 source "
        f'"{SETUP_SCRIPT}"'
    )
    preserve = subprocess.run(
        [
            "bash",
            "-c",
            f'export RLINF_NODE_RANK=77; {common} || :; test "$RLINF_NODE_RANK" = 77',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    unset = subprocess.run(
        [
            "bash",
            "-c",
            f'unset RLINF_NODE_RANK; {common} || :; test -z "${{RLINF_NODE_RANK+x}}"',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert preserve.returncode == 0, preserve.stderr
    assert unset.returncode == 0, unset.stderr

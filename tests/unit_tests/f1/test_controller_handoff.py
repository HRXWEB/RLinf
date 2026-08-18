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

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

WHEEL_NAME = "f1_robot_controller-0.1.0-py3-none-any.whl"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERIFY_SCRIPT = PROJECT_ROOT / "toolkits" / "f1" / "verify_controller_handoff.py"
HANDOFF_FIELDS = (
    "api_contract_path",
    "package_version",
    "python_version",
    "release_commit",
    "repository_url",
    "wheel_path",
    "wheel_sha256",
)


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_controller_handoff", VERIFY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_handoff = _load_verifier().verify_handoff


def _build_wheel(
    wheel: Path,
    *,
    metadata_name: str = "f1-robot-controller",
    metadata_version: str = "0.1.0",
    include_metadata: bool = True,
) -> None:
    with ZipFile(wheel, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("f1_robot_controller/__init__.py", "")
        archive.writestr(
            "f1_robot_controller-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        if include_metadata:
            archive.writestr(
                "f1_robot_controller-0.1.0.dist-info/METADATA",
                "Metadata-Version: 2.4\n"
                f"Name: {metadata_name}\n"
                f"Version: {metadata_version}\n",
            )


def _handoff_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    wheel = tmp_path / WHEEL_NAME
    _build_wheel(wheel)
    payload: dict[str, object] = {
        "api_contract_path": "docs/api-contract-0.1.0.md",
        "package_version": "0.1.0",
        "python_version": "3.12.13",
        "release_commit": "1" * 40,
        "repository_url": "https://example.invalid/f1-robot-controller.git",
        "wheel_path": f"dist/{WHEEL_NAME}",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    handoff = tmp_path / "phase1-controller-handoff.json"
    return handoff, wheel, payload


def _write_handoff(handoff: Path, payload: dict[str, object]) -> None:
    handoff.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_handoff_rejects_wrong_wheel_sha256(tmp_path: Path) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["wheel_sha256"] = "0" * 64
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="SHA-256"):
        verify_handoff(handoff)


def test_verify_handoff_rejects_wrong_package_version(tmp_path: Path) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["package_version"] = "0.2.0"
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="package version"):
        verify_handoff(handoff)


def test_verify_handoff_rejects_missing_api_contract(tmp_path: Path) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    del payload["api_contract_path"]
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="API contract"):
        verify_handoff(handoff)


@pytest.mark.parametrize("missing_field", HANDOFF_FIELDS)
def test_verify_handoff_rejects_each_missing_schema_field(
    tmp_path: Path, missing_field: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    del payload[missing_field]
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match=missing_field):
        verify_handoff(handoff)


def test_verify_handoff_rejects_unrecognized_schema_field(tmp_path: Path) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["unreviewed_provenance"] = "not allowed"
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="unreviewed_provenance"):
        verify_handoff(handoff)


@pytest.mark.parametrize("field", HANDOFF_FIELDS)
def test_verify_handoff_rejects_non_string_schema_values(
    tmp_path: Path, field: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload[field] = 123
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match=field):
        verify_handoff(handoff)


@pytest.mark.parametrize(
    "recorded_path",
    [
        f"other/{WHEEL_NAME}",
        "dist/f1_robot_controller-0.2.0-py3-none-any.whl",
        "dist/controller.txt",
        f"/tmp/{WHEEL_NAME}",
        f"../dist/{WHEEL_NAME}",
    ],
)
def test_verify_handoff_requires_exact_posix_wheel_path(
    tmp_path: Path, recorded_path: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["wheel_path"] = recorded_path
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="wheel_path"):
        verify_handoff(handoff)


def test_verify_handoff_rejects_sibling_wheel_symlink_escape(tmp_path: Path) -> None:
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_wheel = outside_dir / WHEEL_NAME
    _build_wheel(outside_wheel)
    wheel_symlink = handoff_dir / WHEEL_NAME
    wheel_symlink.symlink_to(outside_wheel)
    handoff = handoff_dir / "phase1-controller-handoff.json"
    payload: dict[str, object] = {
        "api_contract_path": "docs/api-contract-0.1.0.md",
        "package_version": "0.1.0",
        "python_version": "3.12.13",
        "release_commit": "1" * 40,
        "repository_url": "https://example.invalid/f1-robot-controller.git",
        "wheel_path": f"dist/{WHEEL_NAME}",
        "wheel_sha256": hashlib.sha256(outside_wheel.read_bytes()).hexdigest(),
    }
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="non-symlink"):
        verify_handoff(handoff)


@pytest.mark.parametrize(
    "repository_url",
    ["", "not-a-repository", "ftp://example.invalid/controller.git"],
)
def test_verify_handoff_rejects_invalid_repository_url(
    tmp_path: Path, repository_url: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["repository_url"] = repository_url
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="repository_url"):
        verify_handoff(handoff)


@pytest.mark.parametrize("release_commit", ["a" * 39, "A" * 40, "g" * 40])
def test_verify_handoff_rejects_invalid_release_commit(
    tmp_path: Path, release_commit: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["release_commit"] = release_commit
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="release_commit"):
        verify_handoff(handoff)


@pytest.mark.parametrize("python_version", ["3.11.9", "3.12", "3.12.x", "3.12.13rc1"])
def test_verify_handoff_rejects_invalid_python_version(
    tmp_path: Path, python_version: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["python_version"] = python_version
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="python_version"):
        verify_handoff(handoff)


@pytest.mark.parametrize("wheel_sha256", ["a" * 63, "A" * 64, "g" * 64])
def test_verify_handoff_rejects_invalid_wheel_sha256_format(
    tmp_path: Path, wheel_sha256: str
) -> None:
    handoff, _, payload = _handoff_fixture(tmp_path)
    payload["wheel_sha256"] = wheel_sha256
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="wheel_sha256"):
        verify_handoff(handoff)


@pytest.mark.parametrize(
    ("metadata_name", "metadata_version", "message"),
    [
        ("another-controller", "0.1.0", "Name"),
        ("f1-robot-controller", "0.2.0", "Version"),
    ],
)
def test_verify_handoff_rejects_wrong_wheel_identity(
    tmp_path: Path,
    metadata_name: str,
    metadata_version: str,
    message: str,
) -> None:
    handoff, wheel, payload = _handoff_fixture(tmp_path)
    _build_wheel(
        wheel,
        metadata_name=metadata_name,
        metadata_version=metadata_version,
    )
    payload["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match=message):
        verify_handoff(handoff)


def test_verify_handoff_rejects_non_wheel_bytes(tmp_path: Path) -> None:
    handoff, wheel, payload = _handoff_fixture(tmp_path)
    wheel.write_bytes(b"not a wheel")
    payload["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="valid wheel"):
        verify_handoff(handoff)


def test_verify_handoff_rejects_wheel_without_metadata(tmp_path: Path) -> None:
    handoff, wheel, payload = _handoff_fixture(tmp_path)
    _build_wheel(wheel, include_metadata=False)
    payload["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    _write_handoff(handoff, payload)

    with pytest.raises(ValueError, match="exactly one METADATA"):
        verify_handoff(handoff)


def test_verify_handoff_returns_digest_verified_wheel(tmp_path: Path) -> None:
    handoff, wheel, payload = _handoff_fixture(tmp_path)
    _write_handoff(handoff, payload)

    verified_wheel = verify_handoff(handoff)

    assert verified_wheel == wheel.resolve()

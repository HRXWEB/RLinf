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

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

WHEEL_NAME = "f1_robot_controller-0.1.0-py3-none-any.whl"
WHEEL_BYTES = b"authoritative wheel fixture"
WHEEL_SHA256 = "1447521cbe3d24803204d4e9927d1031ada8afa7d1dd17b213dae748e2129115"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERIFY_SCRIPT = PROJECT_ROOT / "toolkits" / "f1" / "verify_controller_handoff.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_controller_handoff", VERIFY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_handoff = _load_verifier().verify_handoff


def _handoff_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    wheel = tmp_path / WHEEL_NAME
    wheel.write_bytes(WHEEL_BYTES)
    payload = {
        "api_contract_path": "docs/api-contract-0.1.0.md",
        "package_version": "0.1.0",
        "python_version": "3.12.13",
        "release_commit": "1" * 40,
        "repository_url": "https://example.invalid/f1-robot-controller.git",
        "wheel_path": f"dist/{WHEEL_NAME}",
        "wheel_sha256": WHEEL_SHA256,
    }
    handoff = tmp_path / "phase1-controller-handoff.json"
    return handoff, wheel, payload


def _write_handoff(handoff: Path, payload: dict[str, str]) -> None:
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


def test_verify_handoff_returns_digest_verified_wheel(tmp_path: Path) -> None:
    handoff, wheel, payload = _handoff_fixture(tmp_path)
    _write_handoff(handoff, payload)

    verified_wheel = verify_handoff(handoff)

    assert verified_wheel == wheel.resolve()

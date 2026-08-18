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

"""Verify the immutable F1 controller handoff and print its wheel path."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_API_CONTRACT = "docs/api-contract-0.1.0.md"
EXPECTED_PACKAGE_VERSION = "0.1.0"


def _load_handoff(handoff_path: Path) -> dict[str, object]:
    """Load one JSON object from ``handoff_path``."""

    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("controller handoff must contain a JSON object")
    return payload


def _wheel_sha256(wheel_path: Path) -> str:
    """Return the SHA-256 digest of ``wheel_path``."""

    digest = hashlib.sha256()
    with wheel_path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_handoff(handoff_path: Path) -> Path:
    """Validate a phase-one controller handoff and return its wheel path.

    Args:
        handoff_path: JSON handoff copied beside its controller wheel.

    Returns:
        The absolute path to the digest-verified controller wheel.

    Raises:
        OSError: If the handoff or wheel cannot be read.
        ValueError: If required contract, version, or digest data is invalid.
    """

    handoff_path = handoff_path.resolve()
    payload = _load_handoff(handoff_path)

    package_version = payload.get("package_version")
    if package_version != EXPECTED_PACKAGE_VERSION:
        raise ValueError(
            "controller handoff package version must be "
            f"{EXPECTED_PACKAGE_VERSION}, got {package_version!r}"
        )

    api_contract_path = payload.get("api_contract_path")
    if api_contract_path != EXPECTED_API_CONTRACT:
        raise ValueError(
            f"controller handoff API contract must be {EXPECTED_API_CONTRACT}"
        )

    recorded_wheel_path = payload.get("wheel_path")
    if not isinstance(recorded_wheel_path, str) or not recorded_wheel_path:
        raise ValueError("controller handoff wheel path is missing")
    wheel_path = (handoff_path.parent / Path(recorded_wheel_path).name).resolve()

    expected_sha256 = payload.get("wheel_sha256")
    actual_sha256 = _wheel_sha256(wheel_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "controller wheel SHA-256 mismatch: "
            f"expected {expected_sha256!r}, got {actual_sha256}"
        )
    return wheel_path


def main() -> int:
    """Validate the requested handoff and print the verified wheel path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    args = parser.parse_args()
    try:
        wheel_path = verify_handoff(args.handoff)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(wheel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

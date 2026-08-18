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
import re
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

EXPECTED_API_CONTRACT = "docs/api-contract-0.1.0.md"
EXPECTED_PACKAGE_VERSION = "0.1.0"
EXPECTED_WHEEL_PATH = "dist/f1_robot_controller-0.1.0-py3-none-any.whl"
EXPECTED_WHEEL_NAME = "f1-robot-controller"
EXPECTED_FIELDS = frozenset(
    {
        "api_contract_path",
        "package_version",
        "python_version",
        "release_commit",
        "repository_url",
        "wheel_path",
        "wheel_sha256",
    }
)


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


def _validate_schema(payload: dict[str, object]) -> dict[str, str]:
    """Validate the handoff fields and return their string values."""

    missing_fields = EXPECTED_FIELDS - payload.keys()
    if missing_fields:
        missing_field = sorted(missing_fields)[0]
        label = "API contract" if missing_field == "api_contract_path" else "field"
        raise ValueError(f"controller handoff {label} {missing_field} is missing")
    extra_fields = payload.keys() - EXPECTED_FIELDS
    if extra_fields:
        raise ValueError(
            f"controller handoff contains unrecognized field {sorted(extra_fields)[0]}"
        )

    string_payload: dict[str, str] = {}
    for field in EXPECTED_FIELDS:
        value = payload[field]
        if not isinstance(value, str):
            raise ValueError(f"controller handoff field {field} must be a string")
        string_payload[field] = value
    return string_payload


def _is_repository_url(value: str) -> bool:
    """Return whether ``value`` resembles a URL or SSH repository location."""

    if "://" in value:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https", "ssh"}
            and parsed.hostname is not None
            and parsed.path not in {"", "/"}
        )
    return re.fullmatch(r"(?:[^@\s/:]+@)?[^@\s/:]+:[^\s]+", value) is not None


def _validate_provenance(payload: dict[str, str]) -> None:
    """Validate immutable controller release provenance."""

    repository_url = payload["repository_url"]
    if not _is_repository_url(repository_url):
        raise ValueError(
            "controller handoff repository_url is not a valid repository URL"
        )

    release_commit = payload["release_commit"]
    if re.fullmatch(r"[0-9a-f]{40}", release_commit) is None:
        raise ValueError(
            "controller handoff release_commit must be 40 lowercase hex characters"
        )

    package_version = payload["package_version"]
    if package_version != EXPECTED_PACKAGE_VERSION:
        raise ValueError(
            "controller handoff package version (package_version) must be "
            f"{EXPECTED_PACKAGE_VERSION}, got {package_version!r}"
        )

    python_version = payload["python_version"]
    if re.fullmatch(r"3\.12\.\d+", python_version) is None:
        raise ValueError(
            "controller handoff python_version must be a stable Python 3.12.x version"
        )

    api_contract_path = payload["api_contract_path"]
    if api_contract_path != EXPECTED_API_CONTRACT:
        raise ValueError(
            "controller handoff API contract (api_contract_path) must be "
            f"{EXPECTED_API_CONTRACT}"
        )

    recorded_wheel_path = payload["wheel_path"]
    posix_wheel_path = PurePosixPath(recorded_wheel_path)
    if (
        posix_wheel_path.is_absolute()
        or ".." in posix_wheel_path.parts
        or recorded_wheel_path != EXPECTED_WHEEL_PATH
    ):
        raise ValueError(
            f"controller handoff wheel_path must be exactly {EXPECTED_WHEEL_PATH}"
        )

    wheel_sha256 = payload["wheel_sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", wheel_sha256) is None:
        raise ValueError(
            "controller handoff wheel_sha256 must be 64 lowercase hex characters"
        )


def _flattened_wheel(handoff_path: Path) -> Path:
    """Return the verified non-symlink wheel copied beside the handoff."""

    wheel_name = PurePosixPath(EXPECTED_WHEEL_PATH).name
    wheel_path = handoff_path.parent / wheel_name
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(
            "flattened controller wheel must be a regular non-symlink file "
            "beside the handoff"
        )
    resolved_wheel = wheel_path.resolve(strict=True)
    if resolved_wheel.parent != handoff_path.parent:
        raise ValueError("flattened controller wheel must remain beside the handoff")
    return resolved_wheel


def _validate_wheel_identity(wheel_path: Path) -> None:
    """Validate the distribution name and version in wheel METADATA."""

    try:
        with ZipFile(wheel_path) as archive:
            metadata_paths = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(
                    "controller wheel must contain exactly one METADATA file"
                )
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
    except BadZipFile as error:
        raise ValueError("controller wheel must be a valid wheel archive") from error

    if metadata["Name"] != EXPECTED_WHEEL_NAME:
        raise ValueError(
            f"controller wheel METADATA Name must be {EXPECTED_WHEEL_NAME}"
        )
    if metadata["Version"] != EXPECTED_PACKAGE_VERSION:
        raise ValueError(
            f"controller wheel METADATA Version must be {EXPECTED_PACKAGE_VERSION}"
        )


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
    payload = _validate_schema(_load_handoff(handoff_path))
    _validate_provenance(payload)
    wheel_path = _flattened_wheel(handoff_path)
    _validate_wheel_identity(wheel_path)

    expected_sha256 = payload["wheel_sha256"]
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

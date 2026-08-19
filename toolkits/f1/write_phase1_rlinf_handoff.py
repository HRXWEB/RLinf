# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Write a digest-bound, fail-closed handoff for F1 phase-one RLinf work."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

EXPECTED_ORIGIN = "git@github.com:HRXWEB/RLinf.git"
SCHEMA_VERSION = "phase1-rlinf-handoff-v1"
PRIMARY_CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
E2E_CONFIG_NAME = "realworld_f1_dummy_sac_cnn"
UNIT_COMMAND = [".venv/bin/pytest", "tests/unit_tests/f1", "-q"]
E2E_COMMAND = [
    "bash",
    "tests/e2e_tests/embodied/run_async.sh",
    "realworld_f1_dummy_sac_cnn",
]
WHEEL_NAME = "f1_robot_controller-0.1.0-py3-none-any.whl"
EXPECTED_E2E_RESULT_FIELDS = frozenset(
    {
        "actor_loss",
        "actor_update_count",
        "critic_loss",
        "critic_update_count",
        "replay_buffer_size",
        "weight_sync_success_total",
        "status",
    }
)
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _load_controller_verifier() -> ModuleType:
    """Load the sibling verifier without depending on repository sys.path."""

    verifier_path = Path(__file__).with_name("verify_controller_handoff.py")
    spec = importlib.util.spec_from_file_location(
        "verify_controller_handoff", verifier_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the controller handoff verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_output(repository_root: Path, *args: str) -> str:
    """Return one successful git command's trimmed standard output."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"could not validate the RLinf repository: {' '.join(args)}"
        ) from error
    return completed.stdout.strip()


def _validate_repository(repository_root: Path) -> tuple[Path, str]:
    """Require the expected clean checkout and an immutable commit identifier."""

    resolved_root = repository_root.resolve(strict=True)
    top_level = Path(_git_output(resolved_root, "rev-parse", "--show-toplevel"))
    if top_level.resolve(strict=True) != resolved_root:
        raise ValueError("repository_root must be the RLinf worktree root")

    status = _git_output(
        resolved_root, "status", "--porcelain", "--untracked-files=all"
    )
    if status:
        raise ValueError("RLinf worktree must be clean before writing a handoff")

    origin = _git_output(resolved_root, "remote", "get-url", "origin")
    if origin != EXPECTED_ORIGIN:
        raise ValueError("RLinf origin is missing or is not the expected repository")

    commit = _git_output(resolved_root, "rev-parse", "HEAD")
    if _GIT_SHA.fullmatch(commit) is None:
        raise ValueError("RLinf HEAD must be 40 lowercase hexadecimal characters")
    return resolved_root, commit


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the fields that reveal replacement or mutation during a read."""

    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_regular_snapshot(path: Path, label: str) -> bytes:
    """Read one stable regular file and reject link, replacement, or mutation."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise ValueError(f"{label} is unreadable") from error
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise ValueError(f"{label} was replaced while being read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        after_path = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} changed while being read") from error
    finally:
        os.close(descriptor)

    if _file_identity(after_open) != _file_identity(before) or _file_identity(
        after_path
    ) != _file_identity(before):
        raise ValueError(f"{label} changed while being read")
    snapshot = b"".join(chunks)
    if not snapshot:
        raise ValueError(f"{label} must not be empty")
    return snapshot


def _execution_record_path(log_path: Path) -> Path:
    """Return the committed runner's required sibling record path."""

    return log_path.with_suffix(".record.json")


def _require_exact_fields(
    payload: dict[str, object], fields: frozenset[str], label: str
) -> None:
    """Require one closed JSON object schema."""

    actual = frozenset(payload)
    if actual == fields:
        return
    extras = actual - fields
    if extras:
        raise ValueError(f"{label} contains unrecognized field {sorted(extras)[0]}")
    raise ValueError(f"{label} is missing field {sorted(fields - actual)[0]}")


def _require_integer(value: object, field: str, minimum: int) -> int:
    """Require a JSON integer at least ``minimum`` without accepting bool."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"execution record {field} must be an integer >= {minimum}")
    return value


def _validate_execution_record(
    kind: str,
    log_path: Path,
    log_snapshot: bytes,
    commit: str,
) -> bytes:
    """Require a runner-produced record bound to this log, command, and commit."""

    record_snapshot = _read_regular_snapshot(
        _execution_record_path(log_path), f"{kind} execution record"
    )
    try:
        payload = json.loads(record_snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{kind} execution record must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} execution record must contain a JSON object")
    _require_exact_fields(
        payload,
        frozenset(
            {
                "command",
                "exit_code",
                "finished_at",
                "kind",
                "log_filename",
                "log_sha256",
                "repository_url",
                "result",
                "rlinf_commit",
                "schema_version",
                "started_at",
            }
        ),
        f"{kind} execution record",
    )
    command = UNIT_COMMAND if kind == "unit" else E2E_COMMAND
    if payload["schema_version"] != "f1-execution-record-v1":
        raise ValueError(f"{kind} execution record has an unsupported schema_version")
    if payload["kind"] != kind or payload["command"] != command:
        raise ValueError(f"{kind} execution record is not for the required command")
    if payload["repository_url"] != EXPECTED_ORIGIN:
        raise ValueError(f"{kind} execution record repository origin does not match")
    if payload["rlinf_commit"] != commit:
        raise ValueError(f"{kind} execution record commit does not match RLinf HEAD")
    if payload["log_filename"] != log_path.name:
        raise ValueError(f"{kind} execution record log filename does not match")
    if payload["log_sha256"] != hashlib.sha256(log_snapshot).hexdigest():
        raise ValueError(f"{kind} execution record log SHA-256 does not match")
    if (
        isinstance(payload["exit_code"], bool)
        or not isinstance(payload["exit_code"], int)
        or payload["exit_code"] != 0
    ):
        raise ValueError(f"{kind} execution record exit_code must be integer zero")
    if not all(
        isinstance(payload[field], str) and _UTC_TIMESTAMP.fullmatch(payload[field])
        for field in ("started_at", "finished_at")
    ):
        raise ValueError(f"{kind} execution record timestamps must be UTC")
    result = payload["result"]
    if not isinstance(result, dict):
        raise ValueError(f"{kind} execution record result must be a JSON object")
    if kind == "unit":
        _require_exact_fields(
            result, frozenset({"pytest_passed", "status"}), "unit result"
        )
        _require_integer(result["pytest_passed"], "pytest_passed", 1)
    else:
        _require_exact_fields(result, EXPECTED_E2E_RESULT_FIELDS, "E2E result")
        for field, minimum in (
            ("replay_buffer_size", 1),
            ("actor_update_count", 1),
            ("critic_update_count", 1),
            ("weight_sync_success_total", 2),
        ):
            _require_integer(result[field], field, minimum)
        for field in ("actor_loss", "critic_loss"):
            if (
                isinstance(result[field], bool)
                or not isinstance(result[field], (int, float))
                or not math.isfinite(float(result[field]))
            ):
                raise ValueError(f"execution record {field} must be a number")
    if result.get("status") != "passed":
        raise ValueError(f"{kind} execution record result is not passed")
    return record_snapshot


def _snapshot_controller(
    controller_handoff: Path,
) -> tuple[bytes, bytes, dict[str, str]]:
    """Validate and freeze the controller handoff and wheel in a private staging dir."""

    verifier = _load_controller_verifier()
    wheel_path = verifier.verify_handoff(controller_handoff)
    handoff_snapshot = _read_regular_snapshot(controller_handoff, "controller handoff")
    wheel_snapshot = _read_regular_snapshot(wheel_path, "controller wheel")
    with tempfile.TemporaryDirectory(prefix="f1-controller-handoff-") as staging:
        staging_root = Path(staging)
        staged_handoff = staging_root / "phase1-controller-handoff.json"
        staged_wheel = staging_root / WHEEL_NAME
        staged_handoff.write_bytes(handoff_snapshot)
        staged_wheel.write_bytes(wheel_snapshot)
        verifier.verify_handoff(staged_handoff)
    try:
        payload = json.loads(handoff_snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("controller handoff must contain valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("controller handoff must contain a JSON object")
    release_commit = payload.get("release_commit")
    wheel_sha256 = payload.get("wheel_sha256")
    if not isinstance(release_commit, str) or not isinstance(wheel_sha256, str):
        raise ValueError("controller handoff verifier returned invalid provenance")
    return (
        handoff_snapshot,
        wheel_snapshot,
        {
            "release_commit": release_commit,
            "wheel_sha256": wheel_sha256,
        },
    )


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether an absolute lexical path is within another absolute path."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving symlinks."""

    return Path(os.path.abspath(path))


def _open_directory_no_follow(directory: Path) -> int:
    """Open every absolute path component with ``O_NOFOLLOW``."""

    absolute = _absolute_path(directory)
    if not absolute.is_absolute():
        raise ValueError("handoff output parent must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError("handoff output has a missing or symlink ancestor") from error
    return descriptor


def _same_directory(path: Path, descriptor: int) -> bool:
    """Return whether ``path`` still identifies the opened directory."""

    try:
        current = _open_directory_no_follow(path)
    except ValueError:
        return False
    try:
        return (
            _file_identity(os.fstat(current))[:2]
            == _file_identity(os.fstat(descriptor))[:2]
        )
    finally:
        os.close(current)


def _validate_output_path(output: Path, repository_root: Path) -> Path:
    """Reject symlink, tracked, and unsafe in-repository handoff destinations."""

    output_path = _absolute_path(output)
    try:
        output_status = os.lstat(output_path)
    except FileNotFoundError:
        output_status = None
    if output_status is not None and stat.S_ISLNK(output_status.st_mode):
        raise ValueError("handoff output must not be a symlink")
    descriptor = _open_directory_no_follow(output_path.parent)
    os.close(descriptor)
    if _path_is_within(output_path, repository_root):
        relative_output = str(output_path.relative_to(repository_root))
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    relative_output,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            pass
        else:
            raise ValueError("handoff output must not overwrite a tracked file")
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "check-ignore",
                "--quiet",
                "--",
                relative_output,
            ],
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("in-repository handoff output must be ignored by Git")
    return output_path


def _atomic_write_bytes(output: Path, data: bytes) -> None:
    """Write one file through a no-follow parent descriptor and atomically replace."""

    output_path = _absolute_path(output)
    descriptor = _open_directory_no_follow(output_path.parent)
    temporary_name = f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        written = 0
        while written < len(data):
            written += os.write(temporary_descriptor, data[written:])
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        if not _same_directory(output_path.parent, descriptor):
            raise ValueError("handoff output parent changed during atomic write")
        try:
            existing = os.stat(
                output_path.name, dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ValueError("handoff output must not be a symlink")
        os.replace(
            temporary_name,
            output_path.name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        os.fsync(descriptor)
        if not _same_directory(output_path.parent, descriptor):
            raise ValueError("handoff output parent changed during atomic write")
    except BaseException:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)


def _atomic_write(output: Path, payload: dict[str, object]) -> None:
    """Persist canonical JSON via descriptor-confined atomic replacement."""

    _atomic_write_bytes(
        output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _utc_timestamp() -> str:
    """Return a stable, second-precision UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_handoff(
    *,
    controller_handoff: Path,
    unit_log: Path,
    e2e_log: Path,
    output: Path,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Write the immutable phase-one RLinf handoff after all gates pass.

    Args:
        controller_handoff: Controller handoff copied beside its wheel.
        unit_log: Complete unit-test log with its success sentinel.
        e2e_log: Complete GPU E2E log with metric evidence and success sentinel.
        output: Destination outside Git or explicitly ignored within the worktree.
        repository_root: RLinf checkout root; defaults to this script's repository.

    Returns:
        The canonical payload that was atomically written to ``output``.

    Raises:
        ValueError: If provenance, files, evidence, or output safety is invalid.
        OSError: If atomic persistence fails.
    """

    default_root = Path(__file__).resolve().parents[2]
    root, commit = _validate_repository(repository_root or default_root)
    output_path = _validate_output_path(output, root)
    handoff_snapshot, wheel_snapshot, controller = _snapshot_controller(
        controller_handoff
    )
    unit_snapshot = _read_regular_snapshot(unit_log, "unit log")
    e2e_snapshot = _read_regular_snapshot(e2e_log, "E2E log")
    unit_record_snapshot = _validate_execution_record(
        "unit", unit_log, unit_snapshot, commit
    )
    e2e_record_snapshot = _validate_execution_record(
        "e2e", e2e_log, e2e_snapshot, commit
    )

    payload: dict[str, object] = {
        "action_schema": "f1-action-v1",
        "controller_handoff_sha256": hashlib.sha256(handoff_snapshot).hexdigest(),
        "controller_release_commit": controller["release_commit"],
        "controller_wheel_sha256": hashlib.sha256(wheel_snapshot).hexdigest(),
        "created_at": _utc_timestamp(),
        "e2e_command": E2E_COMMAND,
        "e2e_config_name": E2E_CONFIG_NAME,
        "e2e_log_path": str(e2e_log.resolve(strict=True)),
        "e2e_log_sha256": hashlib.sha256(e2e_snapshot).hexdigest(),
        "e2e_record_path": str(_execution_record_path(e2e_log).resolve(strict=True)),
        "e2e_record_sha256": hashlib.sha256(e2e_record_snapshot).hexdigest(),
        "observation_schema": "f1-observation-v1",
        "primary_config_name": PRIMARY_CONFIG_NAME,
        "repository_url": EXPECTED_ORIGIN,
        "rlinf_commit": commit,
        "schema_version": SCHEMA_VERSION,
        "transition_schema": "f1-transition-v1",
        "unit_command": UNIT_COMMAND,
        "unit_log_path": str(unit_log.resolve(strict=True)),
        "unit_log_sha256": hashlib.sha256(unit_snapshot).hexdigest(),
        "unit_record_path": str(_execution_record_path(unit_log).resolve(strict=True)),
        "unit_record_sha256": hashlib.sha256(unit_record_snapshot).hexdigest(),
    }
    _atomic_write(output_path, payload)
    return payload


def main() -> int:
    """Write one phase-one RLinf handoff from four explicit artifact paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-handoff", required=True, type=Path)
    parser.add_argument("--unit-log", required=True, type=Path)
    parser.add_argument("--e2e-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        write_handoff(
            controller_handoff=args.controller_handoff,
            unit_log=args.unit_log,
            e2e_log=args.e2e_log,
            output=args.output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

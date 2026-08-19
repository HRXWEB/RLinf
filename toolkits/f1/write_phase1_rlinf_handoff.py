# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Write a digest-bound, fail-closed handoff for F1 phase-one RLinf work.

The unit runner must append ``F1_UNIT_TESTS_PASSED`` only after its command
exits zero.  The GPU E2E runner must append the strict evidence JSON and then
``F1_E2E_PASSED`` only after the training command exits zero.  The evidence
JSON is deliberately extracted from the real training log by that runner; this
writer treats a log as an immutable execution record, not as a command runner.
"""

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
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

EXPECTED_ORIGIN = "git@github.com:HRXWEB/RLinf.git"
SCHEMA_VERSION = "phase1-rlinf-handoff-v1"
PRIMARY_CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
E2E_CONFIG_NAME = "realworld_f1_dummy_sac_cnn"
UNIT_COMMAND = ".venv/bin/pytest tests/unit_tests/f1 -q"
E2E_COMMAND = "bash tests/e2e_tests/embodied/run_async.sh realworld_f1_dummy_sac_cnn"
UNIT_SENTINEL = "F1_UNIT_TESTS_PASSED"
E2E_SENTINEL = "F1_E2E_PASSED"
E2E_EVIDENCE_PREFIX = "F1_E2E_EVIDENCE="
WHEEL_NAME = "f1_robot_controller-0.1.0-py3-none-any.whl"
EXPECTED_E2E_FIELDS = frozenset(
    {
        "actor_loss",
        "actor_update_count",
        "critic_loss",
        "critic_update_count",
        "replay_buffer_size",
        "weight_sync_success_total",
    }
)
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_GLOBAL_STEP = re.compile(r"\bGlobal Step\s*:?\s*2\s*/\s*2\b")


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


def _require_final_sentinel(log_text: str, sentinel: str, label: str) -> None:
    """Require exactly one completion sentinel as the final non-blank line."""

    lines = log_text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[-1] != sentinel or lines.count(sentinel) != 1:
        raise ValueError(f"{label} must end with exactly one {sentinel}")


def _validate_unit_log(unit_snapshot: bytes) -> None:
    """Validate the successful unit-run completion contract."""

    try:
        unit_text = unit_snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("unit log must be UTF-8") from error
    _require_final_sentinel(unit_text, UNIT_SENTINEL, "unit log")


def _validate_number(value: object, field: str) -> float:
    """Return one finite JSON number while rejecting bool and non-number values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"F1_E2E_EVIDENCE field {field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"F1_E2E_EVIDENCE field {field} must be finite")
    return number


def _validate_e2e_evidence(e2e_text: str) -> None:
    """Validate the required real-training evidence extracted into the E2E log."""

    if _GLOBAL_STEP.search(e2e_text) is None:
        raise ValueError("E2E log must contain Global Step 2/2")
    evidence_lines = [
        line.removeprefix(E2E_EVIDENCE_PREFIX)
        for line in e2e_text.splitlines()
        if line.startswith(E2E_EVIDENCE_PREFIX)
    ]
    if len(evidence_lines) != 1:
        raise ValueError("E2E log must contain exactly one F1_E2E_EVIDENCE line")
    try:
        evidence = json.loads(evidence_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("F1_E2E_EVIDENCE must contain valid JSON") from error
    if not isinstance(evidence, dict):
        raise ValueError("F1_E2E_EVIDENCE must be a JSON object")
    evidence_fields = frozenset(evidence)
    if evidence_fields != EXPECTED_E2E_FIELDS:
        extras = evidence_fields - EXPECTED_E2E_FIELDS
        if extras:
            raise ValueError(
                f"F1_E2E_EVIDENCE contains unrecognized field {sorted(extras)[0]}"
            )
        raise ValueError(
            "F1_E2E_EVIDENCE is missing field "
            f"{sorted(EXPECTED_E2E_FIELDS - evidence_fields)[0]}"
        )
    replay_size = _validate_number(evidence["replay_buffer_size"], "replay_buffer_size")
    actor_updates = _validate_number(
        evidence["actor_update_count"], "actor_update_count"
    )
    critic_updates = _validate_number(
        evidence["critic_update_count"], "critic_update_count"
    )
    sync_count = _validate_number(
        evidence["weight_sync_success_total"], "weight_sync_success_total"
    )
    if not replay_size.is_integer() or replay_size < 1:
        raise ValueError("F1_E2E_EVIDENCE replay_buffer_size must be an integer >= 1")
    if not actor_updates.is_integer() or actor_updates < 1:
        raise ValueError("F1_E2E_EVIDENCE actor_update_count must be an integer >= 1")
    if not critic_updates.is_integer() or critic_updates < 1:
        raise ValueError("F1_E2E_EVIDENCE critic_update_count must be an integer >= 1")
    if not sync_count.is_integer() or sync_count < 2:
        raise ValueError(
            "F1_E2E_EVIDENCE weight_sync_success_total must be an integer >= 2"
        )
    _validate_number(evidence["actor_loss"], "actor_loss")
    _validate_number(evidence["critic_loss"], "critic_loss")


def _validate_e2e_log(e2e_snapshot: bytes) -> None:
    """Validate the successful E2E completion and metric evidence contract."""

    try:
        e2e_text = e2e_snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("E2E log must be UTF-8") from error
    _require_final_sentinel(e2e_text, E2E_SENTINEL, "E2E log")
    _validate_e2e_evidence(e2e_text)


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


def _validate_output_path(output: Path, repository_root: Path) -> Path:
    """Reject symlink, tracked, and unsafe in-repository handoff destinations."""

    output_path = output.absolute()
    if output_path.is_symlink():
        raise ValueError("handoff output must not be a symlink")
    if not output_path.parent.is_dir() or output_path.parent.is_symlink():
        raise ValueError("handoff output parent must be a real existing directory")
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


def _atomic_write(output: Path, payload: dict[str, object]) -> None:
    """Persist canonical JSON via fsync and one atomic replacement."""

    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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
    _validate_unit_log(unit_snapshot)
    _validate_e2e_log(e2e_snapshot)

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
        "observation_schema": "f1-observation-v1",
        "primary_config_name": PRIMARY_CONFIG_NAME,
        "repository_url": EXPECTED_ORIGIN,
        "rlinf_commit": commit,
        "schema_version": SCHEMA_VERSION,
        "transition_schema": "f1-transition-v1",
        "unit_command": UNIT_COMMAND,
        "unit_log_path": str(unit_log.resolve(strict=True)),
        "unit_log_sha256": hashlib.sha256(unit_snapshot).hexdigest(),
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

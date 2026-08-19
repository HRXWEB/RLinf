# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the fail-closed F1 phase-one RLinf handoff writer."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WRITER_SCRIPT = PROJECT_ROOT / "toolkits" / "f1" / "write_phase1_rlinf_handoff.py"
RUNNER_SCRIPT = PROJECT_ROOT / "toolkits" / "f1" / "run_phase1_rlinf_evidence.py"
EXPECTED_ORIGIN = "git@github.com:HRXWEB/RLinf.git"
WHEEL_NAME = "f1_robot_controller-0.1.0-py3-none-any.whl"


def _load_writer() -> ModuleType:
    assert WRITER_SCRIPT.is_file(), "the phase-one handoff writer must exist"
    spec = importlib.util.spec_from_file_location(
        "write_phase1_rlinf_handoff", WRITER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_phase1_rlinf_evidence", RUNNER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "RLinf"
    repository.mkdir()
    _run_git(repository, "init")
    _run_git(repository, "config", "user.email", "f1-test@example.invalid")
    _run_git(repository, "config", "user.name", "F1 Test")
    _run_git(repository, "remote", "add", "origin", EXPECTED_ORIGIN)
    (repository / "README.md").write_text("F1 test repository\n", encoding="utf-8")
    _run_git(repository, "add", "README.md")
    _run_git(repository, "commit", "-m", "test repository")
    return repository


def _build_wheel(path: Path) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("f1_robot_controller/__init__.py", "")
        archive.writestr(
            "f1_robot_controller-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: f1-robot-controller\nVersion: 0.1.0\n",
        )


def _make_controller_artifacts(tmp_path: Path) -> Path:
    artifacts = tmp_path / "controller"
    artifacts.mkdir()
    wheel = artifacts / WHEEL_NAME
    _build_wheel(wheel)
    handoff = artifacts / "phase1-controller-handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "api_contract_path": "docs/api-contract-0.1.0.md",
                "package_version": "0.1.0",
                "python_version": "3.12.13",
                "release_commit": "a" * 40,
                "repository_url": "ssh://git@example.invalid/f1-robot-controller.git",
                "wheel_path": f"dist/{WHEEL_NAME}",
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return handoff


def _write_logs(tmp_path: Path, repository: Path) -> tuple[Path, Path]:
    unit_log = tmp_path / "f1-unit.log"
    unit_log.write_text("184 passed in 6.77s\nF1_UNIT_TESTS_PASSED\n", encoding="utf-8")
    e2e_log = tmp_path / "f1-e2e.log"
    e2e_log.write_text(
        "Global Step 2/2\n"
        'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
        '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
        '"weight_sync_success_total":2}\n'
        "F1_E2E_PASSED\n",
        encoding="utf-8",
    )
    commit = _git_output(repository, "rev-parse", "HEAD")
    for kind, log_path, command, result in (
        (
            "unit",
            unit_log,
            [".venv/bin/pytest", "tests/unit_tests/f1", "-q"],
            {"pytest_passed": 184, "status": "passed"},
        ),
        (
            "e2e",
            e2e_log,
            [
                "bash",
                "tests/e2e_tests/embodied/run_async.sh",
                "realworld_f1_dummy_sac_cnn",
            ],
            {
                "actor_loss": 0.25,
                "actor_update_count": 1,
                "critic_loss": 0.5,
                "critic_update_count": 1,
                "replay_buffer_size": 2,
                "status": "passed",
                "weight_sync_success_total": 2,
            },
        ),
    ):
        record = {
            "command": command,
            "exit_code": 0,
            "finished_at": "2026-08-19T00:00:01Z",
            "kind": kind,
            "log_filename": log_path.name,
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "repository_url": EXPECTED_ORIGIN,
            "result": result,
            "rlinf_commit": commit,
            "schema_version": "f1-execution-record-v1",
            "started_at": "2026-08-19T00:00:00Z",
        }
        log_path.with_suffix(".record.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return unit_log, e2e_log


def _write_valid_handoff(
    writer: ModuleType,
    repository: Path,
    controller_handoff: Path,
    unit_log: Path,
    e2e_log: Path,
    output: Path,
) -> dict[str, object]:
    return writer.write_handoff(
        controller_handoff=controller_handoff,
        unit_log=unit_log,
        e2e_log=e2e_log,
        output=output,
        repository_root=repository,
    )


def test_writer_emits_stable_digest_bound_phase_one_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)
    output = tmp_path / "phase1-rlinf-handoff.json"
    monkeypatch.setattr(writer, "_utc_timestamp", lambda: "2026-08-19T00:00:00Z")

    payload = _write_valid_handoff(
        writer, repository, controller_handoff, unit_log, e2e_log, output
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert payload == persisted
    assert list(persisted) == sorted(persisted)
    assert persisted == {
        "action_schema": "f1-action-v1",
        "controller_handoff_sha256": hashlib.sha256(
            controller_handoff.read_bytes()
        ).hexdigest(),
        "controller_release_commit": "a" * 40,
        "controller_wheel_sha256": hashlib.sha256(
            (controller_handoff.parent / WHEEL_NAME).read_bytes()
        ).hexdigest(),
        "created_at": "2026-08-19T00:00:00Z",
        "e2e_command": [
            "bash",
            "tests/e2e_tests/embodied/run_async.sh",
            "realworld_f1_dummy_sac_cnn",
        ],
        "e2e_config_name": "realworld_f1_dummy_sac_cnn",
        "e2e_log_path": str(e2e_log.resolve()),
        "e2e_log_sha256": hashlib.sha256(e2e_log.read_bytes()).hexdigest(),
        "e2e_record_path": str(e2e_log.with_suffix(".record.json").resolve()),
        "e2e_record_sha256": hashlib.sha256(
            e2e_log.with_suffix(".record.json").read_bytes()
        ).hexdigest(),
        "observation_schema": "f1-observation-v1",
        "primary_config_name": "realworld_dummy_f1_peg_sac_cnn_async",
        "repository_url": EXPECTED_ORIGIN,
        "rlinf_commit": subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "schema_version": "phase1-rlinf-handoff-v1",
        "transition_schema": "f1-transition-v1",
        "unit_command": [".venv/bin/pytest", "tests/unit_tests/f1", "-q"],
        "unit_log_path": str(unit_log.resolve()),
        "unit_log_sha256": hashlib.sha256(unit_log.read_bytes()).hexdigest(),
        "unit_record_path": str(unit_log.with_suffix(".record.json").resolve()),
        "unit_record_sha256": hashlib.sha256(
            unit_log.with_suffix(".record.json").read_bytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize("mutator", ["dirty", "missing-origin", "wrong-origin"])
def test_writer_rejects_untrusted_repository_provenance(
    tmp_path: Path, mutator: str
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    if mutator == "dirty":
        (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    elif mutator == "missing-origin":
        _run_git(repository, "remote", "remove", "origin")
    else:
        _run_git(
            repository, "remote", "set-url", "origin", "https://example.invalid/RLinf"
        )
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)

    with pytest.raises(ValueError, match="(clean|origin)"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            tmp_path / "handoff.json",
        )


def test_writer_rejects_a_non_commit_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)
    original_git_output = writer._git_output

    def invalid_head(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "not-a-commit"
        return original_git_output(root, *args)

    monkeypatch.setattr(writer, "_git_output", invalid_head)
    with pytest.raises(ValueError, match="HEAD"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            tmp_path / "handoff.json",
        )


@pytest.mark.parametrize(
    ("log_name", "content", "message"),
    [
        ("unit", "", "empty"),
        ("unit", "184 passed\n", "SHA-256"),
        ("e2e", "Global Step 2/2\nF1_E2E_PASSED\n", "SHA-256"),
        (
            "e2e",
            "Global Step 1/2\n"
            'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
            '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
            '"weight_sync_success_total":2}\n'
            "F1_E2E_PASSED\n",
            "SHA-256",
        ),
        (
            "e2e",
            "Global Step 2/2\n"
            'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
            '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
            '"unreviewed":1,"weight_sync_success_total":2}\n'
            "F1_E2E_PASSED\n",
            "SHA-256",
        ),
    ],
)
def test_writer_rejects_incomplete_or_weak_execution_logs(
    tmp_path: Path, log_name: str, content: str, message: str
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)
    (unit_log if log_name == "unit" else e2e_log).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            tmp_path / "handoff.json",
        )


def test_writer_rejects_a_symlink_log_or_invalid_controller_handoff(
    tmp_path: Path,
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)
    linked_log = tmp_path / "linked-unit.log"
    linked_log.symlink_to(unit_log)

    with pytest.raises(ValueError, match="non-symlink"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            linked_log,
            e2e_log,
            tmp_path / "handoff.json",
        )

    controller_handoff.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="controller handoff"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            tmp_path / "handoff.json",
        )


def test_writer_rejects_a_nonregular_log_and_a_log_replaced_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_writer()
    log_directory = tmp_path / "log-directory"
    log_directory.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink"):
        writer._read_regular_snapshot(log_directory, "unit log")

    unit_log = tmp_path / "f1-unit.log"
    unit_log.write_text("stable\n", encoding="utf-8")
    original_read = writer.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if not replaced:
            replaced = True
            unit_log.write_text("replaced\n", encoding="utf-8")
        return chunk

    monkeypatch.setattr(writer.os, "read", replace_after_first_read)
    with pytest.raises(ValueError, match="changed while being read"):
        writer._read_regular_snapshot(unit_log, "unit log")


def test_writer_rejects_unsafe_output_and_preserves_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path, repository)
    unignored_output = repository / "phase1-rlinf-handoff.json"

    with pytest.raises(ValueError, match="ignored"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            unignored_output,
        )

    outside = tmp_path / "outside.json"
    symlink_output = tmp_path / "symlink.json"
    symlink_output.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            symlink_output,
        )

    tracked_output = repository / "tracked-handoff.json"
    tracked_output.write_text("old\n", encoding="utf-8")
    _run_git(repository, "add", tracked_output.name)
    _run_git(repository, "commit", "-m", "tracked handoff")
    with pytest.raises(ValueError, match="tracked"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            tracked_output,
        )

    unit_log, e2e_log = _write_logs(tmp_path, repository)

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(writer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            unit_log,
            e2e_log,
            outside,
        )
    assert not outside.exists()


def test_runner_binds_fixed_commands_to_records_that_writer_verifies(
    tmp_path: Path,
) -> None:
    """Catch records that are not emitted by a fixed-command runner at one commit."""

    assert RUNNER_SCRIPT.is_file(), "the phase-one evidence runner must exist"
    repository = _make_repository(tmp_path)
    pytest_path = repository / ".venv" / "bin"
    pytest_path.mkdir(parents=True)
    test_runner = pytest_path / "pytest"
    test_runner.write_text(
        "#!/bin/sh\nprintf '================ 5 passed in 0.01s ============\\n'\n",
        encoding="utf-8",
    )
    test_runner.chmod(0o755)
    e2e_runner = repository / "tests" / "e2e_tests" / "embodied"
    e2e_runner.mkdir(parents=True)
    (e2e_runner / "run_async.sh").write_text(
        "#!/bin/sh\n"
        'test "$REPO_PATH" = "$PWD" || exit 7\n'
        "printf 'Global Step: 2/2\\n'\n"
        "printf '%s\\n' "
        '\'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
        '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
        '"weight_sync_success_total":2}\'\n',
        encoding="utf-8",
    )
    (e2e_runner / "run_async.sh").chmod(0o755)
    _run_git(
        repository, "add", ".venv/bin/pytest", "tests/e2e_tests/embodied/run_async.sh"
    )
    _run_git(repository, "commit", "-m", "fixed evidence commands")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    for kind in ("unit", "e2e"):
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER_SCRIPT),
                kind,
                "--output-dir",
                str(evidence_dir),
                "--repository-root",
                str(repository),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    unit_record = evidence_dir / "f1-unit.record.json"
    e2e_record = evidence_dir / "f1-e2e.record.json"
    unit_payload = json.loads(unit_record.read_text(encoding="utf-8"))
    assert unit_payload["command"] == [".venv/bin/pytest", "tests/unit_tests/f1", "-q"]
    assert unit_payload["exit_code"] == 0
    assert unit_payload["result"] == {"pytest_passed": 5, "status": "passed"}
    assert unit_payload["rlinf_commit"] == _git_output(repository, "rev-parse", "HEAD")
    assert (
        unit_payload["log_sha256"]
        == hashlib.sha256((evidence_dir / "f1-unit.log").read_bytes()).hexdigest()
    )
    e2e_payload = json.loads(e2e_record.read_text(encoding="utf-8"))
    assert e2e_payload["command"] == [
        "bash",
        "tests/e2e_tests/embodied/run_async.sh",
        "realworld_f1_dummy_sac_cnn",
    ]
    assert e2e_payload["result"] == {
        "actor_loss": 0.25,
        "actor_update_count": 1,
        "critic_loss": 0.5,
        "critic_update_count": 1,
        "replay_buffer_size": 2,
        "status": "passed",
        "weight_sync_success_total": 2,
    }

    writer = _load_writer()
    controller_handoff = _make_controller_artifacts(tmp_path)
    output = tmp_path / "phase1-rlinf-handoff.json"
    _write_valid_handoff(
        writer,
        repository,
        controller_handoff,
        evidence_dir / "f1-unit.log",
        evidence_dir / "f1-e2e.log",
        output,
    )
    unit_payload["exit_code"] = 0.0
    unit_record.write_text(json.dumps(unit_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exit_code"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            evidence_dir / "f1-unit.log",
            evidence_dir / "f1-e2e.log",
            tmp_path / "bad-exit-code.json",
        )
    unit_payload["exit_code"] = 0
    unit_payload["rlinf_commit"] = "0" * 40
    unit_record.write_text(json.dumps(unit_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="commit"):
        _write_valid_handoff(
            writer,
            repository,
            controller_handoff,
            evidence_dir / "f1-unit.log",
            evidence_dir / "f1-e2e.log",
            tmp_path / "rejected.json",
        )


def test_runner_rejects_nonintegral_e2e_observables() -> None:
    """Catch evidence that coerces fractional update or replay counts to integers."""

    runner = _load_runner()
    assert (
        runner._extract_e2e_result(
            "Global Step: 2/2\n"
            'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1.5,'
            '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
            '"weight_sync_success_total":2}\n'
        )
        is None
    )


def test_writer_rejects_symlink_ancestor_and_parent_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch output writes escaping through an ancestor or a replaced parent."""

    writer = _load_writer()
    repository = _make_repository(tmp_path)
    ignored_dir = repository / "artifacts"
    outside_ancestor = tmp_path / "outside"
    outside_ancestor.mkdir()
    (outside_ancestor / "nested").mkdir()
    ignored_dir.mkdir()
    (ignored_dir / "link").symlink_to(outside_ancestor, target_is_directory=True)
    (repository / ".git" / "info" / "exclude").write_text(
        "artifacts/\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="symlink"):
        writer._validate_output_path(
            ignored_dir / "link" / "nested" / "handoff.json", repository
        )

    parent = tmp_path / "output-parent"
    parent.mkdir()
    outside = tmp_path / "outside-race"
    outside.mkdir()
    output = parent / "handoff.json"
    original_open = writer.os.open
    replaced = False

    def replace_parent_before_create(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if dir_fd is not None and flags & os.O_CREAT and not replaced:
            replaced = True
            parent.rename(tmp_path / "old-parent")
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(writer.os, "open", replace_parent_before_create)
    with pytest.raises(ValueError, match="parent.*changed"):
        writer._atomic_write(output, {"schema_version": "test"})
    assert not (outside / "handoff.json").exists()

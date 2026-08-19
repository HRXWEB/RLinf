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
from pathlib import Path
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WRITER_SCRIPT = PROJECT_ROOT / "toolkits" / "f1" / "write_phase1_rlinf_handoff.py"
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


def _run_git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


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


def _write_logs(tmp_path: Path) -> tuple[Path, Path]:
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
    unit_log, e2e_log = _write_logs(tmp_path)
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
        "e2e_command": "bash tests/e2e_tests/embodied/run_async.sh realworld_f1_dummy_sac_cnn",
        "e2e_config_name": "realworld_f1_dummy_sac_cnn",
        "e2e_log_path": str(e2e_log.resolve()),
        "e2e_log_sha256": hashlib.sha256(e2e_log.read_bytes()).hexdigest(),
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
        "unit_command": ".venv/bin/pytest tests/unit_tests/f1 -q",
        "unit_log_path": str(unit_log.resolve()),
        "unit_log_sha256": hashlib.sha256(unit_log.read_bytes()).hexdigest(),
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
    unit_log, e2e_log = _write_logs(tmp_path)

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
    unit_log, e2e_log = _write_logs(tmp_path)
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
        ("unit", "184 passed\n", "F1_UNIT_TESTS_PASSED"),
        ("e2e", "Global Step 2/2\nF1_E2E_PASSED\n", "F1_E2E_EVIDENCE"),
        (
            "e2e",
            "Global Step 1/2\n"
            'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
            '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
            '"weight_sync_success_total":2}\n'
            "F1_E2E_PASSED\n",
            "Global Step",
        ),
        (
            "e2e",
            "Global Step 2/2\n"
            'F1_E2E_EVIDENCE={"actor_loss":0.25,"actor_update_count":1,'
            '"critic_loss":0.5,"critic_update_count":1,"replay_buffer_size":2,'
            '"unreviewed":1,"weight_sync_success_total":2}\n'
            "F1_E2E_PASSED\n",
            "unrecognized",
        ),
    ],
)
def test_writer_rejects_incomplete_or_weak_execution_logs(
    tmp_path: Path, log_name: str, content: str, message: str
) -> None:
    writer = _load_writer()
    repository = _make_repository(tmp_path)
    controller_handoff = _make_controller_artifacts(tmp_path)
    unit_log, e2e_log = _write_logs(tmp_path)
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
    unit_log, e2e_log = _write_logs(tmp_path)
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

    unit_log, _ = _write_logs(tmp_path)
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
    unit_log, e2e_log = _write_logs(tmp_path)
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

    def fail_replace(
        source: str | bytes | os.PathLike[str],
        destination: str | bytes | os.PathLike[str],
    ) -> None:
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

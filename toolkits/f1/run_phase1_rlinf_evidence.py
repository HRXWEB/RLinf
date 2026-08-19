# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run one fixed F1 phase-one gate and atomically record its execution evidence."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

UNIT_COMMAND = [".venv/bin/pytest", "tests/unit_tests/f1", "-q"]
E2E_COMMAND = [
    "bash",
    "tests/e2e_tests/embodied/run_async.sh",
    "realworld_f1_dummy_sac_cnn",
]
_PYTEST_PASSED = re.compile(r"(?m)^=+\s+(\d+) passed(?:,.*)?\s+in [^\n]+\s+=+$")
_GLOBAL_STEP = re.compile(r"\bGlobal Step\s*:?\s*2\s*/\s*2\b")
E2E_EVIDENCE_PREFIX = "F1_E2E_EVIDENCE="
E2E_FIELDS = frozenset(
    {
        "actor_loss",
        "actor_update_count",
        "critic_loss",
        "critic_update_count",
        "replay_buffer_size",
        "weight_sync_success_total",
    }
)


def _load_writer() -> ModuleType:
    """Load writer helpers without relying on the repository import path."""

    writer_path = Path(__file__).with_name("write_phase1_rlinf_handoff.py")
    spec = importlib.util.spec_from_file_location(
        "write_phase1_rlinf_handoff", writer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load phase-one handoff writer helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timestamp() -> str:
    """Return one second-precision UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_e2e_result(log_text: str) -> dict[str, object] | None:
    """Extract the required success observables from real command output."""

    if _GLOBAL_STEP.search(log_text) is None:
        return None
    evidence_lines = [
        line.removeprefix(E2E_EVIDENCE_PREFIX)
        for line in log_text.splitlines()
        if line.startswith(E2E_EVIDENCE_PREFIX)
    ]
    if len(evidence_lines) == 1:
        try:
            evidence = json.loads(evidence_lines[0])
        except json.JSONDecodeError:
            return None
        return _validated_e2e_result(evidence)
    return _extract_tensorboard_result()


def _validated_e2e_result(evidence: object) -> dict[str, object] | None:
    """Validate one extracted E2E evidence object without coercing its schema."""

    if not isinstance(evidence, dict) or frozenset(evidence) != E2E_FIELDS:
        return None
    integer_fields = (
        "replay_buffer_size",
        "actor_update_count",
        "critic_update_count",
        "weight_sync_success_total",
    )
    if any(
        isinstance(evidence[field], bool) or not isinstance(evidence[field], int)
        for field in integer_fields
    ):
        return None
    if any(
        isinstance(evidence[field], bool)
        or not isinstance(evidence[field], (int, float))
        or not math.isfinite(float(evidence[field]))
        for field in ("actor_loss", "critic_loss")
    ):
        return None
    replay_size = evidence["replay_buffer_size"]
    actor_updates = evidence["actor_update_count"]
    critic_updates = evidence["critic_update_count"]
    sync_count = evidence["weight_sync_success_total"]
    actor_loss = float(evidence["actor_loss"])
    critic_loss = float(evidence["critic_loss"])
    if replay_size < 1 or actor_updates < 1 or critic_updates < 1 or sync_count < 2:
        return None
    return {
        "actor_loss": actor_loss,
        "actor_update_count": actor_updates,
        "critic_loss": critic_loss,
        "critic_update_count": critic_updates,
        "replay_buffer_size": replay_size,
        "status": "passed",
        "weight_sync_success_total": sync_count,
    }


def _find_scalar_tag(tags: list[str], suffix: str) -> str | None:
    """Return the sole scalar tag ending in ``suffix`` when it is unambiguous."""

    matches = [tag for tag in tags if tag.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _extract_tensorboard_result() -> dict[str, object] | None:
    """Extract E2E observables from the run's TensorBoard event files on neo."""

    run_dir = os.environ.get("F1_RUN_DIR")
    if not run_dir:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    event_paths = sorted(Path(run_dir).rglob("events.out.tfevents.*"))
    if not event_paths:
        return None
    scalar_values: dict[str, list[object]] = {}
    try:
        for event_path in event_paths:
            accumulator = EventAccumulator(str(event_path))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                scalar_values.setdefault(tag, []).extend(accumulator.Scalars(tag))
    except (OSError, ValueError):
        return None
    actor_tag = _find_scalar_tag(list(scalar_values), "sac/actor_loss")
    critic_tag = _find_scalar_tag(list(scalar_values), "sac/critic_loss")
    sync_tag = _find_scalar_tag(list(scalar_values), "weight_sync_success_total")
    replay_tags = [tag for tag in scalar_values if "replay_buffer" in tag]
    if not actor_tag or not critic_tag or not sync_tag or not replay_tags:
        return None
    actor_events = scalar_values[actor_tag]
    critic_events = scalar_values[critic_tag]
    sync_events = scalar_values[sync_tag]
    replay_events = [event for tag in replay_tags for event in scalar_values[tag]]
    if not actor_events or not critic_events or not sync_events or not replay_events:
        return None
    evidence = {
        "actor_loss": actor_events[-1].value,
        "actor_update_count": len(actor_events),
        "critic_loss": critic_events[-1].value,
        "critic_update_count": len(critic_events),
        "replay_buffer_size": max(int(event.value) for event in replay_events),
        "weight_sync_success_total": int(sync_events[-1].value),
    }
    return _validated_e2e_result(evidence)


def _result_for(kind: str, exit_code: int, log_text: str) -> dict[str, object]:
    """Return a success result only when the fixed command's output proves it."""

    if exit_code != 0:
        return {"status": "failed"}
    if kind == "unit":
        matches = _PYTEST_PASSED.findall(log_text)
        if len(matches) != 1 or int(matches[0]) < 1:
            return {"status": "failed"}
        return {"pytest_passed": int(matches[0]), "status": "passed"}
    result = _extract_e2e_result(log_text)
    return result if result is not None else {"status": "failed"}


def run_gate(kind: str, output_dir: Path, repository_root: Path | None = None) -> int:
    """Execute one fixed gate and write its log plus digest-bound record.

    The record is a local execution attestation, not a cryptographic signature.
    It is trustworthy only to the extent that this committed runner and its
    clean checkout are trusted on the host that performed the execution.
    """

    writer = _load_writer()
    root, commit = writer._validate_repository(
        repository_root or Path(__file__).resolve().parents[2]
    )
    command = UNIT_COMMAND if kind == "unit" else E2E_COMMAND
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise ValueError("evidence output directory must already exist")
    started_at = _timestamp()
    try:
        command_environment = os.environ.copy()
        command_environment["REPO_PATH"] = str(root)
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=command_environment,
        )
        exit_code = completed.returncode
        log_snapshot = completed.stdout
    except OSError as error:
        exit_code = 127
        log_snapshot = str(error).encode("utf-8", errors="replace")
    finished_at = _timestamp()
    log_name = f"f1-{kind}.log"
    record_name = f"f1-{kind}.record.json"
    log_path = output_path / log_name
    record_path = output_path / record_name
    result = _result_for(
        kind, exit_code, log_snapshot.decode("utf-8", errors="replace")
    )
    record = {
        "command": command,
        "exit_code": exit_code,
        "finished_at": finished_at,
        "kind": kind,
        "log_filename": log_name,
        "log_sha256": hashlib.sha256(log_snapshot).hexdigest(),
        "repository_url": writer.EXPECTED_ORIGIN,
        "result": result,
        "rlinf_commit": commit,
        "schema_version": "f1-execution-record-v1",
        "started_at": started_at,
    }
    writer._atomic_write_bytes(log_path, log_snapshot)
    writer._atomic_write_bytes(
        record_path, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    )
    return 0 if result.get("status") == "passed" else 2


def main() -> int:
    """Run one required F1 gate in a clean checkout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("unit", "e2e"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    try:
        return run_gate(args.kind, args.output_dir, args.repository_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

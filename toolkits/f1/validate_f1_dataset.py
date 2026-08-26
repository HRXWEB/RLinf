#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate F1 replay descriptors and JSONL transitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rlinf.data.embodied.f1_schema import (
    validate_f1_replay_descriptor,
    validate_f1_transition,
)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_transition_jsonl(path: Path, descriptor: dict[str, Any]) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"transition line {line_number} must be valid JSON"
                    ) from error
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"transition line {line_number} must be a JSON object"
                    )
                validate_f1_transition(payload, descriptor)
                count += 1
    except OSError as error:
        raise ValueError(f"transitions file is unreadable: {path}") from error
    return count


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--transitions", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without writing, networking, ROS, or Ray.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the F1 dataset validator CLI."""

    args = build_parser().parse_args(argv)
    if not args.validate_only:
        raise ValueError("only --validate-only mode is supported")

    descriptor = validate_f1_replay_descriptor(
        _load_json_object(args.descriptor, "descriptor")
    )

    transition_count = 0
    if args.transitions is not None:
        transition_count = _validate_transition_jsonl(args.transitions, descriptor)

    print(f"validated {transition_count} transitions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error

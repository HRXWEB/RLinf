# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tests for offline BC warmup in the async embodied runner."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner


class _Result:
    def wait(self):
        return [{"bc/loss": 0.25, "bc/action_mae": 0.1}]


class _Actor:
    def __init__(self):
        self.calls = 0

    def run_bc_warmup(self):
        self.calls += 1
        return _Result()


class _MetricLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((metrics, step))


def test_bc_warmup_trains_logs_and_checkpoints_before_online_run() -> None:
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"algorithm": {"bc_warmup_updates": 10}, "runner": {"resume_dir": None}}
    )
    runner.actor = _Actor()
    runner.metric_logger = _MetricLogger()
    runner.global_step = 0
    runner.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    checkpoints = []
    runner._save_checkpoint = lambda: checkpoints.append(runner.global_step)

    runner._run_bc_warmup()

    assert runner.actor.calls == 1
    assert runner.metric_logger.records == [
        ({"train/bc/loss": 0.25, "train/bc/action_mae": 0.1}, 0)
    ]
    assert checkpoints == [0]


def test_bc_warmup_is_skipped_when_resuming() -> None:
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "algorithm": {"bc_warmup_updates": 10},
            "runner": {"resume_dir": "/tmp/global_step_5"},
        }
    )
    runner.actor = _Actor()

    runner._run_bc_warmup()

    assert runner.actor.calls == 0

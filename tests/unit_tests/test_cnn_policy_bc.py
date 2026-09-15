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

"""Tests for behavior-cloning helpers used by the CNN SAC policy."""

import pytest
import torch

from rlinf.models.embodiment.cnn_policy.bc import (
    distributed_minimum_transitions,
    resolve_bc_batch_size,
    squashed_gaussian_bc_loss,
)


def test_squashed_gaussian_bc_loss_prefers_matching_action_mean() -> None:
    actions = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    matching_mean = torch.atanh(actions)
    wrong_mean = torch.zeros_like(actions)
    logstd = torch.full_like(actions, -2.0)

    matching_loss, matching_mae = squashed_gaussian_bc_loss(
        matching_mean, logstd, actions
    )
    wrong_loss, wrong_mae = squashed_gaussian_bc_loss(wrong_mean, logstd, actions)

    assert matching_loss < wrong_loss
    assert matching_mae == 0.0
    assert wrong_mae > 0.0


def test_squashed_gaussian_bc_loss_is_finite_at_action_bounds() -> None:
    actions = torch.tensor([[1.0, -1.0]], dtype=torch.float32)

    loss, action_mae = squashed_gaussian_bc_loss(
        torch.zeros_like(actions), torch.zeros_like(actions), actions
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(torch.as_tensor(action_mae))


def test_resolve_bc_batch_size_shrinks_to_available_micro_batches() -> None:
    assert resolve_bc_batch_size(64, 18, micro_batch_size=4, world_size=1) == 16


def test_resolve_bc_batch_size_rejects_too_few_transitions() -> None:
    with pytest.raises(ValueError, match="at least one actor micro-batch"):
        resolve_bc_batch_size(64, 3, micro_batch_size=4, world_size=1)


def test_distributed_minimum_transitions_uses_the_smallest_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce_to_remote_minimum(value, op):
        assert op == torch.distributed.ReduceOp.MIN
        value.fill_(18)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_to_remote_minimum)

    assert distributed_minimum_transitions(70, device=torch.device("cpu")) == 18


def test_distributed_minimum_transitions_propagates_an_empty_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce_to_empty_remote_rank(value, op):
        assert op == torch.distributed.ReduceOp.MIN
        value.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_to_empty_remote_rank)

    minimum = distributed_minimum_transitions(70, device=torch.device("cpu"))
    assert minimum == 0
    with pytest.raises(ValueError, match="at least one actor micro-batch"):
        resolve_bc_batch_size(64, minimum, micro_batch_size=4, world_size=2)

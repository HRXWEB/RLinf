# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Behavior-cloning losses for bounded Gaussian CNN policies."""

import torch
from torch.distributions.normal import Normal


def distributed_minimum_transitions(
    local_transitions: int,
    *,
    device: torch.device,
) -> int:
    """Return the smallest demonstration shard size across actor ranks."""

    count = torch.tensor(local_transitions, dtype=torch.int64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MIN)
    return int(count.item())


def resolve_bc_batch_size(
    requested_batch_size: int,
    available_transitions_per_rank: int,
    *,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Return a valid global BC batch size for the local demo shard."""

    divisor = micro_batch_size * world_size
    if requested_batch_size <= 0 or requested_batch_size % divisor != 0:
        raise ValueError(
            "bc_warmup_batch_size must be positive and divisible by "
            "actor.micro_batch_size * actor world size."
        )
    available_micro_batches = available_transitions_per_rank // micro_batch_size
    if available_micro_batches < 1:
        raise ValueError(
            "BC warmup requires at least one actor micro-batch of demonstration "
            "transitions on every rank."
        )
    requested_per_rank = requested_batch_size // world_size
    effective_per_rank = min(
        requested_per_rank,
        available_micro_batches * micro_batch_size,
    )
    return effective_per_rank * world_size


def squashed_gaussian_bc_loss(
    action_mean: torch.Tensor,
    action_logstd: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Return demonstration NLL and deterministic-action MAE.

    The CNN SAC actor represents normalized actions by applying ``tanh`` to a
    Gaussian sample. Demonstration actions therefore need the inverse transform
    before evaluating their likelihood.
    """

    bounded_actions = target_actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
    raw_actions = torch.atanh(bounded_actions)
    distribution = Normal(action_mean, torch.exp(action_logstd))
    log_prob = distribution.log_prob(raw_actions) - torch.log(
        1.0 - bounded_actions.square() + epsilon
    )
    loss = -log_prob.sum(dim=-1).mean()
    predicted_actions = torch.tanh(action_mean)
    action_mae = torch.mean(torch.abs(predicted_actions - target_actions)).item()
    return loss, action_mae

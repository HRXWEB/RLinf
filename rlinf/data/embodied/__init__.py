# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Embodied data contracts."""

from rlinf.data.embodied.f1_schema import (
    F1_ACTION_LAYOUT,
    F1_ACTION_SCHEMA_VERSION,
    F1_DATASET_MANIFEST_SCHEMA_VERSION,
    F1_OBSERVATION_SCHEMA_VERSION,
    F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION,
    F1_STATE_LAYOUT,
    F1_TRAJECTORY_FIELD_MAPPING,
    F1_TRANSITION_SCHEMA_VERSION,
    build_f1_replay_descriptor,
    f1_action_bounds_from_motion_envelope,
    validate_f1_dataset_manifest,
    validate_f1_replay_descriptor,
    validate_f1_replay_descriptor_compatibility,
    validate_f1_transition,
)

__all__ = [
    "F1_ACTION_LAYOUT",
    "F1_ACTION_SCHEMA_VERSION",
    "F1_DATASET_MANIFEST_SCHEMA_VERSION",
    "F1_REPLAY_DESCRIPTOR_SCHEMA_VERSION",
    "F1_OBSERVATION_SCHEMA_VERSION",
    "F1_STATE_LAYOUT",
    "F1_TRAJECTORY_FIELD_MAPPING",
    "F1_TRANSITION_SCHEMA_VERSION",
    "build_f1_replay_descriptor",
    "f1_action_bounds_from_motion_envelope",
    "validate_f1_dataset_manifest",
    "validate_f1_replay_descriptor",
    "validate_f1_replay_descriptor_compatibility",
    "validate_f1_transition",
]

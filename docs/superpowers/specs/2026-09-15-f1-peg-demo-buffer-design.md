# F1 Right-Arm Peg Demonstration Buffer Design

## Goal

Convert the labeled F1 right-arm peg-insertion MCAP recordings into a
validated RLinf `TrajectoryReplayBuffer` for RLPD real-robot training, while
preserving the raw recordings and excluding failed, held-out, and post-success
return-to-origin motion from the demonstration buffer.

## Inputs and outputs

The source dataset is
`/data/hrx/datasets/f1_single_arm_peg_insertion` on `50-90`. The immutable
backup is
`/data/hrx/datasets/f1_single_arm_peg_insertion_backup_20260915_before_outcome_filter`.
`episode_outcomes.csv` maps the original 71 episode identifiers to `x`, `0`,
`1`, or `s` outcomes.

Only the 35 `outcome=1` episodes enter the demonstration buffer. The six
`outcome=0` episodes remain available for later offline-replay experiments but
do not enter the expert buffer. The two `outcome=s` episodes remain held out.
The 28 `outcome=x` episodes remain recoverable from the immutable backup.

The converter writes to a new output directory and never modifies MCAP files.
The output contains RLinf `metadata.json`, `trajectory_index.json`, individual
`trajectory_*.pt` files, a conversion manifest, scale statistics, truncation
diagnostics, and validation results.

## Runtime environment

The processing environment lives at
`/home/hrx/Archive/striding/repos/rlinf/.venv-f1-data` and is managed with
`uv`. It is independent of the training environment `.venv-f1`. The minimum
dependencies are the RLinf project, PyTorch, NumPy, PyYAML, OpenCV headless,
`mcap`, and `mcap-ros2-support`.

The reusable CLI and its focused helpers live under `toolkits/f1/`. Unit tests
live under `tests/unit_tests/f1/`.

## MCAP sources and canonical units

The converter reads:

- `/camera/head/color/image_raw/compressed` for head RGB images.
- `/state/right_arm/tcp_pos` for measured right-arm TCP state.

Bag log timestamps provide the common time domain. Message header timestamps
are recorded as diagnostics when available but do not participate in alignment.
The conversion report records per-topic rate, gaps, and timestamp monotonicity.

TCP state is represented as `[x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg]` in
the `right_arm_tcp_pose` frame. The converter validates field count, finite
values, units, and plausible ranges before processing an episode. Euler angles
are unwrapped before interpolation and differencing, then expressed as wrapped
shortest-path deltas in degrees.

## Successful-segment detection

Raw recordings include post-success motion back to the origin. Segmentation
therefore runs on the original high-rate gripper-state stream before 10 Hz
resampling.

Every inspected successful episode begins with the right gripper closed near
100 and contains a distinct release toward zero after insertion. The converter
uses `/motion_ctl/gripper/right/state` as the authoritative marker. It finds the
first transition after a value of at least 80 to a value of at most 20 that
remains at most 20 for at least 0.1 seconds. The retained endpoint is the last
TCP/image time before that sustained release. The corresponding
`/motion_ctl/gripper/right` command transition is recorded as a diagnostic but
does not replace measured state.

The probe found this marker in 35 of 35 successful recordings. Release state
persisted for at least 0.64 seconds, and command-to-state delay was at most
22.1 ms. The marker removed between 2.57 and 7.37 seconds of post-success
motion per episode.

The report records each episode's endpoint timestamp, retained and removed
durations, release hold duration, command-to-state delay, and endpoint pose.
Episodes without a sustained release are rejected for manual review. TCP
distance and later retreat remain diagnostics only and can never trigger a
cut, preventing a temporary pre-success retreat from truncating a trajectory.

## Reward target calibration

The previously configured target pose is systematically displaced from the
35 human-confirmed success endpoints and must not be used for this dataset.
The robust component-wise median of the gripper-release endpoints becomes the
dense-reward reference pose:

`[0.4839348835, -0.2951324388, 0.0669570176, 134.6285661,
5.5589248, 87.4288218]`, in metres/degrees.

The conversion report retains per-axis spread and target-distance diagnostics.
The observed sample standard deviations are approximately 11.64, 11.91, and
10.62 mm for XYZ, and 4.84, 2.25, and 2.74 degrees for RX/RY/RZ. This calibrated
pose is a shaping center, not the definition of task success; human outcome
labels remain authoritative for terminal reward.

## Temporal alignment and actions

After successful-segment detection, each episode is resampled on a strict 10 Hz
grid starting from the first usable head-image timestamp:

- The image at a grid point is the nearest image within 50 ms.
- TCP XYZ is linearly interpolated at the grid point.
- Unwrapped TCP Euler angles are linearly interpolated at the grid point.
- A missing image invalidates transitions adjacent to that grid point; the
  converter never bridges an observation gap.
- `curr_obs[k]` uses state and image at `t_k`.
- `next_obs[k]` uses state and image at `t_(k+1)`.
- Physical action is the shortest-path TCP pose delta from `t_k` to
  `t_(k+1)`.

The last observation is used only as `next_obs`; it does not produce an extra
action.

## Image preprocessing

Each decoded 720-by-1280 head image is center-cropped by removing 280 columns
from both sides, producing 720 by 720. The crop is then resized bilinearly to
the configured policy image shape, initially 128 by 128, and stored as uint8
RGB in both `curr_obs` and `next_obs`.

The online F1 environment must use exactly the same center-crop-before-resize
operation. Tests cover crop bounds, RGB ordering, dtype, output shape, and
offline/online preprocessing parity. Raw full-resolution images remain in the
MCAP backup so the buffer can be regenerated at another policy resolution.

## Action scale

Scale statistics use physical 10 Hz actions from retained segments of accepted
successful episodes only. Return-to-origin motion, failed episodes, held-out
episodes, and rejected segmentations do not contribute.

The report includes per-axis signed min/max, absolute p50/p90/p95/p99/p99.5,
maximum, and candidate saturation rates. The deployed scale uses one shared
position scale for XYZ and one shared orientation scale for roll/pitch/yaw,
matching `F1RobotConfig.action_scale`. Each is the p99 of the pooled absolute
component deltas in its group. Normalized actions equal physical delta divided
by the group scale and are clipped to `[-1, 1]`. The report records pre-clip and
post-clip saturation rates, and the generated configuration values remain
reviewable rather than silently editing the training YAML.

## Reward and episode boundary semantics

Dense reward reuses the same pure calculation as
`RightArmPegInsertionEnv`: position and orientation scores, their configured
weights, potential difference, and step penalty. Shared code prevents an
offline/online formula fork.

The calibrated target pose supplies dense shaping but is not the ground-truth
definition of insertion success. Human `outcome=1` is authoritative. The final
retained transition of every accepted successful demonstration receives the
configured `+5` success bonus and has `termination=True`, `done=True`, and
`truncation=False`. Earlier transitions have all three flags false. The report
also evaluates the geometric success predicate for diagnostics without using
it to revoke a human success label.

## Replay-buffer schema

Each source episode becomes one RLinf `Trajectory` with batch dimension one:

- `actions`: float32 `[T, 1, 6]`, normalized right-TCP deltas.
- `intervene_flags`: bool `[T, 1, 6]`, all true for demonstrations.
- `rewards`: float32 `[T, 1]`.
- `terminations`, `truncations`, and `dones`: bool `[T, 1]`.
- `forward_inputs["action"]`: the same normalized action tensor.
- `curr_obs["states"]` and `next_obs["states"]`: float32 `[T, 1, 6]`.
- `curr_obs["main_images"]` and `next_obs["main_images"]`: uint8
  `[T, 1, 128, 128, 3]`.

The buffer uses RLinf's PT trajectory format. Provenance fields that do not
fit the sampled tensor schema remain in the external conversion manifest,
keyed by generated trajectory ID and source episode ID.

## Failure handling and idempotence

Conversion validates the full dataset before publishing an output. It writes
to a temporary sibling directory, closes the replay buffer to flush async
writes, validates it, and atomically renames it to the requested final path.
An existing final path is never overwritten without an explicit CLI flag.

Malformed MCAP data, missing topics, non-monotonic timestamps, implausible TCP
values, image decode failures, alignment gaps, and ambiguous segmentation are
reported per episode. A rejected episode cannot partially enter the published
buffer. The CLI supports an analysis-only mode that computes segmentation and
scale reports without writing a replay buffer.

## Validation

Unit tests use synthetic timestamped image and TCP streams to cover Euler wrap,
interpolation, missing-image gaps, return-motion truncation, scale calculation,
reward parity, terminal labels, and trajectory shapes.

Dataset validation requires:

- Exactly the expected successful episode IDs are considered.
- Every published trajectory has at least one transition.
- Tensor keys, dtypes, shapes, finite values, and time lengths are consistent.
- Actions are within `[-1, 1]`, with saturation reported.
- Each trajectory has exactly one final termination/done and no truncation.
- Reward recomputation matches stored rewards.
- `TrajectoryReplayBuffer` reloads from disk and samples successfully.
- Several first, middle, and final images are rendered for visual crop review.
- The published manifest records all accepted and rejected episodes and all
  conversion parameters.

## Training integration

The generated successful buffer becomes
`algorithm.demo_buffer.load_path` for the F1 right-arm RLPD configuration.
Online interaction continues to populate `algorithm.replay_buffer`. RLPD
updates mix expert demonstration samples with online replay samples. Failed
offline episodes are deliberately excluded from the expert buffer in this
phase.

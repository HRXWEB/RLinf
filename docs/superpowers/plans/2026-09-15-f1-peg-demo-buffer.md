# F1 Peg Demonstration Buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 35 human-confirmed F1 right-arm peg-insertion MCAP episodes into a validated 10 Hz RLinf demonstration replay buffer for RLPD.

**Architecture:** A reusable `toolkits/f1` converter parses typed ROS2 messages from MCAP, cuts each episode at the first sustained measured gripper release, aligns head images and measured TCP poses on a 10 Hz grid, computes robust action scales in a first pass, and writes RLinf `Trajectory` objects in a second pass. Shared pure image and reward helpers keep offline conversion identical to online F1 behavior.

**Tech Stack:** Python 3.12, uv, MCAP ROS2 support, NumPy, OpenCV, PyTorch, RLinf `TrajectoryReplayBuffer`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-f1-peg-demo-buffer-design.md`

## Global Constraints

- Work only from the labeled dataset and its immutable backup on `50-90`.
- Never modify source MCAP files or overwrite an existing final buffer.
- Use only `outcome=1` episodes for scale statistics and the demo buffer.
- Cut on the first measured right-gripper release from at least 80 to at most 20 sustained for at least 0.1 seconds.
- Resample at exactly 10 Hz and reject image matches farther than 50 ms.
- Crop 720-by-1280 head images to 720 by 720 before resizing to 128 by 128.
- Human success controls the terminal bonus and done flags; the calibrated pose is shaping-only.
- Canonical TCP units are metres and degrees; recorded vendor XYZ values are millimetres.
- Preserve all unrelated worktree changes.

---

### Task 1: Shared F1 image preprocessing

**Files:**
- Modify: `rlinf/envs/realworld/f1/f1_robot_env.py`
- Modify: `tests/unit_tests/f1/test_f1_robot_env.py`

**Interfaces:**
- Produces: `center_crop_and_resize_rgb(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray`
- Consumed by: online F1 policy observations and Task 4's offline converter.

- [ ] **Step 1: Write failing preprocessing tests**

Add tests using a synthetic 720-by-1280 RGB image whose columns encode their source index. Assert that the result is uint8, has shape `(128, 128, 3)`, excludes columns before 280 and after 999, and is copied rather than aliased. Retain a test proving an already square image is resized without horizontal distortion.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/unit_tests/f1/test_f1_robot_env.py -q`

Expected: the new import or crop assertion fails because the helper does not exist and `_resize_policy_image` currently resizes the full rectangular frame.

- [ ] **Step 3: Implement the shared helper**

Add a module-level helper with this contract:

```python
def center_crop_and_resize_rgb(
    image: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Center-crop an RGB frame to a square and resize it."""
```

Validate `HWC`, three channels, uint8-compatible values, and positive output dimensions. Center-crop the longer spatial dimension, then use Pillow bilinear resizing. Make `F1RobotEnv._resize_policy_image` delegate to this helper.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit_tests/f1/test_f1_robot_env.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/realworld/f1/f1_robot_env.py tests/unit_tests/f1/test_f1_robot_env.py
git commit -s -m "feat(f1): center crop policy images"
```

### Task 2: Pure peg reward calculation

**Files:**
- Create: `rlinf/envs/realworld/f1/tasks/peg_reward.py`
- Modify: `rlinf/envs/realworld/f1/tasks/right_arm_peg_insertion_env.py`
- Create: `tests/unit_tests/f1/test_peg_reward.py`
- Modify: `tests/unit_tests/f1/test_peg_insertion_env.py`

**Interfaces:**
- Produces: `PegRewardConfig`, `peg_pose_metrics(pose, config)`, and `peg_transition_reward(previous_pose, next_pose, config, *, terminal_success)`.
- Consumed by: `RightArmPegInsertionEnv` and Task 4's converter.

- [ ] **Step 1: Write failing reward parity tests**

Cover wrapped Euler differences across ±180 degrees, position/orientation score calculation, potential difference, step penalty, and a terminal success bonus applied exactly once. Add an environment regression test that compares one environment reward with the pure helper for the same two poses.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit_tests/f1/test_peg_reward.py tests/unit_tests/f1/test_peg_insertion_env.py -q`

Expected: FAIL because the pure helper does not exist.

- [ ] **Step 3: Extract reward math without changing behavior**

Move the existing wrapped-angle, score, potential, and transition calculations into `peg_reward.py`. Keep success-hold state in the environment. The pure transition helper accepts `terminal_success`; it adds the configured bonus only when true and otherwise computes `next_potential - previous_potential - step_penalty`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit_tests/f1/test_peg_reward.py tests/unit_tests/f1/test_peg_insertion_env.py -q`

Expected: PASS with unchanged online reward behavior.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/realworld/f1/tasks/peg_reward.py rlinf/envs/realworld/f1/tasks/right_arm_peg_insertion_env.py tests/unit_tests/f1/test_peg_reward.py tests/unit_tests/f1/test_peg_insertion_env.py
git commit -s -m "refactor(f1): share peg reward calculation"
```

### Task 3: MCAP parsing, release detection, and 10 Hz alignment

**Files:**
- Create: `toolkits/f1/__init__.py`
- Create: `toolkits/f1/peg_demo_data.py`
- Create: `tests/unit_tests/f1/test_peg_demo_data.py`

**Interfaces:**
- Produces: `EpisodeStreams`, `AlignedEpisode`, `read_episode_mcap(path)`, `find_sustained_release(...)`, `align_episode(...)`, `action_scale_statistics(...)`.
- Consumed by: Task 4's CLI.

- [ ] **Step 1: Write synthetic failing tests**

Construct in-memory timestamp/value arrays and assert:

- A transient open sample does not cut.
- A `100 -> 0` state held for 0.1 seconds cuts immediately before release.
- A later re-close does not change the first cut.
- Missing release raises a typed conversion error.
- Millimetre XYZ converts to metres.
- Euler interpolation unwraps `179 -> -179` through 180, not zero.
- Nearest images beyond 50 ms invalidate adjacent transitions.
- Aligned actions equal consecutive 10 Hz pose differences.
- Pooled absolute p99 produces one XYZ scale and one Euler scale.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/unit_tests/f1/test_peg_demo_data.py -q`

Expected: FAIL because `toolkits.f1.peg_demo_data` does not exist.

- [ ] **Step 3: Implement typed stream parsing**

Read only these topics with `mcap_ros2.reader.read_ros2_messages`:

```python
HEAD_IMAGE_TOPIC = "/camera/head/color/image_raw/compressed"
RIGHT_TCP_TOPIC = "/state/right_arm/tcp_pos"
RIGHT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/right/state"
RIGHT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/right"
```

Use MCAP log time in integer nanoseconds. Validate strict TCP names
`x,y,z,rx,ry,rz`, convert XYZ by `0.001`, decode compressed images as RGB, and retain command timing only for diagnostics.

- [ ] **Step 4: Implement release detection and alignment**

Implement the thresholds from the global constraints. Start the 10 Hz grid at the first usable head image and stop strictly before release. Interpolate TCP state, unwrap Euler angles, match nearest images, and split runs at invalid image grid points so no transition crosses a gap. Reject an episode if no valid run remains.

- [ ] **Step 5: Implement scale statistics**

Return per-axis signed extrema and absolute p50/p90/p95/p99/p99.5/max, pooled XYZ/Euler percentiles, candidate scales, and pre-clip saturation counts. Reject zero or non-finite scales.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/unit_tests/f1/test_peg_demo_data.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add toolkits/f1/__init__.py toolkits/f1/peg_demo_data.py tests/unit_tests/f1/test_peg_demo_data.py
git commit -s -m "feat(f1): parse and align peg demonstrations"
```

### Task 4: Replay-buffer conversion CLI

**Files:**
- Create: `toolkits/f1/convert_peg_demos.py`
- Create: `tests/unit_tests/f1/test_convert_peg_demos.py`

**Interfaces:**
- Consumes: Task 1 image helper, Task 2 reward helper, and Task 3 aligned streams/statistics.
- Produces: CLI `python toolkits/f1/convert_peg_demos.py --dataset PATH --output PATH [--analysis-only]`.

- [ ] **Step 1: Write failing end-to-end fixture test**

Use a temporary synthetic dataset with an outcome manifest and mocked parsed streams. Assert that only `outcome=1` episodes enter the output, each trajectory has `[T,1,...]` tensors, the last transition alone has reward bonus/termination/done, every intervene flag is true, normalized actions are bounded, and provenance maps trajectory IDs to episode IDs.

- [ ] **Step 2: Run the CLI tests and verify failure**

Run: `pytest tests/unit_tests/f1/test_convert_peg_demos.py -q`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement two-pass analysis and conversion**

Pass one parses and aligns all successful episodes, rejects invalid episodes, calibrates the component-wise median success pose, and calculates physical action scales. `--analysis-only` stops after publishing CSV/JSON diagnostics and plot-ready arrays.

Pass two applies the scales, shared image transform, and shared reward helper, creates one `Trajectory` per episode, and writes it through `TrajectoryReplayBuffer(auto_save=True, trajectory_format="pt")`. Close the buffer before validation.

- [ ] **Step 4: Implement atomic publication and reports**

Write to `<output>.tmp-<pid>`, reject an existing final path unless `--overwrite` is explicitly passed, validate all files, then rename the temporary directory. Include `conversion_manifest.json`, `scale_statistics.json`, `episode_diagnostics.csv`, and `validation.json` in the published directory.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/unit_tests/f1/test_convert_peg_demos.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add toolkits/f1/convert_peg_demos.py tests/unit_tests/f1/test_convert_peg_demos.py
git commit -s -m "feat(f1): convert MCAP demos to replay buffer"
```

### Task 5: Calibrate the real task configuration

**Files:**
- Modify: `examples/embodiment/config/realworld_f1_right_arm_peg_rlpd_cnn_async.yaml`
- Modify: `tests/unit_tests/f1/test_f1_config.py`

**Interfaces:**
- Consumes: generated success-pose median and p99 action scales.
- Produces: deployable RLPD configuration with concrete target pose, action scales, and demo-buffer path.

- [ ] **Step 1: Write failing config assertions**

Assert that the target pose equals the generated calibrated median, action scales are finite positive values from `scale_statistics.json`, policy image shape remains 128 by 128, control period is 0.1 seconds, and the configured demo path matches the published buffer.

- [ ] **Step 2: Run config tests and verify failure**

Run: `pytest tests/unit_tests/f1/test_f1_config.py -q`

Expected: FAIL because action scales are still `???` and the target is the previous calibration.

- [ ] **Step 3: Update only generated calibration fields**

Copy the exact median pose, pooled absolute p99 XYZ scale, pooled absolute p99 orientation scale, and final buffer path from the validated conversion reports into the YAML. Record the report path and dataset manifest digest in comments.

- [ ] **Step 4: Run config tests**

Run: `pytest tests/unit_tests/f1/test_f1_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/embodiment/config/realworld_f1_right_arm_peg_rlpd_cnn_async.yaml tests/unit_tests/f1/test_f1_config.py
git commit -s -m "config(f1): calibrate right-arm peg RLPD"
```

### Task 6: Configure 50-90 and convert the real dataset

**Files:**
- Create remotely: `.venv-f1-data/` under `/home/hrx/Archive/striding/repos/rlinf`
- Create remotely: a new replay-buffer output directory under `/data/hrx/datasets/`

**Interfaces:**
- Consumes: the committed converter and the 35 successful MCAP episodes.
- Produces: validated physical dataset artifacts and evidence used by Task 5.

- [ ] **Step 1: Finish the isolated uv environment**

Use `/home/hrx/.local/bin/uv` with a data-specific cache while the training environment is active. Install the editable RLinf checkout, `mcap-ros2-support`, OpenCV headless, NumPy, SciPy, PyYAML, Matplotlib, and a compatible CPU-capable PyTorch. Do not alter `.venv-f1`.

- [ ] **Step 2: Run analysis-only conversion**

Run the CLI against `/data/hrx/datasets/f1_single_arm_peg_insertion`. Confirm 35/35 release markers, inspect scale statistics and saturation, and visually inspect first/middle/final frames plus several truncation plots.

- [ ] **Step 3: Run full conversion to a new path**

Publish the buffer to a versioned sibling directory such as
`/data/hrx/datasets/f1_single_arm_peg_insertion_demo_buffer_10hz_v1`.

- [ ] **Step 4: Validate with RLinf's loader**

Load the buffer using `TrajectoryReplayBuffer`, assert size 35, compare total samples with the manifest, sample at least 64 transitions, and verify keys, shapes, dtypes, finite rewards, bounded actions, and terminal counts.

- [ ] **Step 5: Record artifact hashes**

Compute SHA-256 for metadata, index, reports, and a sorted digest of trajectory files. Save these in the conversion report and retain command output as completion evidence.

### Task 7: Final regression verification

**Files:**
- No new files.

**Interfaces:**
- Consumes: all code, tests, configuration, and the real generated buffer.
- Produces: final completion evidence.

- [ ] **Step 1: Run focused F1 unit tests**

Run:

```bash
pytest tests/unit_tests/f1/test_f1_robot_env.py \
  tests/unit_tests/f1/test_peg_reward.py \
  tests/unit_tests/f1/test_peg_insertion_env.py \
  tests/unit_tests/f1/test_peg_demo_data.py \
  tests/unit_tests/f1/test_convert_peg_demos.py \
  tests/unit_tests/f1/test_f1_config.py -q
```

Expected: PASS.

- [ ] **Step 2: Run lint and formatting checks**

Run: `pre-commit run --files <all changed Python and YAML files>`

Expected: Ruff format/lint and repository checks pass.

- [ ] **Step 3: Re-run real buffer smoke validation**

Reload and sample the published buffer in `.venv-f1-data`; compare its report digest and counts with the analysis-only result.

- [ ] **Step 4: Review the final diff**

Run: `git diff HEAD~4 --check` and `git status --short`.

Expected: no whitespace errors and no unrelated files staged or modified by this work.
